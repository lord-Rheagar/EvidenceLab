"""Guided terminal workflows over the same services used by the command interface."""

import json
from pathlib import Path

from pydantic import ValidationError

from evidencelab.config import (
    AppConfig,
    ConfigError,
    Depth,
    ModelProfile,
    Reranker,
    load_config,
    resolve_profile,
    resolve_services,
)
from evidencelab.credentials import profile_slot, read_credential, service_slot
from evidencelab.evidence import EvidenceError
from evidencelab.fetching import normalize_url
from evidencelab.ingestion import ingest_sources
from evidencelab.interactive_setup import configure_profile
from evidencelab.providers import check_model
from evidencelab.reports import export_report, render_report
from evidencelab.reranking import RERANK_LIMITS
from evidencelab.research import create_job, execute_saved_research, validate_research_settings
from evidencelab.research_records import LIMITS, ResearchJob
from evidencelab.smoke import smoke_test
from evidencelab.storage import data_path, open_store
from evidencelab.terminal import Back, Cancelled, Terminal, clean


def local_path(value: str) -> Path:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return Path(value).expanduser().resolve()


def existing_path(value: str) -> bool | str:
    if not value.strip():
        return "Enter a file or directory path."
    try:
        return local_path(value).exists() or "Path does not exist; check spelling and quotes."
    except (OSError, ValueError):
        return "Enter a valid local path."


def public_url(value: str) -> bool | str:
    try:
        normalize_url(value)
        return True
    except EvidenceError:
        return "Enter a public HTTPS URL without embedded credentials."


class Workspace:
    def __init__(self, path: Path, terminal: Terminal) -> None:
        self.path, self.ui = path, terminal

    def profile(self, config: AppConfig) -> tuple[str, ModelProfile]:
        if not config.profiles:
            raise ConfigError(
                "No profile yet. Open Configure profiles to add a model and services."
            )
        name = self.ui.select(
            "Research profile",
            [
                (key, f"{key} - {profile.provider.value} / {profile.model}")
                for key, profile in config.profiles.items()
            ],
            config.default_profile,
        )
        return resolve_profile(config, name)

    def run(self) -> None:
        while True:
            try:
                config = load_config(self.path)
                subtitle = "No profile configured"
                if config.default_profile:
                    profile = config.profiles[config.default_profile]
                    subtitle = (
                        f"{config.default_profile} | {profile.provider.value} | {profile.model}"
                    )
                self.ui.heading("EvidenceLab", subtitle)
                if self.ui.plain:
                    self.ui.tell("Numbered menus. :back returns; :cancel cancels a text prompt.")
                action = self.ui.select(
                    "Workspace",
                    [
                        ("research", "Research a question"),
                        ("runs", "Browse saved runs"),
                        ("import", "Import documents locally"),
                        ("configure", "Configure profiles"),
                        ("readiness", "Check readiness"),
                        ("exit", "Exit"),
                    ],
                    "configure" if not config.profiles else "research",
                )
            except (Back, Cancelled):
                return
            if action == "exit":
                return
            try:
                if action == "research":
                    self.research()
                elif action == "runs":
                    self.browse()
                elif action == "import":
                    self.import_documents()
                elif action == "configure":
                    configure_profile(self.ui, self.path)
                elif action == "readiness":
                    self.readiness()
            except (Back, Cancelled):
                self.ui.tell(
                    "Returned to workspace. Unsaved setup changes were discarded; "
                    "saved runs remain available."
                )
            except (ConfigError, EvidenceError) as exc:
                self.ui.tell(exc, error=True)
            except ValidationError:
                self.ui.tell(
                    "Invalid settings. Review configuration before continuing.", error=True
                )

    def paid(self, title: str, details: str) -> bool:
        self.ui.heading(title)
        self.ui.tell(details)
        self.ui.tell("API charges may apply. Requests are not a dollar spending limit.")
        return (
            self.ui.select("Confirm paid action", [("start", title), ("back", "Back")], "back")
            == "start"
        )

    def sources(self, files: list[Path], urls: list[str]) -> None:
        while True:
            self.ui.heading("Sources", f"{len(files)} local paths | {len(urls)} URLs")
            action = self.ui.select(
                "Sources",
                [
                    ("file", "Add a file or directory"),
                    ("url", "Add a public URL"),
                    ("remove", "Remove a source"),
                    ("done", "Done"),
                ],
                "done",
            )
            if action == "done":
                return
            try:
                if action == "file":
                    path = local_path(self.ui.path("File or directory", validate=existing_path))
                    if path not in files:
                        files.append(path)
                elif action == "url":
                    url = normalize_url(self.ui.text("Public URL", validate=public_url))
                    if url not in urls:
                        urls.append(url)
                elif action == "remove" and (files or urls):
                    all_sources = [*[str(p) for p in files], *urls]
                    choice = self.choose(
                        "Remove source", [(str(i), v) for i, v in enumerate(all_sources)]
                    )
                    if choice is not None:
                        index = int(choice)
                        if index < len(files):
                            files.pop(index)
                        else:
                            urls.pop(index - len(files))
            except Back:
                continue

    def research(self) -> None:
        config = load_config(self.path)
        name, saved_profile = self.profile(config)
        profile = saved_profile.model_copy(deep=True)
        question = self.ui.text(
            "Research question",
            validate=lambda v: (
                (1 <= len(v.strip()) <= 8000) or "Enter a question of 1-8000 characters."
            ),
        ).strip()
        web = True
        files: list[Path] = []
        urls: list[str] = []
        while True:
            self.ui.heading("Review research", question)
            self.ui.tell(f"{name}: {profile.provider.value} / {profile.model}")
            self.ui.tell(
                f"Depth: {profile.depth.value} | Web: {'on' if web else 'off'} | "
                f"Reranker: {profile.reranker.value}"
            )
            self.ui.tell(f"Sources: {len(files)} local paths, {len(urls)} URLs")
            action = self.ui.select(
                "Research",
                [
                    ("start", "Start paid research"),
                    ("question", "Edit question"),
                    ("depth", "Change depth"),
                    ("sources", "Add or remove sources"),
                    ("web", "Toggle web discovery"),
                    ("ranking", "Change reranker"),
                    ("profile", "Change profile"),
                    ("cancel", "Cancel"),
                ],
                "start",
            )
            try:
                if action == "cancel":
                    return
                if action == "question":
                    question = self.ui.text(
                        "Research question",
                        question,
                        validate=lambda v: (
                            (1 <= len(v.strip()) <= 8000) or "Enter 1-8000 characters."
                        ),
                    ).strip()
                elif action == "depth":
                    profile.depth = Depth(
                        self.ui.select(
                            "Research depth",
                            [(d.value, d.value) for d in Depth],
                            profile.depth.value,
                        )
                    )
                elif action == "sources":
                    self.sources(files, urls)
                elif action == "web":
                    web = not web
                elif action == "ranking":
                    profile.reranker = Reranker(
                        self.ui.select(
                            "Reranker",
                            [("none", "None"), ("jev", "Jev (TypeSafe)")],
                            profile.reranker.value,
                        )
                    )
                elif action == "profile":
                    name, selected = self.profile(load_config(self.path))
                    profile = selected.model_copy(deep=True)
                elif action == "start":
                    services = resolve_services(load_config(self.path))
                    if not web and not files and not urls:
                        self.ui.tell(
                            "Add sources or enable web discovery before starting.", error=True
                        )
                        continue
                    try:
                        validate_research_settings(profile, services, web)
                        if (
                            profile.api_key_required
                            and not read_credential(
                                profile.api_key_env or "", profile_slot(name, profile)
                            ).value
                        ):
                            raise ConfigError("Model key missing. Configure this profile first.")
                    except ConfigError as exc:
                        self.ui.tell(exc, error=True)
                        if (
                            self.ui.select(
                                "Missing setup",
                                [("configure", "Configure this profile"), ("back", "Back")],
                                "back",
                            )
                            == "configure"
                        ):
                            configure_profile(self.ui, self.path, name)
                            name, profile = resolve_profile(load_config(self.path), name)
                            profile = profile.model_copy(deep=True)
                        continue
                    cap = LIMITS[profile.depth]
                    jev_cap = (
                        RERANK_LIMITS[profile.depth] if profile.reranker == Reranker.JEV else 0
                    )
                    if not self.paid(
                        "Start paid research",
                        f"Model: {profile.model}. "
                        f"Ceilings: {cap[4]} model requests, {cap[2] if web else 0} searches, "
                        f"{jev_cap} Jev requests. "
                        "Evidence goes to Elasticsearch and your model; enabled web queries go to "
                        "Tavily and Jev excerpts to TypeSafe.",
                    ):
                        continue
                    with open_store() as store:
                        job = create_job(store, question, name, profile, services, files, urls, web)
                    self.execute(job.run_id)
                    self.run_details(job.run_id)
                    return
            except Back:
                continue
            except (EvidenceError, ConfigError) as exc:
                self.ui.tell(exc, error=True)

    def execute(self, run_id: str) -> None:
        self.ui.tell(f"Saved run: {run_id}. Ctrl+C interrupts; the saved stage can be resumed.")
        services = resolve_services(load_config(self.path))
        try:
            with open_store() as store, self.ui.progress(run_id) as progress:

                def update(message: str) -> None:
                    saved = store.get_job(run_id)
                    progress(
                        f"{message} | {saved.model_calls} model, {saved.search_calls} web, "
                        f"{saved.rerank_calls} Jev | {len(saved.warnings)} warnings"
                    )

                job = execute_saved_research(store, run_id, services, update)
            self.ui.tell(f"Research {job.status}. Saved locally in {data_path()}")
        except (EvidenceError, ConfigError) as exc:
            self.ui.tell(exc, error=True)
            self.ui.tell(f"Saved run: {run_id}. Review Diagnostics before a paid resume.")

    def choose(self, label: str, choices: list[tuple[str, str]]) -> str | None:
        page = 0
        while True:
            options = [
                (key, clean(title)[:110]) for key, title in choices[page * 8 : (page + 1) * 8]
            ]
            if (page + 1) * 8 < len(choices):
                options.append(("__next__", "Next page"))
            if page:
                options.append(("__previous__", "Previous page"))
            options.append(("__back__", "Back"))
            try:
                action = self.ui.select(label, options)
            except Back:
                return None
            if action == "__back__":
                return None
            if action == "__next__":
                page += 1
            elif action == "__previous__":
                page -= 1
            else:
                return action

    def browse(self) -> None:
        query = self.ui.text("Filter runs by question or ID (blank shows all)").lower()
        while True:
            with open_store() as store:
                choices = []
                for run in store.runs():
                    summary = store.job_summary(run.run_id)
                    title = str(summary["question"]) if summary else "Imported evidence"
                    status = str(summary["status"]) if summary else run.status
                    if query in f"{title} {run.run_id}".lower():
                        choices.append(
                            (
                                run.run_id,
                                f"{status} | {run.depth.value} | {run.created_at[:10]} | {title}",
                            )
                        )
            if not choices:
                self.ui.tell("No matching saved runs.")
                return
            selected = self.choose("Saved runs", choices)
            if selected is None:
                return
            self.run_details(selected)

    def run_details(self, run_id: str) -> None:
        while True:
            with open_store() as store:
                summary = store.job_summary(run_id)
                job = store.get_job(run_id) if summary else None
                self.ui.heading("Saved run", job.question if job else run_id)
                self.ui.tell(f"Run: {run_id}")
                if job:
                    self.ui.tell(f"{job.status} | {job.stage} | {job.profile.model}")
                    self.ui.tell(
                        f"Requests: {job.model_calls} model, {job.search_calls} web, "
                        f"{job.rerank_calls} Jev | {len(job.warnings)} warnings"
                    )
                    if job.error:
                        self.ui.tell(job.error, error=True)
                options = []
                if job and job.status == "complete":
                    options += [
                        ("summary", "Read summary"),
                        ("report", "Read full report"),
                        ("citations", "Inspect citation passages"),
                        ("export", "Export report"),
                    ]
                elif job:
                    options.append(("resume", "Resume saved stage (paid)"))
                options += [("sources", "Browse source snapshots")]
                if job:
                    options += [
                        ("diagnostics", "View diagnostics"),
                        ("setup", "Configure profiles"),
                    ]
                    if job.warnings:
                        options.append(("warnings", "View source warnings"))
                options += [("back", "Back")]
                try:
                    action = self.ui.select("Run actions", options)
                    if action == "back":
                        return
                    if action == "sources":
                        sources = store.sources(run_id)
                        source_id = self.choose(
                            "Sources", [(s.source_id, s.title) for s in sources]
                        )
                        if source_id:
                            source = next(s for s in sources if s.source_id == source_id)
                            content = "\n\n".join(
                                f"{p.locator}\nPassage: {p.passage_id}\n{p.text}"
                                for p in store.passages(run_id)
                                if p.source_id == source_id
                            )
                            self.ui.view(source.title, f"{source.location}\n\n{content}")
                    elif job and action == "summary" and job.report:
                        self.ui.view("Summary", "\n\n".join(c.text for c in job.report.summary))
                    elif job and action == "report":
                        self.ui.view("Research report", render_report(store, job), markdown=True)
                    elif job and action == "citations" and job.report:
                        claims = [*job.report.summary, *job.report.findings, *job.report.tradeoffs]
                        index = self.choose(
                            "Claims", [(str(i), c.text) for i, c in enumerate(claims)]
                        )
                        if index is not None:
                            claim = claims[int(index)]
                            self.ui.view(
                                "Citation passages",
                                claim.text
                                + "\n\n"
                                + "\n\n".join(
                                    f"Passage: {c.passage_id}\n{c.quote}" for c in claim.citations
                                ),
                            )
                    elif job and action == "diagnostics":
                        self.ui.view(
                            "Diagnostics",
                            "\n\n".join(
                                json.dumps(e.model_dump(exclude_none=True), indent=2)
                                for e in job.diagnostics
                            )
                            or "No diagnostic events recorded.",
                        )
                    elif job and action == "warnings":
                        self.ui.view("Source warnings", "\n\n".join(job.warnings))
                    elif job and action == "export":
                        self.export(job)
                    elif action == "setup":
                        configure_profile(self.ui, self.path)
                    elif job and action == "resume":
                        self.ui.tell(
                            f"Consumed: {job.model_calls} model, {job.search_calls} web, "
                            f"{job.rerank_calls} Jev requests. Saved limits still apply."
                        )
                        if self.paid(
                            "Resume paid research",
                            f"Resume {job.stage} using saved model "
                            f"{job.profile.model}. Evidence will be sent to configured services.",
                        ):
                            self.execute(run_id)
                except Back:
                    continue
                except (EvidenceError, ConfigError) as exc:
                    self.ui.tell(exc, error=True)

    def export(self, job: ResearchJob) -> None:
        format = self.ui.select("Export format", [("markdown", "Markdown"), ("json", "JSON")])
        extension = "md" if format == "markdown" else "json"
        suggested = str(Path.cwd() / f"research-{job.run_id[:8]}.{extension}")
        while True:
            target = local_path(self.ui.path("Export path", suggested))
            if target.exists():
                self.ui.tell("That file exists and will not be overwritten.", error=True)
                suggested = str(target.with_name(target.stem + "-copy" + target.suffix))
                continue
            if not target.parent.exists():
                if (
                    self.ui.select(
                        "Directory is missing",
                        [("create", "Create directory"), ("back", "Choose another path")],
                        "back",
                    )
                    != "create"
                ):
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    self.ui.tell("Cannot create directory; choose another path.", error=True)
                    continue
            try:
                with open_store() as store, store.lock(job.run_id):
                    export_report(store, store.get_job(job.run_id), target, format)
                self.ui.tell(f"Saved: {target}")
                return
            except EvidenceError as exc:
                self.ui.tell(exc, error=True)

    def import_documents(self) -> None:
        files: list[Path] = []
        urls: list[str] = []
        self.sources(files, urls)
        if not files and not urls:
            return
        self.ui.tell(
            "This saves local snapshots without cloud indexing or model calls. "
            "Supplied URLs will be downloaded."
        )
        if (
            self.ui.select("Import", [("start", "Import locally"), ("cancel", "Cancel")], "cancel")
            != "start"
        ):
            return
        with open_store() as store:
            result = ingest_sources(store, files, urls)
        self.ui.tell(
            f"Saved run {result['run_id']}: {result['sources']} sources, "
            f"{result['passages']} passages"
        )
        if result["skipped"]:
            self.ui.view("Skipped inputs", json.dumps(result["skipped"], indent=2))

    def readiness(self) -> None:
        config = load_config(self.path)
        name, profile = self.profile(config)
        services = resolve_services(config)
        self.ui.heading(
            "Local readiness", "Credential presence only; connections have not been tested."
        )
        credential = read_credential(profile.api_key_env or "", profile_slot(name, profile))
        self.ui.tell(
            f"Model: {credential.source}"
            if profile.api_key_required
            else "Model: no authentication required"
        )
        self.ui.tell(f"Elasticsearch: {services.elasticsearch_url or 'not configured'}")
        self.ui.tell(f"Inference endpoint: {services.inference_id or 'not configured'}")
        for label, env, endpoint in [
            ("elasticsearch", services.elasticsearch_api_key_env, services.elasticsearch_url or ""),
            ("tavily", services.tavily_api_key_env, ""),
            ("typesafe", services.typesafe_api_key_env, ""),
        ]:
            self.ui.tell(
                f"{label} credential: {read_credential(env, service_slot(label, endpoint)).source}"
            )
        action = self.ui.select(
            "Readiness actions",
            [
                ("back", "Back"),
                ("configure", "Configure profile"),
                ("model", "Test model (paid)"),
                ("smoke", "Run integration smoke test (paid)"),
            ],
            "back",
        )
        if action == "configure":
            configure_profile(self.ui, self.path, name)
        elif action == "model" and self.paid("Test model", "One model capability request."):
            check_model(name, profile)
            self.ui.tell("Model tool-call check passed.")
        elif action == "smoke":
            web = (
                self.ui.select("Include Tavily?", [("no", "No"), ("yes", "Yes (one paid search)")])
                == "yes"
            )
            if self.paid(
                "Run integration smoke test",
                "At most two model requests, two indexed "
                "synthetic passages, one retrieval query, and enabled Tavily/Jev requests. "
                "The synthetic run is retained locally and in Elasticsearch.",
            ):
                with open_store() as store:
                    result = smoke_test(store, name, profile, services, web, self.ui.tell)
                self.ui.view("Smoke test", json.dumps(result, indent=2))


def run_ui(path: Path, *, plain: bool = False) -> None:
    try:
        Workspace(path, Terminal(plain=plain)).run()
    except (Cancelled, Back, KeyboardInterrupt):
        return
