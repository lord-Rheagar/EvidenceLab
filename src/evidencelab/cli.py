"""Console entry point for EvidenceLab."""

import json
import sys
from collections.abc import Callable
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from evidencelab.config import (
    KEY_ENV,
    ConfigError,
    Depth,
    ModelProfile,
    Provider,
    Reranker,
    Services,
    config_path,
    load_config,
    resolve_profile,
    resolve_services,
    save_config,
)
from evidencelab.credentials import profile_slot, read_credential, service_slot, store_credential
from evidencelab.evidence import EvidenceError
from evidencelab.providers import MODEL_CATALOGS, check_model
from evidencelab.wizard import PendingSecret, collect_profile, collect_services, with_profile

app = typer.Typer(
    help="Technical research with inspectable evidence and user-selected models.",
    no_args_is_help=False,
    add_completion=False,
)
profiles_app = typer.Typer(help="Manage saved model profiles.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect supported model providers.", no_args_is_help=True)
app.add_typer(profiles_app, name="profiles")
app.add_typer(models_app, name="models")


def handle_errors[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (ConfigError, ValidationError, EvidenceError) as exc:
            message = str(exc)
            if isinstance(exc, ValidationError):
                message = "; ".join(
                    f"{'.'.join(map(str, error['loc'])) or 'configuration'}: {error['msg']}"
                    for error in exc.errors(include_input=False, include_context=False)
                )
            if kwargs.get("json_output"):
                typer.echo(json.dumps({"schema_version": 1, "error": message}))
            else:
                typer.echo(f"Error: {message}", err=True)
            raise typer.Exit(1 if isinstance(exc, EvidenceError) else 2) from None

    return wrapped


def selected_path(ctx: typer.Context) -> Path:
    return Path(ctx.find_root().obj)


@app.command("smoke-test")
@handle_errors
def smoke_test_command(
    ctx: typer.Context,
    profile: str | None = None,
    web: Annotated[
        bool,
        typer.Option(
            "--web/--no-web",
            help="Also run one paid Tavily search; requires a Tavily key. No pages are fetched.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run paid live checks with synthetic data; keep the small evidence run for inspection."""
    from evidencelab.smoke import smoke_test
    from evidencelab.storage import open_store

    config = load_config(selected_path(ctx))
    name, selected = resolve_profile(config, profile)
    typer.echo(
        "Live smoke test: at most two model requests, two indexed synthetic passages, "
        "one hybrid query, optionally one Tavily and one Jev request. API charges may apply. "
        "The synthetic evidence run is retained locally and in Elasticsearch.",
        err=True,
    )
    with open_store() as store:
        result = smoke_test(
            store,
            name,
            selected,
            resolve_services(config),
            web,
            lambda message: typer.echo(message, err=True),
        )
    emit(result, json_output)
    if result["status"] != "passed":
        raise typer.Exit(1)


def is_interactive() -> bool:
    return sys.stdin.isatty()


def emit(payload: dict[str, object], json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"schema_version": 1, **payload}, indent=2))
    else:
        for label, value in payload.items():
            typer.echo(f"{label}: {value}")


def echo_report(report: str) -> None:
    # Redirected Windows stdout may use cp1252 even when evidence is multilingual.
    # Escape only unrepresentable display characters; stored/exported text stays UTF-8.
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    typer.echo(report.encode(encoding, errors="backslashreplace").decode(encoding))


@app.callback(invoke_without_command=True)
@handle_errors
def main(
    ctx: typer.Context,
    show_version: Annotated[
        bool, typer.Option("--version", help="Show the installed version.", is_eager=True)
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="Configuration file path.")
    ] = None,
) -> None:
    """Run EvidenceLab."""
    if show_version:
        typer.echo(f"EvidenceLab {version('evidencelab')}")
        raise typer.Exit()
    ctx.obj = config_file or config_path()
    if ctx.invoked_subcommand is None:
        from evidencelab.terminal import interactive_terminal

        if interactive_terminal():
            from evidencelab.interactive import run_ui

            run_ui(Path(ctx.obj))
        else:
            typer.echo(ctx.get_help())
            raise typer.Exit(2)


@app.command()
@handle_errors
def ui(
    ctx: typer.Context,
    plain: Annotated[bool, typer.Option("--plain", help="Use static numbered menus.")] = False,
) -> None:
    """Open the interactive research workspace. No paid actions run automatically."""
    from evidencelab.interactive import run_ui
    from evidencelab.terminal import interactive_terminal

    if not interactive_terminal(plain=plain):
        raise ConfigError(
            "Interactive UI requires terminal input/output outside CI. "
            "Use normal commands with --json or configure --non-interactive."
        )
    run_ui(selected_path(ctx), plain=plain)


@app.command()
@handle_errors
def configure(
    ctx: typer.Context,
    profile: Annotated[str, typer.Option(help="Profile to create or edit.")] = "default",
    provider: Provider | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key_required: Annotated[
        bool | None, typer.Option("--api-key-required/--no-api-key")
    ] = None,
    depth: Depth | None = None,
    reranker: Reranker | None = None,
    jev_model: str | None = None,
    elasticsearch_url: str | None = None,
    inference_id: str | None = None,
    set_default: Annotated[bool, typer.Option("--set-default")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    """Save a model profile and service settings; no network requests are made."""
    path = selected_path(ctx)
    config = load_config(path)
    if not non_interactive and not is_interactive():
        raise ConfigError("Use a terminal for the wizard, or pass --non-interactive with settings.")
    if (
        not non_interactive
        and not set_default
        and all(
            value is None
            for value in [
                provider,
                model,
                base_url,
                api_key_env,
                api_key_required,
                depth,
                reranker,
                jev_model,
                elasticsearch_url,
                inference_id,
            ]
        )
    ):
        from evidencelab.terminal import Back, Cancelled, Terminal, interactive_terminal

        if interactive_terminal():
            from evidencelab.interactive_setup import configure_profile

            try:
                configure_profile(Terminal(), path, profile)
            except (Back, Cancelled):
                typer.echo("Configuration unchanged.")
            return
    existing = config.profiles.get(profile)
    seed = existing.model_dump() if existing else {}
    if provider is not None and existing and provider != existing.provider:
        seed = {
            "depth": existing.depth,
            "reranker": existing.reranker,
            "jev_model": existing.jev_model,
        }
    overrides = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "depth": depth,
        "reranker": reranker,
        "jev_model": jev_model,
    }
    seed.update({key: value for key, value in overrides.items() if value is not None})
    if api_key_required is not None:
        seed["api_key_required"] = api_key_required
    service_data = config.services.model_dump()
    if elasticsearch_url is not None:
        service_data["elasticsearch_url"] = elasticsearch_url
    if inference_id is not None:
        service_data["inference_id"] = inference_id
    services = type(config.services).model_validate(service_data)
    secrets: list[PendingSecret] = []
    if non_interactive:
        chosen = ModelProfile.model_validate(seed)
    else:
        chosen, secrets = collect_profile(profile, seed)
        services, service_secrets = collect_services(services, chosen.reranker)
        secrets.extend(service_secrets)
        if config.default_profile and config.default_profile != profile:
            set_default = set_default or typer.confirm(
                "Make this the default profile?", default=False
            )
    updated = with_profile(config, profile, chosen, services, make_default=set_default)
    if not non_interactive:
        typer.echo(f"Profile: {profile}; provider: {chosen.provider}; model: {chosen.model}")
        if not typer.confirm("Save configuration?", default=True):
            typer.echo("Configuration unchanged.")
            return
    for secret in secrets:
        store_credential(secret.slot, secret.value)
    save_config(path, updated)
    typer.echo(f"Saved profile '{profile}' to {path}")
    typer.echo("No model calls made. Run 'evidence doctor' to check local readiness.")


@profiles_app.command("list")
@handle_errors
def list_profiles(
    ctx: typer.Context, json_output: Annotated[bool, typer.Option("--json")] = False
) -> None:
    """List saved profiles without credentials."""
    config = load_config(selected_path(ctx))
    items = [
        {"name": name, "default": name == config.default_profile, **item.model_dump(mode="json")}
        for name, item in config.profiles.items()
    ]
    if json_output:
        emit({"profiles": items}, True)
    elif not items:
        typer.echo("No profiles saved. Run 'evidence configure'.")
    else:
        for item in items:
            marker = "*" if item["default"] else " "
            typer.echo(f"{marker} {item['name']}  {item['provider']}  {item['model']}")


@profiles_app.command("show")
@handle_errors
def show_profile(
    ctx: typer.Context,
    profile: str | None = None,
    provider: Provider | None = None,
    model: str | None = None,
    base_url: str | None = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show the effective profile, optionally previewing unsaved model overrides."""
    name, selected = resolve_profile(
        load_config(selected_path(ctx)), profile, provider=provider, model=model, base_url=base_url
    )
    emit({"name": name, **selected.model_dump(mode="json")}, json_output)


@profiles_app.command("use")
@handle_errors
def use_profile(ctx: typer.Context, name: str) -> None:
    """Choose the default profile."""
    path = selected_path(ctx)
    config = load_config(path)
    resolve_profile(config, name)
    config.default_profile = name
    save_config(path, config)
    typer.echo(f"Default profile: {name}")


@models_app.command("providers")
def list_providers(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List providers, credential variables, and model catalogs (no network)."""
    providers = [
        {"provider": p.value, "api_key_env": KEY_ENV[p], "catalog": MODEL_CATALOGS[p]}
        for p in Provider
    ]
    if json_output:
        emit({"providers": providers}, True)
    else:
        for item in providers:
            typer.echo(f"{item['provider']} ({item['api_key_env']})\n  {item['catalog']}")


@app.command()
@handle_errors
def doctor(
    ctx: typer.Context,
    profile: str | None = None,
    provider: Provider | None = None,
    model: str | None = None,
    base_url: str | None = None,
    model_check: Annotated[
        bool, typer.Option(help="Send one model capability probe; API charges may apply.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check local settings and credential presence; optionally test the model live."""
    config = load_config(selected_path(ctx))
    name, selected = resolve_profile(
        config, profile, provider=provider, model=model, base_url=base_url
    )
    credential = read_credential(selected.api_key_env or "", profile_slot(name, selected))
    ready = not selected.api_key_required or credential.value is not None
    checks: dict[str, object] = {
        "profile": name,
        "provider": selected.provider.value,
        "model": selected.model,
        "model_credential": credential.source if selected.api_key_required else "not-required",
        "model_ready": ready,
        "model_check": "not-run",
    }
    services = resolve_services(config)
    for label, env, endpoint in [
        ("elasticsearch", services.elasticsearch_api_key_env, services.elasticsearch_url or ""),
        ("tavily", services.tavily_api_key_env, ""),
        ("typesafe", services.typesafe_api_key_env, ""),
    ]:
        value = read_credential(env, service_slot(label, endpoint))
        checks[f"{label}_credential"] = value.source
        if label == "typesafe" and selected.reranker == Reranker.JEV and value.value is None:
            ready = False
    checks["elasticsearch_url"] = services.elasticsearch_url or "not-configured"
    checks["inference_id"] = services.inference_id or "not-configured"
    checks["note"] = (
        "Credential presence is not authentication. Service connectivity is not tested."
    )
    if model_check:
        typer.echo("Sending one model capability probe; API charges may apply.", err=True)
        check_model(name, selected)
        checks["model_check"] = "passed: tool calling and structured tool arguments"
    checks["profile_credentials_ready"] = ready
    emit(checks, json_output)
    if not ready:
        raise typer.Exit(1)


@app.command()
@handle_errors
def ingest(
    ctx: typer.Context,
    files: Annotated[
        list[Path] | None, typer.Option("--file", help="File or directory; repeatable.")
    ] = None,
    urls: Annotated[
        list[str] | None, typer.Option("--url", help="Public document URL; repeatable.")
    ] = None,
    run_id: str | None = None,
    profile: str | None = None,
    depth: Depth | None = None,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Save snapshots without cloud indexing.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Extract cited passages from supplied documents and optionally index them."""
    from evidencelab.ingestion import ingest_sources
    from evidencelab.retrieval import open_index, validate_search_settings
    from evidencelab.storage import open_store

    config = load_config(selected_path(ctx))
    services = resolve_services(config)
    selected_depth = depth
    if profile or config.default_profile:
        _, selected = resolve_profile(config, profile)
        selected_depth = depth or selected.depth
    if not local_only:
        validate_search_settings(services)
        typer.echo(
            "Extracted passages will be sent to Elasticsearch and its embedding service.", err=True
        )
    with open_store() as store:
        result = ingest_sources(
            store, files or [], urls or [], run_id=run_id, depth=selected_depth or Depth.THOROUGH
        )
        actual_id = str(result["run_id"])
        if result["passages"] and not local_only:
            try:
                with store.lock(actual_id), open_index(services) as index:
                    index.index_run(store, actual_id)
            except EvidenceError as exc:
                result["error"] = str(exc)
        result["status"] = store.get_run(actual_id).status
        emit(result, json_output)
        if result["skipped"] or not result["passages"] or "error" in result:
            raise typer.Exit(1)


@app.command("index")
@handle_errors
def index_evidence(
    ctx: typer.Context, run_id: str, json_output: Annotated[bool, typer.Option("--json")] = False
) -> None:
    """Index or retry a run's saved snapshots in Elasticsearch (embedding charges may apply)."""
    from evidencelab.retrieval import open_index
    from evidencelab.storage import open_store

    services = resolve_services(load_config(selected_path(ctx)))
    with open_store() as store, store.lock(run_id), open_index(services) as index:
        count = index.index_run(store, run_id)
        emit({"run_id": run_id, "indexed_passages": count, "status": "indexed"}, json_output)


@app.command()
@handle_errors
def search(
    ctx: typer.Context,
    run_id: str,
    question: str,
    limit: Annotated[int, typer.Option(min=1, max=30)] = 12,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Retrieve passages with citations; does not generate an LLM answer."""
    from evidencelab.retrieval import open_index
    from evidencelab.storage import open_store

    services = resolve_services(load_config(selected_path(ctx)))
    with open_store() as store, store.lock(run_id), open_index(services) as index:
        hits = index.search(store.get_run(run_id), question, limit)
        # Local snapshots are the citation source of truth, even if the remote index was modified.
        saved = {p.passage_id: p for p in store.passages(run_id)}
        if any(saved.get(hit.passage.passage_id) != hit.passage for hit in hits):
            raise EvidenceError("Search evidence does not match local snapshots; reindex the run.")
        if json_output:
            emit({"run_id": run_id, "hits": [hit.model_dump(mode="json") for hit in hits]}, True)
        elif not hits:
            typer.echo("No matching passages found.")
        else:
            for number, hit in enumerate(hits, 1):
                p = hit.passage
                typer.echo(
                    f"[{number}] {p.title}\n{p.location}\n{p.locator}\n"
                    f"Passage: {p.passage_id}\n{p.text}\n"
                )


@app.command()
@handle_errors
def runs(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List saved evidence runs."""
    from evidencelab.storage import open_store

    with open_store() as store:
        records = store.runs()
        if json_output:
            emit(
                {
                    "runs": [
                        {**run.model_dump(mode="json"), "research": store.job_summary(run.run_id)}
                        for run in records
                    ]
                },
                True,
            )
        elif not records:
            typer.echo("No evidence runs yet. Use 'evidence ingest'.")
        else:
            for run in records:
                job = store.job_summary(run.run_id)
                status = f"research:{job['status']} ({job['stage']})" if job else run.status
                if run.purpose == "smoke":
                    status = f"smoke:{status}"
                typer.echo(f"{run.run_id}  {status}  {run.depth}  {run.created_at}")


@app.command()
@handle_errors
def inspect(
    run_id: str,
    sources: Annotated[bool, typer.Option("--sources")] = False,
    passage: str | None = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect local evidence metadata or a cited passage without network access."""
    from evidencelab.storage import open_store

    with open_store() as store:
        run = store.get_run(run_id)
        saved_sources = store.sources(run_id)
        passages = store.passages(run_id)
        result: dict[str, object] = {
            **run.model_dump(mode="json"),
            "source_count": len(saved_sources),
            "passage_count": len(passages),
            "research": store.job_summary(run_id),
        }
        if sources:
            result["sources"] = [
                source.model_dump(mode="json", exclude={"segments"}) for source in saved_sources
            ]
        if passage:
            selected = next((p for p in passages if p.passage_id == passage), None)
            if selected is None:
                raise EvidenceError("Passage not found in this run.")
            result["passage"] = selected.model_dump(mode="json")
        if json_output:
            emit(result, True)
        else:
            emit(
                {key: value for key, value in result.items() if key not in {"sources", "passage"}},
                False,
            )
            if sources:
                for source in saved_sources:
                    typer.echo(f"\n{source.title}\n{source.location}\nSource: {source.source_id}")
                    for warning in source.warnings:
                        typer.echo(f"Warning: {warning}")
            if passage and selected is not None:
                typer.echo(f"\n{selected.locator}\n{selected.location}\n{selected.text}")


@app.command()
@handle_errors
def diagnostics(
    run_id: str,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read safe request diagnostics and stage timings without service calls."""
    from evidencelab.storage import open_store

    with open_store() as store:
        job = store.get_job(run_id)
        if json_output:
            emit(
                {
                    "run_id": run_id,
                    "status": job.status,
                    "stage": job.stage,
                    "diagnostics": [e.model_dump(mode="json") for e in job.diagnostics],
                },
                True,
            )
        else:
            typer.echo(f"Run: {run_id} | {job.status} | {job.stage}")
            if not job.diagnostics:
                typer.echo("No diagnostics recorded. Older runs acquire diagnostics on resume.")
            for event in job.diagnostics:
                status = f" HTTP {event.http_status}" if event.http_status else ""
                typer.echo(
                    f"{event.timestamp} {event.stage} {event.service}/{event.operation}: "
                    f"{event.code}{status} ({event.elapsed_seconds:.1f}s)"
                )


def execute_research(run_id: str, services: "Services", json_output: bool) -> None:
    from evidencelab.reports import render_report
    from evidencelab.research import execute_saved_research
    from evidencelab.storage import open_store

    with open_store() as store:
        job = store.get_job(run_id)
        if job.status != "complete":
            typer.echo(f"Research run: {run_id}; resume with 'evidence resume {run_id}'.", err=True)
            typer.echo(
                "Research sends questions and evidence to your selected model and Elasticsearch; "
                "web queries go to Tavily and candidate excerpts to TypeSafe when enabled. "
                "API charges may apply.",
                err=True,
            )
            try:
                job = execute_saved_research(
                    store, run_id, services, lambda message: typer.echo(message, err=True)
                )
            except (EvidenceError, ConfigError) as exc:
                job = store.get_job(run_id)
                emit(
                    {
                        "run_id": run_id,
                        "status": job.status,
                        "stage": job.stage,
                        "error": job.error or str(exc),
                        "diagnostics": [
                            event.model_dump(mode="json") for event in job.diagnostics[-10:]
                        ],
                    },
                    json_output,
                )
                raise typer.Exit(1) from None
        if json_output:
            emit({"research": job.model_dump(mode="json")}, True)
        else:
            echo_report(render_report(store, job))
            typer.echo(f"\nExport: evidence export {run_id} --output report.md", err=True)


@app.command()
@handle_errors
def research(
    ctx: typer.Context,
    question: str,
    profile: str | None = None,
    provider: Provider | None = None,
    model: str | None = None,
    base_url: str | None = None,
    depth: Depth | None = None,
    files: Annotated[list[Path] | None, typer.Option("--file")] = None,
    urls: Annotated[list[str] | None, typer.Option("--url")] = None,
    web: Annotated[bool, typer.Option("--web/--no-web")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Research a technical question and generate a cited report (paid services)."""
    from evidencelab.research import create_job
    from evidencelab.storage import open_store

    config = load_config(selected_path(ctx))
    name, selected = resolve_profile(
        config, profile, provider=provider, model=model, base_url=base_url
    )
    if depth is not None:
        selected = selected.model_copy(update={"depth": depth})
    services = resolve_services(config)
    with open_store() as store:
        job = create_job(store, question, name, selected, services, files or [], urls or [], web)
    execute_research(job.run_id, services, json_output)


@app.command()
@handle_errors
def resume(
    ctx: typer.Context,
    run_id: str,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume saved research using its original model settings and remaining budget."""
    execute_research(run_id, resolve_services(load_config(selected_path(ctx))), json_output)


@app.command("export")
@handle_errors
def export_research(
    run_id: str,
    output: Annotated[Path, typer.Option("--output")],
    format: Annotated[str, typer.Option("--format", help="markdown or json")] = "markdown",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Export a completed report without network calls; never overwrite a file."""
    from evidencelab.reports import export_report
    from evidencelab.storage import open_store

    with open_store() as store, store.lock(run_id):
        export_report(store, store.get_job(run_id), output, format)
        emit({"run_id": run_id, "output": str(output.resolve()), "format": format}, json_output)
