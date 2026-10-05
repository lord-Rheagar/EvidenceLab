import json
import sys
from collections import deque
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from evidencelab import interactive, interactive_setup, terminal
from evidencelab.cli import app
from evidencelab.config import (
    AppConfig,
    ConfigError,
    Depth,
    ModelProfile,
    Provider,
    Services,
    load_config,
    save_config,
)
from evidencelab.credentials import profile_slot
from evidencelab.interactive import Workspace, existing_path, local_path
from evidencelab.interactive_setup import SetupDraft, configure_profile, credential, save_draft
from evidencelab.storage import open_store
from evidencelab.terminal import Back, Cancelled, Terminal
from evidencelab.wizard import PendingSecret


class ScriptedTerminal(Terminal):
    def __init__(self, actions=(), texts=(), *, width=80):
        self.buffer = StringIO()
        super().__init__(
            console=Console(file=self.buffer, width=width, height=24, color_system=None)
        )
        self.actions = deque(actions)
        self.texts = deque(texts)
        self.questions = []
        self.views = []

    def select(self, label, choices, default=None):
        self.questions.append((label, choices, default))
        value = self.actions.popleft()
        if isinstance(value, BaseException):
            raise value
        assert value in {key for key, _ in choices}, (label, value, choices)
        return value

    def text(self, label, default="", *, secret=False, validate=None):
        self.questions.append((label, secret, default))
        value = self.texts.popleft()
        if isinstance(value, BaseException):
            raise value
        if validate:
            assert validate(value) is True, (label, value)
        return value

    def view(self, title, content, *, markdown=False):
        self.views.append((title, content))

    def path(self, label, default="", *, validate=None):
        return self.text(label, default, validate=validate)


def configured(path):
    profile = ModelProfile(provider=Provider.OPENAI, model="test-model", depth=Depth.QUICK)
    config = AppConfig(
        default_profile="default",
        profiles={"default": profile},
        services=Services(elasticsearch_url="https://elastic.example", inference_id="embed"),
    )
    save_config(path, config)
    return config


@pytest.mark.parametrize("args", [["ui"], ["ui", "--plain"]])
def test_non_terminal_ui_never_prompts(runner, monkeypatch, args):
    monkeypatch.setattr(interactive, "run_ui", lambda *a, **k: pytest.fail("Unexpected prompt"))
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "terminal input/output" in result.output


def test_bare_command_routes_only_eligible_terminals(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(interactive, "run_ui", lambda *a, **k: calls.append(a))
    result = runner.invoke(app, [])
    assert result.exit_code == 2 and "Usage" in result.output and not calls
    monkeypatch.setattr(terminal, "interactive_terminal", lambda **kwargs: True)
    assert runner.invoke(app, []).exit_code == 0 and len(calls) == 1
    assert runner.invoke(app, ["--help"]).exit_code == 0 and len(calls) == 1
    assert runner.invoke(app, ["--version"]).exit_code == 0 and len(calls) == 1


@pytest.mark.parametrize(
    "stdin,stdout,ci,term,plain,expected",
    [
        (True, True, "", "", False, True),
        (False, True, "", "", False, False),
        (True, False, "", "", False, False),
        (True, True, "true", "", True, False),
        (True, True, "false", "dumb", False, False),
        (True, True, "", "dumb", True, True),
    ],
)
def test_terminal_policy(monkeypatch, stdin, stdout, ci, term, plain, expected):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: stdin))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: stdout))
    monkeypatch.setenv("CI", ci)
    monkeypatch.setenv("TERM", term)
    assert terminal.interactive_terminal(plain=plain) == expected


@pytest.mark.parametrize(
    "keys,expected", [("\r", "first"), ("\x1b[B\r", "second"), ("2\r", "second")]
)
def test_real_keyboard_menu(keys, expected):
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        input.send_text(keys)
        assert Terminal().select("Test", [("first", "First"), ("second", "Second")]) == expected


@pytest.mark.parametrize("keys,error", [("\x1b", Back), ("\x03", Cancelled), ("\x04", Cancelled)])
def test_real_keyboard_cancellation(keys, error):
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        input.send_text(keys)
        with pytest.raises(error):
            Terminal().select("Test", [("exit", "Exit")])


def test_real_password_and_inline_validation(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        input.send_text("private-api-key\r")
        assert Terminal().text("API key", secret=True) == "private-api-key"
        input.send_text("invalid\r\x15VALID_NAME\r")
        result = Terminal().text(
            "Variable", validate=lambda value: value == "VALID_NAME" or "Invalid"
        )
        assert result == "VALID_NAME"


@pytest.mark.parametrize("save", ["save", "cancel"])
def test_guided_setup_stages_keys_until_final_save(isolated_settings, vault, save):
    ui = ScriptedTerminal(
        actions=["openrouter", "paste", "balanced", "none", "keep", "keep", save],
        texts=["vendor/model", "PRIVATE_KEY"],
    )
    configure_profile(ui, isolated_settings, "default")
    assert bool(vault) == (save == "save")
    assert isolated_settings.exists() == (save == "save")
    assert "PRIVATE_KEY" not in ui.buffer.getvalue()
    assert any(
        label == "API key" and secret
        for label, secret, _ in ui.questions
        if isinstance(secret, bool)
    )
    if save == "save":
        assert "PRIVATE_KEY" not in isolated_settings.read_text()
        assert load_config(isolated_settings).profiles["default"].depth == "balanced"


def test_setup_back_retains_model_and_cancel_makes_no_writes(isolated_settings, vault):
    ui = ScriptedTerminal(
        actions=["openai", Back(), "openai", Cancelled()], texts=["my-model", "my-model"]
    )
    with pytest.raises(Cancelled):
        configure_profile(ui, isolated_settings, "default")
    model_questions = [q for q in ui.questions if q[0] == "Model ID"]
    assert model_questions[1][2] == "my-model"
    assert not isolated_settings.exists() and not vault


def test_switching_provider_drops_old_staged_secret():
    config = AppConfig()
    old = ModelProfile(provider=Provider.OPENAI, model="old")
    slot = profile_slot("default", old)
    draft = SetupDraft(
        config, "default", old.model_dump(), Services(), {slot: PendingSecret(slot, "PRIVATE")}
    )
    ui = ScriptedTerminal(actions=["anthropic"], texts=["new"])
    interactive_setup.model_step(ui, draft)
    assert not draft.secrets and draft.selected().api_key_env == "ANTHROPIC_API_KEY"


def test_variable_branch_never_asks_for_unmasked_key():
    draft = SetupDraft(AppConfig(), "default", {}, Services())
    ui = ScriptedTerminal(actions=["environment"], texts=["CUSTOM_KEY"])
    assert credential(ui, draft, "Model", "OPENAI_API_KEY", "slot") == "CUSTOM_KEY"
    assert ui.questions[-1][1] is True and not draft.secrets
    assert interactive_setup.environment_name("sk-secret-key") is not True
    assert "sk-secret-key" not in interactive_setup.environment_name("sk-secret-key")


def test_partial_keyring_save_failure_is_explicit(isolated_settings, monkeypatch):
    draft = SetupDraft(
        AppConfig(),
        "default",
        {"provider": "openai", "model": "m"},
        Services(),
        {"one": PendingSecret("one", "SECRET"), "two": PendingSecret("two", "SECRET")},
    )
    written = []

    def write(slot, value):
        if written:
            raise ConfigError("PRIVATE_FAILURE")
        written.append(slot)

    monkeypatch.setattr(interactive_setup, "store_credential", write)
    with pytest.raises(ConfigError, match=r"1 credential\(s\) were stored") as error:
        save_draft(isolated_settings, draft)
    assert "PRIVATE" not in str(error.value) and not isolated_settings.exists()


def test_paid_launch_is_explicit_and_cancel_has_no_run(isolated_settings, monkeypatch):
    configured(isolated_settings)
    for key in ["OPENAI_API_KEY", "ELASTICSEARCH_API_KEY", "TAVILY_API_KEY"]:
        monkeypatch.setenv(key, "test-key")
    called = []
    monkeypatch.setattr(interactive, "create_job", lambda *a: called.append(a))
    ui = ScriptedTerminal(actions=["default", "start", "back", "cancel"], texts=["Question?"])
    Workspace(isolated_settings, ui).research()
    assert not called
    with open_store() as store:
        assert store.runs() == []


def test_paid_launch_uses_shared_engine_and_unsaved_depth(isolated_settings, monkeypatch):
    configured(isolated_settings)
    for key in ["OPENAI_API_KEY", "ELASTICSEARCH_API_KEY", "TAVILY_API_KEY"]:
        monkeypatch.setenv(key, "test-key")
    calls = []
    ui = ScriptedTerminal(
        actions=["default", "depth", "balanced", "start", "start"], texts=["Question?"]
    )
    workspace = Workspace(isolated_settings, ui)
    monkeypatch.setattr(workspace, "execute", lambda run_id: calls.append(run_id))
    monkeypatch.setattr(workspace, "run_details", lambda run_id: None)
    workspace.research()
    assert len(calls) == 1
    with open_store() as store:
        assert store.get_job(calls[0]).profile.depth == "balanced"
    assert load_config(isolated_settings).profiles["default"].depth == "quick"


def test_missing_setup_never_launches(isolated_settings, monkeypatch):
    configured(isolated_settings)
    monkeypatch.setattr(interactive, "create_job", lambda *a: pytest.fail("Must not launch"))
    ui = ScriptedTerminal(actions=["default", "start", "back", "cancel"], texts=["Question?"])
    Workspace(isolated_settings, ui).research()


def test_offline_readiness_never_calls_model(isolated_settings, monkeypatch):
    configured(isolated_settings)
    monkeypatch.setattr(interactive, "check_model", lambda *a: pytest.fail("Paid call"))
    ui = ScriptedTerminal(actions=["default", "back"])
    Workspace(isolated_settings, ui).readiness()
    assert "connections have not been tested" in ui.buffer.getvalue()


@pytest.mark.parametrize("width", [40, 80, 120])
def test_header_layout_control_sanitization_and_pagination(width):
    buffer = StringIO()
    console = Console(file=buffer, width=width, height=15, color_system=None)
    ui = Terminal(console=console)
    ui.heading("EvidenceLab", "provider/" + "long-model-" * 12)
    ui.tell("[bold]untrusted\x1b[2J\u202e")
    assert "\x1b" not in buffer.getvalue() and "\u202e" not in buffer.getvalue()
    assert "[bold]" in buffer.getvalue()
    assert all(len(line) <= width for line in buffer.getvalue().splitlines())
    actions = iter(["next", "previous", "back"])
    ui.select = lambda *args: next(actions)
    ui.view("Report", "\n".join(f"Line {i}" for i in range(25)))
    assert "Line 0" in buffer.getvalue()


def test_path_spaces_and_blank(tmp_path):
    path = tmp_path / "research notes.txt"
    path.write_text("local notes")
    assert local_path(f'"{path}"') == path
    assert existing_path(str(path)) is True
    assert existing_path("") is not True


def test_json_commands_do_not_open_ui(runner, monkeypatch):
    monkeypatch.setattr(terminal, "interactive_terminal", lambda **kwargs: True)
    monkeypatch.setattr(interactive, "run_ui", lambda *a, **k: pytest.fail("Must not show UI"))
    result = runner.invoke(app, ["runs", "--json"])
    assert result.exit_code == 0 and json.loads(result.stdout)["runs"] == []
    assert "\x1b" not in result.stdout


def completed_run(path):
    from evidencelab.documents import extract_source
    from evidencelab.research_records import Citation, Claim, ResearchJob, ResearchReport

    config = configured(path)
    with open_store() as store:
        run = store.create_run(Depth.QUICK)
        source = extract_source(
            run.run_id,
            "https://docs.example/test",
            b"Exact source quotation about technical research.",
            "text",
            "Official documentation",
        )
        store.add_source(source)
        passage = store.passages(run.run_id)[0]
        claim = Claim(
            text="A supported claim.",
            citations=[Citation(passage_id=passage.passage_id, quote=passage.text)],
        )
        job = ResearchJob(
            run_id=run.run_id,
            question="Saved technical question?",
            profile_name="default",
            profile=config.profiles["default"],
            elasticsearch_url=config.services.elasticsearch_url,
            inference_id="embed",
            web=False,
            status="complete",
            stage="complete",
            evidence_ids=[passage.passage_id],
            model_calls=3,
            report=ResearchReport(
                title="Saved report",
                summary=[claim],
                findings=[claim],
                tradeoffs=[],
                limitations=["Limited source coverage."],
            ),
        )
        store.save_job(job)
    return job, source


def test_browser_and_report_actions_are_offline(isolated_settings, monkeypatch):
    job, source = completed_run(isolated_settings)
    monkeypatch.setattr(
        interactive, "execute_saved_research", lambda *a: pytest.fail("No requests")
    )
    ui = ScriptedTerminal(
        actions=[
            job.run_id,
            "summary",
            "report",
            "citations",
            "0",
            "sources",
            source.source_id,
            "diagnostics",
            "back",
            "__back__",
        ],
        texts=[""],
    )
    Workspace(isolated_settings, ui).browse()
    assert [v[0] for v in ui.views] == [
        "Summary",
        "Research report",
        "Citation passages",
        "Official documentation",
        "Diagnostics",
    ]
    assert job.evidence_ids[0] in ui.views[2][1]
    with open_store() as store:
        assert store.get_job(job.run_id) == job


def test_export_collision_preserves_existing_file(isolated_settings, tmp_path):
    job, _ = completed_run(isolated_settings)
    original = tmp_path / "report.md"
    original.write_text("DO NOT REPLACE")
    target = tmp_path / "report-copy.md"
    ui = ScriptedTerminal(actions=["markdown"], texts=[str(original), str(target)])
    Workspace(isolated_settings, ui).export(job)
    assert original.read_text() == "DO NOT REPLACE"
    assert "Saved report" in target.read_text(encoding="utf-8")
    assert "will not be overwritten" in ui.buffer.getvalue()


def test_resume_decline_keeps_failed_job_unchanged(isolated_settings, monkeypatch):
    job, _ = completed_run(isolated_settings)
    job.status, job.stage, job.report = "failed", "write", None
    job.error = "Safe diagnostic failure."
    with open_store() as store:
        store.save_job(job)
    monkeypatch.setattr(interactive, "execute_saved_research", lambda *a: pytest.fail("Paid retry"))
    ui = ScriptedTerminal(actions=["resume", "back", "back"])
    Workspace(isolated_settings, ui).run_details(job.run_id)
    with open_store() as store:
        assert store.get_job(job.run_id) == job


def test_import_is_explicit_local_and_supports_paths_with_spaces(isolated_settings, tmp_path):
    document = tmp_path / "research notes.txt"
    document.write_text("Local technical source text for importing.")
    ui = ScriptedTerminal(actions=["file", "done", "start"], texts=[f'"{document}"'])
    Workspace(isolated_settings, ui).import_documents()
    with open_store() as store:
        run = store.runs()[0]
        assert run.status == "local" and len(store.sources(run.run_id)) == 1


def test_actual_password_output_is_masked():
    class RecordingOutput(DummyOutput):
        def __init__(self):
            self.fragments = []

        def write(self, data):
            self.fragments.append(data)

        def write_raw(self, data):
            self.fragments.append(data)

    output = RecordingOutput()
    with create_pipe_input() as input, create_app_session(input=input, output=output):
        input.send_text("SECRET_DO_NOT_SHOW\r")
        assert Terminal().text("API key", secret=True) == "SECRET_DO_NOT_SHOW"
    assert "SECRET_DO_NOT_SHOW" not in "".join(output.fragments)


def test_real_path_prompt_accepts_spaces(tmp_path):
    document = tmp_path / "source notes.txt"
    document.write_text("Evidence.")
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        input.send_text(str(document) + "\r")
        assert Terminal().path("Path", validate=existing_path) == str(document)


def test_plain_numbered_menu_validation_and_cancellation(monkeypatch):
    answers = iter(["bad", "2", ":back", ":cancel"])
    monkeypatch.setattr(terminal.typer, "prompt", lambda *a, **k: next(answers))
    ui = Terminal(plain=True, console=Console(file=StringIO(), color_system=None))
    assert ui.select("Pick", [("a", "First"), ("b", "Second")]) == "b"
    with pytest.raises(Back):
        ui.text("Value")
    with pytest.raises(Cancelled):
        ui.text("Value")
