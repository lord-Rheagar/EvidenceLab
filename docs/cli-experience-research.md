# EvidenceLab Interactive CLI Research

Date: 2026-10-05. Research performed by a GPT-6.1 Sol subagent and consolidated against the codebase. This is the original design recommendation. A subsequent implementation added the Rich + Questionary workspace, guided setup/launch, saved-run browsing, progress and recovery actions; see the README's Quick Start section. Textual remains a future option.

## Recommendation

Build a terminal user interface (TUI) over the existing Typer commands using **Rich + Questionary** first. Keep the research pipeline, checkpoints, request accounting, credential handling and citation validation shared between interactive and scripted usage. Consider Textual later if persistent multi-view report browsing becomes necessary.

The reference screenshot is useful for its clear identity, focused onboarding and deliberate continuation. EvidenceLab should use its own compact branding, without reproducing Claude's logo, authentication flow or security wording. Returning users should reach their work immediately rather than repeatedly dismissing a welcome screen.

## Current Friction

| Current behavior | Proposed change |
| --- | --- |
| Bare invocation displays help (`cli.py`, Typer application configuration). | Add an explicit interactive home first; retain help and all commands. |
| `wizard.prompt_choice` requires exact typed enum values. | Arrow-key selection with defaults and a numbered fallback. |
| A visible API-key environment-variable prompt precedes credential selection. | Ask how the user wants to supply credentials first; keep actual key entry masked. |
| Model, environment-variable and profile validation happens after several questions. | Validate each field immediately; preserve nonsecret draft answers for correction. |
| Setup presents advanced fields alongside essential choices. | Reveal custom endpoints, environment-variable overrides and Jev version selection only when relevant. |
| `is_interactive()` checks only stdin. | Check stdin and the stream rendering the UI; explicitly handle JSON, CI and redirected output. |
| Progress consists of append-only messages and long silent calls. | Show current stage, elapsed time, operation and budget counters in interactive mode. |
| Completion prints the entire report. | Show a short result summary and actions to read, inspect citations or export. |

The existing pipeline already supports durable stages, bounded schema correction and safe diagnostics. Those are implementation assets to preserve, not replace with a second execution engine.

## Library Choice

| Option | Verified capabilities | Recommendation |
| --- | --- | --- |
| Rich + Questionary | Styled output; keyboard selection, defaults, disabled choices, masked passwords, input validation and path completion. | Phase 1. Rich and Typer are already dependencies; Questionary would be new. |
| Direct prompt_toolkit | Custom key bindings, input handling and testing support. | Use selectively where Questionary cannot express the needed interaction. |
| Textual | Screens, inputs, Markdown viewing, workers and headless UI testing; Windows support. | Later, for a persistent run dashboard and report workspace. |

Sources: [Questionary prompt types](https://questionary.readthedocs.io/en/stable/pages/types.html), [prompt_toolkit input](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/asking_for_input.html), [Textual screens](https://textual.textualize.io/guide/screens/), [Textual inputs](https://textual.textualize.io/widgets/input/), [Textual getting started](https://github.com/Textualize/textual/blob/main/docs/getting_started.md).

These capabilities do not make a finished wizard automatically. Back navigation, draft state, cross-field validation, safe persistence and paid-action confirmation remain application responsibilities.

## Proposed Experience

### Home

Introduce `evidence ui` as a proposed explicit entry point. After compatibility testing, bare `evidence` could open it in an eligible terminal. Noninteractive invocation must continue to show help or require explicit arguments without waiting for input. Typer provides the relevant `no_args_is_help` and `invoke_without_command` routing controls. [Typer reference](https://typer.tiangolo.com/reference/typer/)

Illustrative home, not current application output:

```text
EvidenceLab                                      default
Technical research

Provider   OpenRouter
Model      deepseek/deepseek-v4.1-flash
Readiness  Credentials present; connection not checked

> Research a question
  Browse saved runs
  Import documents
  Configure profiles
  Check readiness
  Exit
```

Show configured and verified states separately. A stored key is not proof of provider access, credit availability or tool compatibility. Selecting a task must not launch paid work before the user reviews it.

### Setup

1. Select a provider and enter/select a model. Retain manual model-ID entry so users are not restricted to an outdated catalog.
2. Choose credential handling: keep existing, paste securely into the OS credential manager, use an environment variable, or configure later.
3. Ask for an environment-variable name only in that branch. Display the provider's default name and hide customization under advanced settings.
4. Collect the Elasticsearch connection and embedding endpoint. Explain missing prerequisites at that step rather than allowing apparent readiness.
5. Reveal Tavily setup when web discovery is wanted, and TypeSafe setup when Jev is enabled.
6. Review nonsecret settings and choose Save, Edit or Cancel.

Back preserves nonsecret draft values. A provider/endpoint change discards obsolete staged credentials. Before Save, cancellation must make no configuration or credential writes. Commit failures must report what succeeded without a plaintext fallback; do not assume that a TOML write and an OS keyring update form one atomic transaction.

Questionary provides masked password input, but `.ask()` returns `None` on Ctrl+C while `.unsafe_ask()` propagates interruption. Cancellation must therefore be handled explicitly, never converted to a string or treated as a default answer. [Prompt types](https://questionary.readthedocs.io/en/stable/pages/types.html), [advanced behavior](https://questionary.readthedocs.io/en/stable/pages/advanced.html)

### Research Launch

Collect the question, depth and source choices first. Show optional model overrides, reranking and advanced settings only when opened. Review the selected model, sources, web discovery, reranker, data destinations and request ceilings before a clearly labelled paid launch.

Do not imply that a request ceiling is a dollar limit. Offline readiness checks, paid model capability checks and live smoke tests must be distinct actions. Ordinary scripted commands keep their existing contracts; the new guided flow must not add blocking prompts to them.

### Progress and Recovery

Show the current stage, elapsed time, current operation, run ID, warning count and request-budget usage. Use exact progress only when the amount of work is known. Source and search limits are ceilings, not a valid overall completion denominator. Rich supports indeterminate progress with `total=None`. [Rich progress](https://rich.readthedocs.io/en/latest/progress.html)

On failure, show a short reason and relevant actions: view diagnostics, correct configuration, explicitly resume, or exit. Preserve bounded schema correction and durable request reservations. Never silently select another model, disable Jev, widen validation, or retry a paid request because the interface wants to appear successful.

Examples of safe proactive correction:

- Prevent invalid provider/depth choices with selection menus.
- Keep invalid URL input editable and explain the endpoint requirement without echoing credentials.
- If a secret appears to have been entered as a variable name, suppress it in errors and offer a return to masked key entry. Do not silently store it.
- Suggest an alternative export filename when a file exists; never overwrite automatically.
- Suggest a likely command spelling, but do not execute the inferred command.
- Diagnose a quota or authentication failure before offering a retry that is unlikely to help.

### Results and Saved Runs

Provide run selection by question, date, depth and status rather than requiring users to copy IDs. A completed run offers Summary, Findings, Sources, Citation passages, Diagnostics and Export. Failed runs offer recovery actions instead of pretending a report exists.

Start with paged, sanitized Rich Markdown and a static fallback. Textual's Markdown viewer is a later option for a table of contents and persistent navigation. Preserve exact quotes and passage IDs, local/offline access and exclusive-create exports. Do not automatically open a browser or write an export merely because a run finished. [Rich paging](https://rich.readthedocs.io/en/latest/console.html#paging), [Textual Markdown](https://textual.textualize.io/widgets/markdown/)

## Compatibility and Safety

- Graphical mode requires an interactive input and display stream. Suppress it for JSON, CI, noninteractive mode and dumb terminals; an unset `TERM` on Windows alone should not imply a dumb terminal.
- Keep JSON stdout free of banners, prompts, progress and ANSI. Rich Live redirects stdout/stderr by default; use an explicit stderr console with redirection disabled, or bypass Live entirely in machine-readable mode. [Rich Live](https://rich.readthedocs.io/en/latest/live.html#redirecting-stdout-stderr)
- Respect `NO_COLOR` in both rendering and prompts, but distinguish no color from plain/static output. Rich can retain bold/dim styling with `NO_COLOR`. [Rich environment behavior](https://rich.readthedocs.io/en/latest/console.html#environment-variables)
- Use textual status labels and selection markers, not color alone. Support keyboard-only operation, narrow terminals, ASCII fallback and a plain mode. Screen-reader usability requires manual testing, not a library-based assumption.
- Test PowerShell 7/Windows Terminal and Windows PowerShell 5.1/conhost. Do not assume `less`, Bash shortcuts or a particular shell escape syntax. Microsoft notes that `$PSStyle.OutputRendering` does not control native executable output, so Python must manage its own terminal behavior. [Microsoft ANSI guidance](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_ansi_terminals?view=powershell-7.5)
- Treat research questions, filenames, document titles and generated text as untrusted display content. Prevent terminal-control or markup injection when adding richer rendering.
- Textual would require a worker boundary for the synchronous pipeline. Cancellation of a thread worker does not stop an in-flight service request; cancellation must be cooperative at safe boundaries. Keep database ownership and message passing explicit. [Textual workers](https://textual.textualize.io/guide/workers/)

## Implementation Phases

| Phase | Scope | Completion gate |
| --- | --- | --- |
| 1: Interaction foundation | Presentation/prompt adapter; explicit home; progressive setup; immediate validation; Back/Cancel; secure credential entry. | Existing command and JSON tests stay green; keyboard navigation and cancelled drafts are tested. |
| 2: Research workflow | Guided launch; truthful progress; actionable recovery; saved-run browser; paged results and export choices. | Paid actions are explicit; checkpoint/budget behavior is unchanged; offline browsing makes no service calls. |
| 3: Persistent workspace, if justified | Evaluate Textual for a full-screen dashboard, report/source views and managed workers. | Windows, accessibility, cancellation and worker/database lifecycle testing pass. |

Keep prompt/rendering code separate from collection, persistence and research execution. Reuse `wizard.py`'s staged-secret concept, `ResearchPipeline`'s progress/checkpoint behavior, `diagnostics` records and `reports.py` exports. Do not call Typer command functions recursively as an alternate application architecture.

## Acceptance Tests

- Arrow selection, numbered fallback, defaults, Back, Escape, Ctrl+C and EOF.
- No writes before Save; no visible/logged keys; provider changes clear obsolete staged secrets.
- Field-specific validation retains nonsecret drafts and handles key-versus-variable mistakes without echoing secrets.
- JSON purity and unchanged schemas/exit codes; redirected stdin/stdout/stderr, CI and dumb-terminal cases never hang.
- Layouts at 40, 80 and 120 columns; long questions/model IDs and Unicode Windows paths remain usable.
- Local browsing, configuration and diagnostics make zero service calls.
- Research launch, live checks and paid resume require the appropriate explicit action; budgets and saved stages remain intact.
- Unknown-duration operations never display fabricated completion percentages.
- Failed exports preserve existing files; incomplete research cannot be shown as a completed report.
- Plain mode, monochrome mode, keyboard-only operation and manual screen-reader checks.

Retain Typer command-contract tests. Add prompt_toolkit pipe-input tests for interaction, and Textual `run_test(size=...)`/Pilot tests only if that phase is adopted. [prompt_toolkit testing](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/unit_testing.html), [Textual testing](https://textual.textualize.io/guide/testing/)

## Research Boundary

The original research task added this document only. Implementation followed in a separate user request. Library capabilities above are sourced; screen layouts, defaults, sequencing and phase boundaries are EvidenceLab-specific design recommendations.
