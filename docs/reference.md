# EvidenceLab

A technical research CLI with user-selected models, Elasticsearch evidence retrieval,
and optional Jev reranking.

## Current Status

Guided configuration, model profiles, document/URL ingestion, local evidence snapshots,
Elasticsearch hybrid retrieval, and a bounded research workflow are implemented.
Research includes planning, web discovery, gap review, cited reports, resume, and offline
Markdown/JSON export, optional Jev reranking, and an explicit live smoke-test command.
Conversational follow-up, autonomous subagent delegation, and evaluation commands
are not implemented yet.

## Interactive Workspace

From the project directory in PowerShell:

```powershell
uv run evidence ui
# Bare invocation also opens the workspace in an interactive terminal:
uv run evidence
# Static numbered menus for limited terminals or a simpler display:
uv run evidence ui --plain
```

The workspace includes keyboard menus, staged configuration, guided research launch,
saved-run browsing, paged reports and citation snapshots, diagnostics, export, local
document import, and readiness checks. Use arrow keys or numbered shortcuts and Enter.
Escape returns from a field/view; Ctrl+C cancels the current task or exits at home.
In plain text fields, `:back` and `:cancel` provide navigation; secret fields never
interpret their contents as commands. File paths support completion and quoted paths
with spaces. Longer lists and reports are paginated.

Setup asks for a credential source before a variable name, validates fields immediately,
and masks both key entry and custom variable-name entry to prevent accidental key echoes.
Back preserves nonsecret drafts. Changing provider/endpoint discards staged model keys.
No keys or configuration are written before Save. Partial keyring/file failures report
how many keys were stored; those two stores do not form one atomic transaction.

Guided launches display model, depth, sources and optional services before a separate
paid confirmation (default: Back). Live model tests, integration smoke tests and resume
are also explicit paid actions. Readiness, saved-run browsing, diagnostics and exports
are local. Import downloads supplied URLs but does not index or call models. Progress
shows the current operation and elapsed time, not an invented overall percentage.
Interrupting does not guarantee that an in-flight provider request is cancelled.

Completed reports remain in the local SQLite store until you explicitly export them.
The workspace shows a summary or paged report instead of dumping the full report at
completion. Exports never overwrite existing files. Changing configuration does not
change a saved run's model; resume uses its original settings and remaining budgets.

All existing commands, `--json` schemas and noninteractive setup remain available.
The UI never opens for JSON commands, redirected input/output or CI. Bare invocation
shows help outside a supported terminal. `TERM=dumb` requires explicit `ui --plain`.
`NO_COLOR` disables color in both menus and output; `--plain` also disables live redraws.
With a supported terminal and no setting overrides, `evidence configure` uses the new
guided setup. Flag-based configuration keeps its existing command behavior.

## Development Setup

For a fresh installation, start with the [repository README](../README.md#quick-start).

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required. The development
environment uses Python 3.13, as selected by `.python-version`.

```powershell
uv sync --locked
uv run evidence --help
uv run evidence --version
```

`uv` manages the repository-local `.venv`; manual activation is unnecessary.
Commit `uv.lock` to preserve the resolved dependency versions. After intentional
dependency changes, use `uv sync` to refresh the lock and environment.

## Credentials

`.env.example` documents the expected environment variables. A populated `.env`
is ignored by Git. Keys are not needed for help, version, or local static checks.
To load a local `.env` explicitly, use `uv run --env-file .env evidence --help`.
The wizard can save keys in the OS credential manager. Configuration files never
contain API keys. Environment variables take precedence over stored credentials.

Live research will require an Elastic Cloud endpoint and a selected model provider.
Web search additionally uses Tavily; optional Jev reranking uses TypeSafe. Local
document passages will also be processed by the configured cloud services.

## Guided Setup

```powershell
uv run evidence configure
uv run evidence models providers
uv run evidence doctor
```

Choose OpenAI, Anthropic, Gemini, OpenRouter, or a custom OpenAI-compatible endpoint,
then enter the model ID from that provider's catalog. There is no forced default
model. The wizard also offers research-depth settings, optional Jev ranking,
Elastic Cloud settings, and credentials for search and reranking. These research
settings are saved for research jobs; configuring them does not start a job.
Selecting Jev requires a TypeSafe credential; service failures stop the run without
silently reverting to hybrid ranking.

The wizard is offline. Save incomplete setup and supply credentials later if needed.
Hidden key entry writes to the OS credential manager only after the final save
confirmation. Cancelling before that confirmation leaves configuration and credentials
unchanged. If secure storage is unavailable, use environment variables; the app does
not fall back to a plaintext credential file. Selecting environment/keep preserves
existing stored keys as a fallback.

Configuration defaults to the platform's user-config directory. Use
`EVIDENCE_CONFIG_FILE` or the global `--config PATH` option to override
it. Avoid committing local configuration even though it contains no API keys.

## Model Profiles

```powershell
# Replace MODEL_ID with a tool-capable model ID available to your account.
uv run evidence configure --non-interactive --profile work --provider openai --model MODEL_ID
uv run evidence configure --profile reviewer
uv run evidence profiles list
uv run evidence profiles use reviewer
uv run evidence profiles show --profile work --json

# Preview a one-off override without changing the saved profile.
uv run evidence profiles show --profile work --provider gemini --model MODEL_ID --json
```

The first profile becomes the default. Later profiles only become default when you
choose them in the wizard, supply `--set-default`, or run `profiles use NAME`.
Editing an existing profile preserves settings you do not override. Changing its
provider clears the old model, endpoint, and credential reference. Credential-manager
entries are scoped to the profile, provider, and custom endpoint.

Custom endpoints require `--provider compatible --base-url https://HOST/v1`.
For a loopback server, HTTP is allowed. Use `--no-api-key` for an endpoint that needs
no authentication, and `--api-key-required` to re-enable it. The OpenAI-compatible
adapter supports standard chat-completions fields; it does not guarantee support
for provider-specific extensions. OpenRouter uses its dedicated adapter.

Noninteractive setup accepts `--api-key-env VARIABLE_NAME`, `--depth`, `--reranker`, `--jev-model`,
`--elasticsearch-url`, and `--inference-id`. It never accepts an API key on the command
line. A custom credential environment variable supports separate accounts/profiles.
`ELASTICSEARCH_URL` and `EVIDENCE_INFERENCE_ID` override saved service settings at runtime.

## Readiness and Capability Checks

```powershell
# Offline: checks configuration and whether credentials are present.
uv run evidence doctor --profile work --json

# Online: sends one inference request to the explicitly selected model.
# API charges may apply; there are no automatic retries or model substitutions.
uv run evidence doctor --profile work --model-check
```

The model probe checks that the model returns a tool call with schema-valid arguments
and the requested values. It does not certify overall research quality or native
JSON-schema output support. Probe tracing is disabled. Errors are reported without
raw provider responses or credential values.

Ordinary `doctor` does not test service connectivity or authenticate credentials.
It reports Elastic, Tavily, and TypeSafe configuration separately; successful model
readiness does not imply research services are ready. Missing selected-profile
credentials return exit code 1; invalid configuration or failed model probes return 2.
`--json` writes machine-readable data to stdout and probe notices to stderr.

## Credentialed Smoke Test

Configure your chosen model, Elasticsearch endpoint/key and embedding inference ID
locally first. Never paste credentials into chat or command arguments. The smoke test
uses the normal environment/keyring credential resolution and is **live and billable**:

```powershell
uv run evidence configure
uv run evidence smoke-test --json
# Also check Tavily with one search:
uv run evidence smoke-test --web --json
# Explicitly load a local .env when using that credential setup:
uv run --env-file .env evidence smoke-test --web --json
```

The command preflights all selected credentials before making requests. It checks one
model capability probe, indexes two synthetic passages, performs one hybrid retrieval
(lexical plus semantic requests), and extracts a strictly typed integer answer checked
against the known fixture value and its exact supporting citation. Jev-enabled profiles
add one relevance-ranking request; `--web` adds one Tavily query without fetching result
pages and requires `TAVILY_API_KEY` or a saved Tavily key. Tavily is skipped by default
and with `--no-web`. It does not read your documents.
There are at most two generative-model requests and no automatic request retries.

Checks report `passed`, `failed`, `not_run`, or `skipped`. Missing setup exits with code
2 before live checks; a failed live check exits with code 1 and stops subsequent checks.
JSON goes to stdout and notices to stderr. The output includes the run ID when one was
created. Model-probe failures include an allowlisted `error_code` (such as `rate_limit`
or `timeout`) and a safe explanation, never a raw provider response. Synthetic snapshots
and indexed passages are retained for `inspect`; runs are
marked `purpose: smoke`. Nothing is deleted automatically, and rerunning creates a new
small run. A smoke run is not resumable research. A pass checks connectivity and this
small fixture, not broad research quality, ranking quality, or all model capabilities.

## Ingest and Retrieve Evidence

Start with local extraction; this does not require service credentials or a model:

```powershell
uv run evidence ingest --file .\README.md --local-only
uv run evidence runs
uv run evidence inspect RUN_ID --sources
```

Replace `RUN_ID` with the ID printed by `ingest`. Repeated `--file` arguments accept
Markdown, UTF-8 text, text PDFs, or directories. Directories are scanned recursively,
excluding hidden entries, symbolic links, and junctions. Explicit public URLs are also
supported, including HTML articles and PDFs:

```powershell
uv run evidence ingest --url "https://www.elastic.co/search-labs/blog/deep-agents-langchain-elasticsearch-research-pipeline" --local-only --json
```

`--local-only` skips cloud indexing; URL inputs still make public HTTP requests.
Sources are limited to 20 MB and PDFs to 300 pages. Encrypted and scanned PDFs are
reported as unsupported. Partially readable PDFs include warnings for textless pages.
Malformed or inaccessible inputs are reported individually while valid inputs are saved.

To index saved evidence, first configure an Elastic Cloud URL, API key, and an existing
embedding inference endpoint using `configure` or the corresponding environment variables.
This milestone does not create your cloud account or provision an inference endpoint.

```powershell
uv run evidence index RUN_ID
uv run evidence search RUN_ID "How is evidence evaluated?"
uv run evidence search RUN_ID "How is evidence evaluated?" --limit 5 --json
uv run evidence inspect RUN_ID --passage PASSAGE_ID --json
```

Indexing sends extracted passages to Elasticsearch and its configured inference service;
embedding charges may apply. Live search also invokes semantic inference. Without
`--local-only`, `ingest` performs indexing automatically after saving snapshots. No
generative-model calls, Tavily searches, or Jev requests occur in this workflow.

Each result includes the source, exact passage text, passage ID, retrieval date, and
page or line locator. Text/Markdown line numbers refer to the normalized text snapshot;
HTML lines refer to extracted article text, not HTML source lines. PDF citations retain
physical page numbers. `inspect` works offline against the same saved snapshots.

Lexical and semantic queries are independently restricted to the chosen run. Their top
30 candidates are fused with reciprocal-rank fusion, then diversified by source. The
default output is up to 12 passages, not a generated answer. Returned evidence is checked
against local snapshots before display. Scores are ranking signals, not confidence values.

Snapshots live in the OS application-data directory, or `EVIDENCE_DATA_DIR`. They contain
document text and are not encrypted by this application. Keep that directory private.
Files are stored in a local SQLite database; source documents are never edited.

Use `--run-id RUN_ID` to add sources to an existing run. Within a run, the first snapshot
of a location is retained; reingestion does not silently replace evidence with changed
content. Identical normalized content is deduplicated, retaining alternate locations as
aliases. Start a new run to research a newer version of the same source.

The selected profile supplies the depth, overridable with `--depth` on a new run.
Without a profile, depth is `thorough`. Source limits are 10/25/50 for quick/balanced/thorough;
an existing run keeps its original limit. Source input sets over that limit are rejected.
Adding new evidence makes a previously indexed run require indexing again.

Embedding endpoints determine versioned index names. Runs remember their cluster and
embedding endpoint; switching either requires a new run. The API key needs permission
to read inference endpoint metadata, create/read/write the evidence index, and refresh it.
Indexing failures keep local snapshots. Retry `evidence index RUN_ID`; stable passage IDs
prevent duplicate records, though retrying may recompute embeddings and incur charges.
Concurrent updates to one run are blocked. Partial ingestion returns exit code 1 with
the saved run ID and skipped-input details. JSON output stays on stdout; notices use stderr.

## Research and Reports

After configuring a tool-capable model, Elasticsearch, and Tavily:

```powershell
uv run evidence research "When should a technical docs search use hybrid retrieval?" --depth quick
uv run evidence research "Compare retrieval approaches for exact API identifiers" --profile work --json
```

Supply starting URLs/files with repeated `--url` and `--file`. They are read before
discovered sources. To restrict research to supplied sources and skip Tavily:

```powershell
uv run evidence research "What are this system's constraints?" --no-web --file .\README.md --depth quick
```

`--no-web` disables discovery, not cloud processing. Explicit URLs are still fetched;
the selected model, Elasticsearch, and embedding endpoint still receive your evidence.
Queries may disclose the research topic to Tavily when discovery is enabled. Avoid
researching confidential documents with web discovery unless that disclosure is acceptable.
There is no forced model: use saved profiles or one-off `--provider`, `--model`, and
`--base-url` overrides, with the same rules as `profiles show`.

LangGraph orders planning, gathering, indexing, evidence review, and report writing.
Restricted Deep Agents calls generate schema-validated plans, reviews, and reports
using your chosen model. The application, not the model, owns network access and
source limits. Shell, filesystem, subagent, and automatic summarization tools are
disabled. LangSmith tracing is disabled for research. This first implementation uses
sequential stages, not autonomous parallel researchers.

| Depth | Angles | Sources | Web searches | Gap passes | Model requests |
| --- | --- | --- | --- | --- | --- |
| quick | 2-3 | 10 | 6 | 0 | 30 |
| balanced | 3-4 | 25 | 12 | 1 | 60 |
| thorough | 4-6 | 50 | 24 | 2 | 120 |

These are ceilings, not targets or dollar budgets. Each review retrieves up to four
passages per angle, round-robin, with at most 48,000 evidence characters total and
4,000 characters per passage. Reports describe only that evidence window; they are
not exhaustive audits of every ingested page. Use a model with sufficient context
and reliable structured tool calling. `doctor --model-check` tests basic tool calling,
not context capacity or research quality. Embedding calls are separate from model
request counts. SDK/network failures consume reserved attempts; no automatic provider
fallback or unlimited retry loop is used.

Research permits at most one schema-correction attempt per generation call, with a
visible progress notice and a reservation against the same persisted model-call budget.
This handles invalid item counts or citation choices without dropping data or weakening
validation. A second invalid response stops the stage. Service/authentication errors
are not retried, and smoke-test request limits are unchanged. Planner schemas reflect
the selected depth and prohibit search queries when web discovery is disabled.

Run IDs and progress are printed to stderr, leaving `--json` stdout machine-readable.
Inspect status, recover a failed/interrupted stage, or export a completed report:

```powershell
uv run evidence runs
uv run evidence inspect RUN_ID --json
uv run evidence diagnostics RUN_ID
uv run evidence diagnostics RUN_ID --json
uv run evidence resume RUN_ID
uv run evidence export RUN_ID --output .\report.md
uv run evidence export RUN_ID --format json --output .\report.json
```

SQLite stores application-level stage checkpoints, plans, discovered URLs, evidence
IDs, warnings, request counters, and the completed structured report. Resume starts
at the saved stage, preserving the original model/profile settings and remaining
budgets. Credentials are resolved afresh, never stored in the job. Restore the original
Elasticsearch URL/inference endpoint before resuming. Completed runs can be resumed
for display or exported without making service calls. Existing export files are
never overwritten. JSON exports include research metadata and the retrieved passages.

Completed stages are not replayed. A crash between a service response and its local
save can repeat that request on resume and incur another charge; reservations persist
before requests. Once retrieval finishes, its selected review evidence is checkpointed so a
model-output failure can resume without repeated searches or Jev charges. Adding
evidence or changing the plan invalidates that checkpoint. Ranking caches are independent
of candidate ordering and include the excerpt-window version. Old ranking audits retain
their recorded 1,500-character limit; new requests use the review's 4,000-character window.
Jev receives the research question as structured shared state and each candidate only
in its own scoring question. Large candidate sets are split using conservative UTF-8
byte budgets (not exact token counts), without shortening these excerpts. Each batch
counts against the existing Jev request limit and is cached immediately, so a later
batch failure does not discard successful batches. Scores are merged before selecting
the best passages. The new request-layout version invalidates older ranking caches
for unfinished reviews; completed reports remain unchanged.
Successfully saved sources and searches are reused within a gathering
stage. Failed source fetches are listed as warnings, not retried automatically. Add a
missing source with `ingest --run-id RUN_ID --local-only` before resuming gathering,
or start a new run. Force-stopped processes may remain marked `running`; the OS releases
their run lock and `resume` can recover them. Stage checkpoints are not conversational
LangGraph checkpoints or time-travel history.

### Failure Diagnostics

`diagnostics` is local and read-only. Each research job retains its latest 500 safe
events, including stage/request duration, persisted request counters, schema-validation
field paths, allowlisted failure categories, numeric HTTP status when available,
model finish reason, and reported token counts. Failed CLI JSON includes the last 10
events; `diagnostics --json` and `inspect --json` include the retained history.
Older jobs start with an empty history; resuming adds new diagnostics without changing
their saved model settings. Completed resumes do not add events or make service calls.

Categories distinguish timeouts, connectivity, authentication, quota/rate limiting,
provider failures, malformed responses, truncated output, missing/invalid tool calls,
and schema errors. Jev's `max_tokens_exceeded` rejection is reported as `context_limit`,
not an invalid model. Jev events include candidate counts and UTF-8 request/state sizes;
these byte counts are not token counts. Unknown errors stay explicitly unknown. No raw exception strings,
provider bodies, keys, headers, prompts, tool arguments or source text are recorded in
diagnostic events. Jev validation failures retain only finite numeric score/distribution
details and a candidate index, not passage content. A successful HTTP response does
not imply a schema-valid or factually correct answer. Token counts may be unavailable
for failed calls; diagnostic counters are not billing totals.

Every summary, finding, and tradeoff must contain passage references and exact quotes.
Review and report models select short excerpt IDs from request-local allowed choices;
the application inserts each excerpt's original passage ID and exact snapshot text.
Models cannot supply replacement quote text or pair an excerpt with another passage.
Saved runs and exports retain full passage IDs and quotes. Valid quotes in older review
checkpoints remain selectable when resuming. Prompts also spell out output-size limits.
Schema failures identify known field paths and safe error categories; citation
failures identify the claim/citation position and distinguish unknown IDs from
non-verbatim quotes. Neither diagnostic includes model output or quoted source text.
The application rejects unknown IDs, modified retrieval results, and quotes absent
from the supplied snapshots. It renders citations itself rather than accepting
model-generated bibliography links. **This verifies provenance, not entailment or
truth.** A real quote can still be misinterpreted or come from an unreliable source.
Review the report's quotations, gaps, dates, and limitations before relying on it.
Prompts instruct the model to treat source content as untrusted, but cannot guarantee
immunity to prompt injection. Reports escape HTML, Markdown syntax, and control
characters in generated prose. JSON retains original text for inspection.

## Jev Reranking

Jev is optional and does not replace your selected research/report model. Configure
the TypeSafe key through the wizard or `TYPESAFE_API_KEY`, then enable it on a profile:

```powershell
uv run evidence configure --non-interactive --profile work --reranker jev
uv run evidence smoke-test --profile work --json
uv run evidence research "When does hybrid retrieval help?" --profile work --depth quick
```

The example assumes `work` already exists. `--jev-model MODEL_ID` selects a TypeSafe
model ID; the default is `jev-latest`. The saved choice survives profile edits and
research resume. The endpoint is explicitly `https://api.typesafe.ai`; ambient SDK
base-URL/model environment variables cannot redirect the integration. This version
does not use TypeSafe-compatible gateways.

For each research angle, hybrid retrieval supplies up to 12 candidates. After checking
all candidates against local snapshots, the application sends the question and up to
4,000 characters per candidate to TypeSafe, matching the review window. This can increase
reranking input cost relative to the previous 1,500-character limit. Size-bounded requests ask a separate
relevance question for each candidate using the official
[Score primitive](https://docs.typesafe.ai/primitives/score). Scores run from 0 to 3
against a descriptive relevance rubric; they are not factual correctness probabilities.
The four highest-scoring passages continue to review, with ties preserving hybrid
order. Citation IDs, original text, and hybrid scores remain unchanged.

Jev requests are separately capped at 6/16/36 for quick/balanced/thorough. Attempts are
reserved before sending, including failed attempts. Successful scores, confidence,
returned model version and candidate fingerprints are saved locally and included in
research JSON exports. Resuming reuses scores for the same query, candidates and
configured model. Different candidates trigger new scoring within the remaining budget.
The `jev-latest` alias can change between uncached calls; pin a version for reproducible
experiments. Reports list the versions actually returned.

Missing, malformed, non-finite or out-of-range scores are rejected. Authentication,
quota and transport failures stop review; `resume` retries within the saved budget.
Failures report sanitized causes without raw response bodies or passage text. The smoke
test exposes codes such as `jev_timeout`, `jev_rate_limit`, `jev_score_mismatch`, and
`jev_wrong_order`, distinguishing service errors from rejected scores or poor ranking.
There is no automatic retry or ranking fallback. Use `--reranker none` for **new** runs
to disable Jev; changing a profile does not change an existing run's saved settings.
Plain `evidence search` continues to return hybrid results without calling Jev.

Score consistency retains the existing 0.02 tolerance (plus floating-point epsilon).
For scores/probabilities on a two-decimal grid only, a larger difference is accepted
only if independently rounded values could describe a normalized distribution with
that score. This is a mathematical rounding check, not an unrestricted tolerance
increase or a provider precision guarantee. Reported scores are never rewritten;
diagnostics count these rounding matches. Invalid ranges, non-finite values, missing
candidates/rubric levels, invalid probability sums and infeasible scores still fail.

Jev sees truncated excerpts, so relevant material later in a passage can be missed.
Source text is treated as untrusted, but model-based relevance scoring is not immune
to prompt injection. SDK payload logging is disabled during reranking. Local records
and JSON exports contain research metadata and evidence and should be kept private.

## Development Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Tests isolate configuration in temporary directories, replace credential storage,
and mock inference responses and Elasticsearch. They cover citation offsets, PDF pages,
deduplication, input limits, redirect validation, run isolation, partial-index recovery,
and the ingest/search/inspect command flow. Research tests exercise the installed
Deep Agents and LangGraph runtimes with a fake model, covering structured output,
citation rejection, call limits, gap discovery, failed-stage resume, and offline export.
Jev adapter tests use the installed TypeSafe SDK with a mocked HTTP transport. They
cover score validation, request shape, fixed endpoint/model selection, stable ties,
rate limits, persisted budgets and resume caching. Smoke tests mock service calls;
the normal test suite never runs the credentialed smoke command automatically.
They require no paid API calls; live service compatibility needs a separately configured
smoke test with your account.

## Architecture Reference

Inspired by Elastic's [Deep Agents research pipeline](https://www.elastic.co/search-labs/blog/deep-agents-langchain-elasticsearch-research-pipeline).


