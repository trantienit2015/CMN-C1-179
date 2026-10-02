# CMN-C1-179 — Design Specification

> Template: **CMN-C1-179** — RL Policy Drift & Anomaly Detection Agent
> Category: **Cat 1** (single technical capability, industry-agnostic)
> Issue: #1 (`[impl] Design docs/02_design.md`)

## 1. Architecture & Inheritance

- **L1 Base**: `AgentBaseGraph` (L1 direct inheritance, 2026-05-18 policy). No
  Level 2 base agent and no Level 0 (`agenticstar`) import.
- **The graph IS the agent**: `RLAnomalyDetectionAgent(AgentBaseGraph)` in
  `src/graph/graph.py` is the single class — there is no separate agent class and
  no second graph (no double-graph; review criterion #10). It exposes the `name` and
  `state_schema` properties and fills the backbone slots in `register_nodes()`.
- **Public entry point**: `RLAnomalyDetectionAgent().compile()` then
  `agent.invoke(user_input, ctx=InvocationContext(...), input_context={...})` —
  **never `.run()`** (criterion #11). The framework owns `__call__`, routing,
  retry, timing, error capture, and the S-1 trust gate.
- **Trust level**: per-agent `required_trust_level: "INTERNAL"` in
  `config/agent.yaml`, plus `required_trust_level = TrustLevel.INTERNAL` on the
  privileged domain nodes — a valid production enum value (criterion #13). RL
  output logs are internal operational telemetry, so INTERNAL is the appropriate
  floor.

### Three-layer separation

| Layer | This template |
|-------|---------------|
| State | `RLAnomalyState(AgentState)` — flat TypedDict, primitives + JSON strings only (`src/schemas/state.py`). |
| Node | 3 `FunctionNode` slot subclasses, each overriding `execute(self, state) -> dict` (no `config` param), returning a partial dict with an `AgentStatus.*` enum. No `__call__` override, no `_invoke_impl`. Domain logic lives in `src/services/`. |
| Graph | `RLAnomalyDetectionAgent` calls `super().register_nodes()` then fills `pre_process`/`main`/`post_process`. It does NOT override `add_edges()` or `route()`. |

## 2. Pipeline (Cat 1 backbone + 7-step → slot/service mapping)

The framework provides the fixed 5-node backbone:

```
initialize → pre_process → main → {route} → post_process → finalize
```

The 7 documented RL-anomaly steps are mapped onto the three domain slots; each
step's logic is extracted **verbatim** into a deterministic `src/services/`
function so no business logic is collapsed:

| Slot | Step (service fn) | Input | Output | Key logic |
|------|------|-------|--------|-----------|
| `pre_process` | 1 InputParse (`rl_parsing.parse_logs`) | `input_context.raw_logs`, `input_context.rl_framework` | `parsed_logs` | Per-framework (ART/GRPO/OpenAI RL) normalization + S-1 sanitization (HTML strip + truncate). |
| `pre_process` | 2 BaselineLoad (`baseline.validate_baseline`) | `input_context.baseline` | `baseline` | Validate baseline policy (`reward_mean`, `reward_std` required). Fail-fast (`AgentStatus.ERROR`) if absent/invalid. |
| `main` | 3 RLRewardDriftCompute (`drift.compute_deviations`) | `parsed_logs`, `baseline` | `deviation_scores` | Deterministic reward-distribution delta (\|z-score\|). Optional semantic distance (§3). |
| `main` | 4 PolicyDeviationScore (`scoring.score_entries`) | `deviation_scores` | `scored_entries` | Threshold comparison (`drift_threshold`) → per-entry score + `threshold_exceeded` flag. |
| `main` | 5 AnomalyClassify (`classify.classify_anomalies`) | `scored_entries` | `classified_anomalies` | 3-class rule-based classification: `policy_drift` / `reward_hacking` / `behavioral_collapse` + severity. |
| `main` | 6 AlertGenerate (`report.build_report`) | `classified_anomalies` | `alert_report` | Config-format structured report (ISO 42001 / NIST AI RMF / EU AI Act / FSA AI MRM / generic). No hardcoded format. |
| `main` | 6b NarrativeGenerate (`MainNode._generate_narrative`, optional) | `alert_report` | `alert_report.narrative` | LLM (Azure OpenAI) executive summary of the already-built report. Never runs when there are no findings; any failure omits `narrative` (§9). |
| `post_process` | 7 SecurityGateOutput (`masking.*`) | `alert_report` | `masked_report`, `formatted_output`, `audit_log` | **S-3** deterministic masking (regex, not LLM) + **S-5** ISO 42001-style audit trail + **S-4** trace event. |

Early-exit: `pre_process` sets `AgentStatus.ERROR` on a bad baseline; `main`
short-circuits when `baseline` is absent; the framework `route()` then sends an
ERROR status straight to `finalize`.

## 3. sentence-transformers dependency (semantic layer)

The semantic-similarity scoring in the drift step depends on the
`sentence-transformers` package, which is not a declared dependency of this
template. Therefore:

- The drift service ships a **deterministic reward-distribution-delta core**
  (always available, the primary detection signal).
- Semantic similarity is an **optional hook**: a `semantic_scorer` callable
  injected via `input_context["semantic_scorer"]` is used only if present. Absent
  ⇒ `semantic_dist` is `None` and detection runs on the deterministic delta alone.
  A misbehaving injected scorer is caught and logged at WARNING (with
  `correlation_id`), never silently swallowed.
- `sentence-transformers` is **NOT declared in `pyproject.toml`** and is never
  hard-imported, so installing the template does not pull it in.
- `semantic_weight` defaults to `0.0`, keeping scoring fully deterministic unless
  a scorer is wired.

The optional layer does not block the template.

## 4. State schema (`src/schemas/state.py`)

`RLAnomalyState(AgentState)` — flat, msgpack-safe. Shared fields (user_input,
input_context, status, session_id, correlation_id, node_history, error_log,
hitl_*, …) are inherited from `AgentState`. Structured collections are stored as
**JSON-serialized strings** (helpers in `src/services/serialization.py`). No
Pydantic/dataclass, no credentials, no `InvocationContext` in state (the framework rules
§4). Raw inputs (raw_logs / baseline / semantic_scorer / generated_at) arrive via
the framework `input_context` and are read in the slots — never stored as live
objects.

## 5. Configuration (`config/agent.yaml` + `config/config.yaml`)

Configuration is split across two files. `config/agent.yaml` is the discovery
manifest and carries only identity and contract keys, read at root level:
`id`, `name`, `namespace`, `version`, `enabled`, `category`, `industry`,
`generation_mode`, `base_type`, a single dotted `class` path,
`required_trust_level`, and `requires.{secrets, extras}`.
`config/config.yaml` carries the runtime parameters (`max_retry`,
`timeout_seconds`). Domain knobs (`drift_threshold`, `hacking_multiplier`,
`collapse_multiplier`, `report_format`, `semantic_weight`,
`max_observation_chars`) are passed into the node constructors via the graph
`config` dict (with `DEFAULT_*` fallbacks in the services) — not hardcoded in
code (Cat 1 = industry-agnostic).

## 6. Five-layer security model

| Layer | Implementation | Location |
|-------|----------------|----------|
| S-1 | Per-agent `required_trust_level: INTERNAL` (manifest) + `required_trust_level = TrustLevel.INTERNAL` on the privileged nodes; the framework `BaseNode.__call__` gate denies an under-trusted caller before `execute()` runs | `config/agent.yaml` + `src/nodes/*` |
| S-2 | `caller_trust_level` carried on `InvocationContext` by the entry point | `src/api/server.py` / caller |
| S-3 | deterministic masking (`mask_sensitive`, regex) on the serialized report | `post_process` node + `src/services/masking.py` |
| S-4 | `emit_trace_event(event_type, payload, state)` (from `shared.utils.audit_logger`) on the side-effect path | `post_process` node |
| S-5 | ISO 42001-style audit-trail entry (`audit_log`) | `post_process` node |

S-3 fires inside the terminal `post_process` slot, the only place the rendered
output exists. Injection/secret blocking is at S-3 (output), NOT S-1 (input
normalization only) — dev guide §5 / the source proposalruling. Secrets, when needed,
are provisioned at startup via `agent.provision_secrets(secrets_factory(...))`
and bound per-request with `with bound_secrets(agent._secrets_provider):` — never
`os.environ`. Detection itself (steps 1-6) needs no external secret. The optional
narrative enhancement (§9) declares three secrets in `requires.secrets` and resolves
them the same way, per invocation, only inside `MainNode._generate_narrative()`.

## 7. Error handling

Validation failures set `status = AgentStatus.ERROR` + `error_log`; the framework
routes ERROR to `finalize`. The terminal slot still emits an S-5 audit entry and
an S-4 trace on the success path, so no normal outcome is un-audited.

## 8. Deployment entry points

The agent is reachable through two entry points, and both resolve their runtime inputs the same way.

| Entry point | File | Used by |
|---|---|---|
| Standalone HTTP | `src/api/server.py` | staging rehearsal, direct invocation |
| Marketplace one-shot Pod | `cli.py` | the platform runner (container `CMD`) |

Both read runtime parameters from `config/config.yaml` — the manifest carries discovery metadata
only — and both scope secrets to the same location (`namespace=cmn`, `agent_name=cmn-c1-179`), so a
secret provisioned for one path resolves identically on the other.

**No LLM client is constructed at either entry point.** Detection is deterministic end to end. A
client is constructed only inside `MainNode._generate_narrative()`, per invocation, when there is
at least one finding to narrate — see §9.

**Progress events.** Each pipeline stage emits a non-terminal progress event at its start, so a
caller sees the run advancing. Outside the Marketplace runtime the emitter resolves to a no-op, so
the same code is safe on every path. Terminal events belong to the platform runner and are never
emitted by this agent.

**Caller authentication at the standalone entry point.** A caller that no upstream middleware
vouched for stays anonymous unless it presents a deployment-level credential: the external bearer
token grants the verified-external level, and a separate staging-only runner credential is the only
way to reach the internal level. Trust established upstream is never changed.

## 9. Optional LLM-enhanced narrative (Step 8e — graceful degrade, not mandatory)

`MainNode` already produces a complete, valid `alert_report` with no LLM (steps 1-6, fully
deterministic). `MainNode._generate_narrative()` optionally adds one more field, `narrative`, an
executive-summary sentence or two written for the selected `report_format` governance framework —
genuine NL synthesis over the structured findings (which governance framework, which finding types,
what severity mix), not a templated string.

- **Trigger:** only when `classified_anomalies` is non-empty. A clean report is never sent to the
  LLM — nothing to narrate, no reason to spend the call.
- **Secrets:** `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` — all
  three declared in `config/agent.yaml` `requires.secrets` (`requires.extras: ["openai"]`); none of
  it in `config/config.yaml`. Resolved via `ctx.secrets.require(...)` from a fresh
  `InvocationContext.from_state(state)` inside `_generate_narrative()`, never cached — a node
  instance is constructed once in `register_nodes()` (registry LRU cache) and reused across every
  invocation, so a client built from one caller's secrets must never survive to serve the next
  caller.
- **Failure contract — graceful degrade, never fail-closed:** a missing secret
  (`MissingSecret`), an API error, a malformed/wrong-shape JSON response, or no findings at all —
  every one of these is caught inside `_generate_narrative()` and returns `None`. The caller
  (`execute()`) then simply does not add `narrative` to the report. `status` stays `success`; the
  deterministic fields (`format`, `summary`, `findings`) are identical whether or not the LLM call
  ran. This is deliberately the opposite failure contract from a Step 8f (mandatory-LLM) design —
  here the LLM is a pure enhancement, and an Azure OpenAI outage must never turn a working
  detection run into an error.
- **`generation_mode: "llm"`** in `config/agent.yaml` reflects that an LLM path now exists in the
  template, not that detection depends on it.
- **Tests:** `tests/unit/test_main_node.py::TestMainNodeNarrative` — valid JSON, markdown-fenced
  JSON, malformed response, wrong-shape JSON, a raising client, no secret bound (the real
  production shape absent configuration), and the no-findings skip path. All via a test-double
  (`_FakeLLM`); no real network call anywhere in the suite.
