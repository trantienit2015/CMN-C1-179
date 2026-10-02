# CMN-C1-179 — Test Specification

> Issue: #8 (`[impl] Test spec docs/03_test_spec.md`), #9 (Proof-of-Boundary)
> All tests are deterministic (no network, no optional packages; semantic layer
> disabled by default). This includes the optional LLM-narrative tests (MN-10..MN-15,
> `docs/02_design.md` §9): every one goes through a test-double, never a real Azure
> OpenAI call. Run: `python -m pytest tests/ -v`.
> Layout: `tests/{unit,integration,proof_of_boundary}` (standard `src/` regime).

## 1. Unit tests — per slot node

Each node test asserts the node contract (`execute(self, state)`; second param
named `state`; no `config` param; no `_invoke_impl`) and the step logic. Status
is asserted as an `AgentStatus.*` **enum** (not `.value`).

### `tests/unit/test_pre_process_node.py` (Steps 1+2 — InputParse + BaselineLoad)

| TC | Scenario | Expected |
|----|----------|----------|
| PP-contract | `execute` signature | `params[1] == "state"`, no `config`, no `_invoke_impl` |
| PP-01 | ART framework normalization (+ HTML strip) | `status SUCCESS`, reward/action parsed, observation HTML-stripped |
| PP-02 | GRPO field aliases (`group_reward`/`policy_action`/`obs`) | canonical fields populated |
| PP-03 | Unknown framework | generic adapter resolves reward; parse succeeds |
| PP-04 | Empty batch | `status SUCCESS`, `parsed_logs == []` (valid, not error) |
| PP-05 | Observation truncation cap (constructor) | digest truncated to cap |
| PP-06 | Missing baseline | `status ERROR` + `error_log` |
| PP-07 | Baseline missing required keys | `status ERROR` |
| PP-08 | Negative `reward_std` | `status ERROR` |

### `tests/unit/test_main_node.py` (Steps 3-6 — drift → score → classify → report)

| TC | Scenario | Expected |
|----|----------|----------|
| MN-contract | `execute` signature | `params[1] == "state"`, no `config`, no `_invoke_impl` |
| MN-01 | Within 3σ | 0 anomalies, report status `clean` |
| MN-02 | z == threshold band | `policy_drift` / `medium` |
| MN-03 | hacking band (z ≥ 4.5) | `reward_hacking` / `high` |
| MN-04 | collapse band (z ≥ 7.5) | `behavioral_collapse` / `critical` |
| MN-05 | custom `drift_threshold` (constructor) | reclassifies accordingly |
| MN-06 | `report_format: iso_42001` (constructor) | report `format == iso_42001` |
| MN-07 | unknown format | falls back to `generic` |
| MN-08 | semantic scorer + `semantic_weight=1.0` | scorer path exercised, score == semantic dist |
| MN-09 | baseline missing in state | `status ERROR` + `error_log` |
| MN-10 | LLM narrative — valid JSON response (`TestMainNodeNarrative`, test-double only) | `alert_report.narrative` set to the returned text |
| MN-11 | LLM narrative — markdown-fenced JSON response | still parses via `extract_json_object` |
| MN-12 | LLM narrative — malformed / wrong-shape response | `narrative` key absent, `status SUCCESS` |
| MN-13 | LLM narrative — client raises | `narrative` key absent, `status SUCCESS` |
| MN-14 | LLM narrative — no double injected, no secret bound (real prod shape w/o config) | `narrative` key absent, `status SUCCESS`, never raises |
| MN-15 | LLM narrative — clean report (no anomalies) | LLM never called (`_AssertNotCalled` double) |

### `tests/unit/test_post_process_node.py` (Step 7 — S-3/S-5/S-4)

| TC | Scenario | Expected |
|----|----------|----------|
| PO-contract | `execute` signature | `params[1] == "state"`, no `config`, no `_invoke_impl` |
| PO-01 | clean report | `status SUCCESS`, `masked_report` mirrored to `formatted_output` |
| PO-02 | embedded `sk-…` secret in a finding | redacted (`[REDACTED]`), `audit.s3_redactions_applied true` |
| PO-03 | embedded email PII | redacted |
| PO-04 | audit entry always emitted | `audit.event == rl_anomaly_scan`, `template_id`, `s3_masked true` |

## 2. Integration tests (`tests/integration/test_graph.py`)

| TC | Boundary | Expected |
|----|----------|----------|
| PB-6 | Full pipeline `compile()` + `invoke(ctx=INTERNAL)` | `status == success`; `output` set; `node_history` == Initialize → PreProcess → Main → PostProcess → Finalize (5) |
| INT-clean | no-drift batch | `status success`, output contains `"clean"` |
| INT-err | missing baseline | pre_process ERROR → main short-circuit → route to finalize → `status error` |
| PB-1 | S-1 trust gate, ANONYMOUS caller | blocked before privileged `execute()`; `status error` |

## 3. Proof-of-Boundary tests (`tests/proof_of_boundary/`)

| PB | Boundary | File | Expected |
|----|----------|------|----------|
| PB-2/PB-5 | State safety (no creds/Pydantic/InvocationContext) | `test_state_safety.py` | AST scan of `src/schemas/state.py`: 0 violations |
| PB-4 | Import isolation (no Level 0 `agenticstar`) | `test_import_isolation.py` | AST scan of `src/`: 0 violations |

## 4. the review criteria coverage

- #1 / #7 — L1-direct + 3-layer separation; no `agenticstar`/`mediator` import (PB-4).
- #2 — 5-layer security: S-1 trust gate (PB-1), S-3 masking (PO-02/03), S-4
  `emit_trace_event`, S-5 audit (PO-04). No `os.environ`.
- #10 — graph IS the agent (`AgentBaseGraph`); no double-graph.
- #11 — `execute(self, state)` (no config), status enum, no `_invoke_impl`,
  `.compile()` + `.invoke()` not `.run()` (PP/MN/PO-contract + PB-6).
- #13 — `required_trust_level` valid enum (INTERNAL).
- #14 — `src/` scaffold structure; reference files untracked.

## 5. Deployment path tests

### TC-DEP-01 Marketplace entry point identity — `tests/unit/test_cli_entry_point.py`

| Case | Expected |
|---|---|
| Identity is concrete | `agent_name` / `namespace` are non-empty and not the runner default |
| Identity matches the HTTP entry point | the values equal what `src/api/server.py` passes to `secrets_factory`; where that entry point provisions no secrets, they equal the manifest `namespace` and the lower-cased template id |
| The runner call uses the constants | `run_agent_marketplace` is called with the module constants, not inline literals |

### TC-DEP-02 Standalone entry point boundary — `tests/proof_of_boundary/test_server_llm_injection.py`

| Case | Expected |
|---|---|
| Boots with no credential provisioned | the module imports and the app/agent objects are constructed |
| External bearer never reaches the internal level | resolves to the verified-external level |
| Staging runner credential reaches the internal level | resolves to the internal level |
| Wrong or missing bearer while auth is enabled | rejected with 401 |
