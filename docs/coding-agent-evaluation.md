# Coding-agent evaluation and trajectory lab

For actual execution results rather than configuration examples, see the
[October 4 full audit](live-upgrade-audit-2026-10-04.md). It preserves original
failures, separates internal smoke completion from fresh artifact replay, and
does not report a held Requests pilot as an official benchmark solution.

The opt-in [counterfactual regression gate](counterfactual-regression-verification.md)
adds `--counterfactual-verification` for verified-PR workflows and inherited
tournament candidates. A completed result must include intact original-fail /
patched-pass evidence; missing proof holds quality publication. Its authored
controls and optional real-container fixture are not live coding-resolution lift.

The [execution-guided retrieval comparison](execution-guided-retrieval.md) adds
`--execution-retrieval` and a two-arm frozen smoke matrix at
`evals/experiments/execution_retrieval_matrix.json`. Its no-provider localization
controls are separate from resolution benchmarks; real comparisons require Docker,
available model credits, and independently reviewed held-out tasks.

The [reviewed repair-memory controls](reviewed-repair-memory.md) validate retrieval
and lifecycle behavior without a provider. Mutable memory remains rejected by the
frozen runner. The [snapshot protocol](frozen-repair-memory.md) allows explicitly
pinned memory experiments with source lineage, overlap auditing, family metadata,
and validated exposure receipts; no historical solution content is silently
introduced into benchmark prompts. Invalid exposure evidence holds quality
publication rather than counting as a valid memory-on measurement.

This subsystem measures whether the sandboxed coding agent resolves repository failures, not whether its response sounds plausible. Each completed run is content-addressed by dataset and configuration fingerprints and produces a portable artifact directory.

## Regression-challenge ablation

The [regression designer](regression-challenge-agent.md) has a Docker-backed baseline/weak-patch/full-fix control: `python -m evals.regression_challenge_evaluation --output data/evaluations/regression-challenges/my-review-v1.json`. Existing happy-path tests cannot distinguish the authored variants; a common behavioral suite is expected to reject the example-specific weak patch. This is one synthetic fixture, not generated-oracle accuracy or live-model resolution lift. Docker absence returns unavailable rather than executing on the host. CI uploads JSON/Markdown without activation.

The [oracle-calibration controls](oracle-calibration.md) distinguish correct-reference agreement, insufficient fault sensitivity, and wrong expectations. `python -m evals.oracle_calibration_evaluation --output data/evaluations/oracle-calibration/my-review-v1.json` uses the Docker sandbox and no provider. The four authored faults and greedy probe-cover diagnostic are not general test adequacy or live-model quality measurements.

## Metrics

For interruption-safe runs, use the optional `--checkpoint-dir` and follow
[provider-aware recovery](provider-recovery.md). Terminal provider failures pause
the batch. `execution_status`, `quality_metrics_valid`, pending case IDs, retained
attempts and unknown recovery usage distinguish operational evidence from a complete
quality evaluation. Numeric legacy score fields are not publishable when the validity
flag is false. Recovered repeated attempts cannot be labelled Pass@1.

The report records pass@1 with a deterministic bootstrap 95% confidence interval, final-test pass rate, verification pass rate, repair rounds, blocking findings, regression rate, timeout rate, sandbox-policy rejection rate, p50/p95 duration, average iterations/tool calls/changed files, provider-reported input/output tokens, estimated cost, cost per resolved task, failure categories, and repository/tag slices.

A case is resolved only when the agent reports completion, produces a non-empty patch, changes at least one file, passes the fixed final test command without timing out, and operates only on the disposable repository copy.

## Credential-free smoke dataset

Five intentionally failing Python repositories live under `tests/fixtures/code_repositories/`. The versioned dataset contains validation and test splits plus behavior tags:

```bash
python -m code_agent.evaluation evals/datasets/code_agent_smoke.jsonl --validate-only
```

Run a live comparison after building the sandbox image and configuring provider credentials:

```bash
python -m code_agent.evaluation evals/datasets/code_agent_smoke.jsonl \
  --repository-root tests/fixtures/code_repositories \
  --dataset-name agentforge-code-smoke-v1 \
  --workflow verified_pr \
  --model gpt-4o-mini \
  --model your-strong-model \
  --input-cost-per-million 0 \
  --output-cost-per-million 0
```

Replace the zero price fields with the prices applicable at the time of the run. The tool never embeds mutable pricing assumptions in source code.

`verified_pr` is the CLI default. Use `--workflow single_agent` only for an explicit ablation against the independently reviewed workflow; the selected workflow is included in the configuration fingerprint and report.

## SWE-bench compatibility

Export a selected official SWE-bench subset as JSON or JSONL, then convert it:

```bash
python -m code_agent.swebench official-swebench.jsonl \
  data/evaluations/swebench-agentforge.jsonl
```

The importer retains the instance ID, repository, base commit, version, FAIL_TO_PASS, PASS_TO_PASS, and environment-setup metadata. It deliberately drops the gold solution patch and hidden test patch so they cannot leak into the agent prompt or trajectory.

Prepare each repository below the selected repository root at its recorded base commit. Real SWE-bench repositories often need different dependency images. Supply an operator-controlled mapping and pre-allowlist those locally built images through the evaluation command:

```json
{
  "django__django": "agentforge/swebench-django@sha256:replace-with-real-digest",
  "pytest-dev__pytest": "agentforge/swebench-pytest@sha256:replace-with-real-digest"
}
```

```bash
python -m code_agent.evaluation data/evaluations/swebench-agentforge.jsonl \
  --repository-root repositories/swebench \
  --dataset-name swe-bench-verified-selection-v1 \
  --image-map data/evaluations/swebench-images.json \
  --model gpt-4o-mini \
  --max-cases 25
```

The generated `predictions.jsonl` uses `instance_id`, `model_name_or_path`, and `model_patch`, so it can be handed to the official SWE-bench evaluator. Always use the official harness for the final published resolved score; AgentForge's local score is an iteration signal.

For SWE-bench Multilingual, pass `--benchmark-name swe-bench-multilingual` during import. The importer retains a normalized `language:<name>` tag when the source row supplies language metadata, enabling language-specific failure and Pass@1 slices.

For controlled model/workflow/retrieval comparisons, use the frozen-matrix and paired-release workflow in [benchmark-release.md](benchmark-release.md). It adds exact case-set locking, paired bootstrap deltas, exact McNemar tests, Pareto analysis, and an integrity-bound Markdown benchmark card.

## Artifact contract

Each run creates:

```text
data/evaluations/code-agent/<run-id>/
  report.json
  predictions.jsonl
  manifest.json
  patches/<case>.diff
  trajectories/<case>.json
```

`report.json` contains aggregate and case metrics without patch bodies. A trajectory contains the issue, test evidence, structured actions, bounded outputs, timings, changed paths, sandbox metadata, and score. Patches are separate. `manifest.json` records the SHA-256 and byte size of every artifact; each outcome also binds its patch by SHA-256.

Trajectories can contain source excerpts and test output. Generated evaluation directories are gitignored; production systems should encrypt them, enforce tenant-scoped access, redact secrets again before persistence, and apply an explicit retention policy.

## Replay dashboard

```bash
streamlit run code_agent/dashboard.py
```

The dashboard compares runs sharing the same dataset fingerprint, displays pass@1/cost/latency/safety metrics and slices, and replays each action and observation. Patch display performs an integrity check and is off by default.

Or launch only the read-only containerized dashboard:

```bash
docker compose --profile evaluation up --build code_eval_dashboard
```

The container runs as UID 10001 with a read-only root filesystem, dropped capabilities, no privilege escalation, a bounded temporary filesystem, and a read-only bind mount containing only generated evaluation artifacts.

## Reporting rules

The [patch-mutation comparison](patch-targeted-mutation-testing.md) uses
`--mutation-testing off|report|enforce` or the frozen `patch_mutation_matrix.json`.
Report/checkpoint summaries retain statuses, measured cases, killed/survived/invalid
mutants, and completed trials across retained review rounds. These are actions,
not net final test strength or all losing-team results. Missing/inconsistent
configured evidence withholds completed-case resolution and cost-per-resolution
claims. Validate the matrix separately from executing it; its five authored smoke
cases cannot establish real-issue gains. Report mode can influence reviewer prompts.

The [cycle-control comparison](agent-cycle-control.md) uses `--loop-guard` or the
frozen `loop_guard_matrix.json` with off/shadow/enforce variants. Reports retain
stage/branch `loop_guard_status_counts` across checkpoint resume. These are
signals, not saved-call estimates or correctness scores; compare resolution,
regressions, total calls, latency, and fully accounted cost on the same tasks.

The [patch-minimization ablation](execution-guided-patch-minimization.md) uses
`--patch-minimization` or the frozen `patch_minimization_matrix.json` with verified
workflows. Reports retain accepted file-removal actions, completed trials, case
exposure, and abstention counts, including checkpoint resume. Measure final patch
size and quality together; missing or invalid completed minimization evidence holds
publication. Authored predicate controls are not live coding-resolution scores.

The optional [spectrum-localization ablation](spectrum-fault-localization.md) uses
`--fault-localization` or the locked `fault_localization_matrix.json`. Outcomes,
reports, and release evidence retain context-selection counts and probe statuses,
including on checkpoint resume. A feature-enabled arm with zero selected contexts
is not evidence of localization benefit. Authored controls and trusted-fixture
instrumentation tests are separate from live coding-resolution measurements.

- Do not describe the five smoke cases as SWE-bench performance.
- Do not combine results from different dataset fingerprints.
- Report the model identifier, sandbox image digest, dataset fingerprint, configuration fingerprint, sample size, confidence interval, prices used, and run date.
- Keep validation cases for iteration and publish test-split results only after configuration choices are frozen.
- Report failed infrastructure cases and zero-token runs; do not silently remove them from the denominator.
## Multi-team repair experiments

The [repair tournament](multi-agent-repair-tournament.md) is supported by the real coding
benchmark harness as `--workflow repair_tournament`. Use
`--tournament-policy evals/experiments/repair_tournament_policy.json` to freeze its budgets.
The `repair_tournament_matrix.json` comparison registers verified-team, one-team challenge,
and three-team challenge arms using the existing smoke dataset lock. No live run is implied.
Reported provider usage includes losing teams and challenge calls when callbacks supply it.
Unreported usage must not be interpreted as zero cost.

Credential-free `evals.repair_tournament_evaluation` controls use scripted model responses
and authored command outcomes. They test orchestration, not model accuracy or patch success
on unseen repositories. CI runs only these controls, not the billable live matrix.
