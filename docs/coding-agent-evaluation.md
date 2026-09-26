# Coding-agent evaluation and trajectory lab

This subsystem measures whether the sandboxed coding agent resolves repository failures, not whether its response sounds plausible. Each completed run is content-addressed by dataset and configuration fingerprints and produces a portable artifact directory.

## Metrics

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

- Do not describe the five smoke cases as SWE-bench performance.
- Do not combine results from different dataset fingerprints.
- Report the model identifier, sandbox image digest, dataset fingerprint, configuration fingerprint, sample size, confidence interval, prices used, and run date.
- Keep validation cases for iteration and publish test-split results only after configuration choices are frozen.
- Report failed infrastructure cases and zero-token runs; do not silently remove them from the denominator.
