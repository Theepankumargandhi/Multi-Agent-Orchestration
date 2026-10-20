# Coding-agent benchmark release engineering

The [October 4 execution audit](live-upgrade-audit-2026-10-04.md) is a development
pilot, not a comparative release. A smoke candidate needed post-hoc patch-framing
repair, adaptive inference failed on schema validation, and the real issue was
held before inference. No workflow winner, official score, or cost saving is
published from those attempts.

## Why this exists

Single benchmark runs are easy to cherry-pick and hard to compare. AgentForge therefore treats benchmark publication as a frozen experiment release:

1. Select and lock exact case IDs before tuning.
2. Declare every model, workflow, retrieval strategy, and price assumption in a matrix.
3. Run every variant on the identical selection.
4. Compare outcomes at the paired case level.
5. Publish metrics, uncertainty, cost, latency, failures, and artifact hashes together.

The release layer consumes the existing sandbox evaluator. It does not change the definition of a resolved task and does not hide infrastructure failures from the denominator.

Mutable [reviewed repair memory](reviewed-repair-memory.md) is not permitted in
these frozen runs without an explicitly pinned snapshot. The
[snapshot protocol](frozen-repair-memory.md) supports memory-on/off comparisons for
single-agent and verified-PR variants: the candidate supplies both
`repair_memory_snapshot` and `repair_memory_snapshot_sha256`. Validation and
execution audit every candidate against the locked selection before any model
calls. Family metadata and owner-attested source lineage are required. Snapshot
and audit digests enter the run configuration; invalid exposure receipts hold
quality publication. Authored controls remain separate from live coding results.

## Credential-free validation

For the optional original-fail / patched-pass test gate, validate
`evals/experiments/counterfactual_matrix.json`. The two variants differ only in
`counterfactual_verification_enabled`; this adds container work, not direct model
calls. The [implementation guide](counterfactual-regression-verification.md)
describes strict pytest support, evidence holds, and the separate control drill.

The repository includes a four-variant smoke matrix and a frozen five-case lock:

```bash
python -m code_agent.benchmark_release validate \
  evals/experiments/code_agent_release_matrix.json
```

CI runs this command without model credentials. It proves that task content, task order, and the committed dataset fingerprint have not drifted. The matrix compares:

- single-agent plus repository-map context;
- verified PR plus lexical/graph retrieval;
- verified PR plus hybrid/reranked retrieval;
- a stronger-model verified/reranked configuration.

Zero prices are placeholders. Set current provider prices in the matrix immediately before a paid run; the values become part of each run configuration fingerprint.

## Adaptive dispatch comparison

For the static-versus-execution-guided context experiment, validate
`evals/experiments/execution_retrieval_matrix.json`. A variant's
`execution_retrieval_enabled` flag is copied into its sandbox policy and therefore
its benchmark configuration fingerprint. The checked-in arms change only that
flag. This is an experiment template, not a completed live comparison; see
[the execution-retrieval guide](execution-guided-retrieval.md).

The [October 3 live-study attempt](real-task-benchmark-attempt.md) records the actual
readiness results, selected real-task pilot, initial Docker failure, successful
container retry, and subsequent API-quota blocker. No live resolution or cost result
is implied by those preparation artifacts.

`evals/experiments/adaptive_supervisor_matrix.json` freezes three arms on the same smoke
selection: verified PR, a fixed three-team tournament, and the opt-in adaptive supervisor.
The adaptive arm explicitly fixes sequential dispatch, escalation limits, verification
reserves and repeated-failure stopping. It does not change the dataset or activation state.

```bash
python -m code_agent.benchmark_release validate evals/experiments/adaptive_supervisor_matrix.json
```

Use this as a template for independently reviewed held-out tasks after a separate
development pilot. Execution requires prepared repositories/images, provider credentials,
and an approved spending limit. The checked-in matrix is not a completed real-world run.
See [the supervisor guide](adaptive-coding-supervisor.md) for the resource differences
between arms and the credential-free control comparison.

Reports also include regression rate, logical model calls, calls per task, provider-usage
completeness, and price availability. Cost-per-resolved and paired dollar comparisons are
unavailable unless both compared arms have complete usage and configured input/output
prices. Descriptive ranking omits cost unless every arm meets that condition; Pareto
comparisons omit the cost dimension for pairs with incomplete accounting. Hidden SDK
retries are not logical calls, and reported costs remain estimates rather than bills.

## Freeze a real selection

Import an official selection without solution or hidden-test patches. Language metadata is preserved for multilingual slices:

```bash
python -m code_agent.swebench official-multilingual.jsonl \
  data/evaluations/swebench-multilingual-25.jsonl \
  --benchmark-name swe-bench-multilingual

python -m code_agent.benchmark_release lock \
  data/evaluations/swebench-multilingual-25.jsonl \
  data/evaluations/swebench-multilingual-25.lock.json \
  --name swe-bench-multilingual-25-v1 \
  --max-cases 25
```

Create a matrix JSON using the checked-in smoke matrix as a template. Point it to the frozen dataset, lock, prepared base-commit repositories, and operator-built image map. Every repository image must be explicitly allowlisted and should use an immutable digest.

## Execute the matrix

After Docker, repository images, and provider credentials are available:

```bash
python -m code_agent.benchmark_release run \
  data/evaluations/swebench-multilingual-matrix.json
```

Each cell still produces its normal report, predictions, patches, trajectories, and manifest. The release additionally writes:

```text
data/evaluations/benchmark-release/
  runs/<run-id>/...
  releases/<release-id>/
    release.json
    benchmark-card.md
    manifest.json
```

You can also release existing compatible reports:

```bash
python -m code_agent.benchmark_release release \
  baseline=data/evaluations/code-agent/<baseline>/report.json \
  candidate=data/evaluations/code-agent/<candidate>/report.json \
  --baseline baseline \
  --title "AgentForge SWE-bench Multilingual ablation" \
  --output-dir data/evaluations/benchmark-release/releases
```

Reports are rejected when dataset fingerprints, exact case IDs, totals, or labels differ.

## Statistics and selection

Pass@1 is shown with its per-run bootstrap interval. Each candidate is also compared against the declared baseline using:

- candidate-only and baseline-only wins on identical tasks;
- paired Pass@1 delta;
- a deterministic paired-bootstrap 95% interval;
- a two-sided exact McNemar p-value over discordant outcomes;
- total-cost and P95-latency deltas.

The displayed winner is ordered by higher Pass@1, then lower total cost, then lower P95 latency. Pareto status is computed jointly across those three objectives, so a cheaper or faster configuration is not erased merely because another configuration solves more tasks.

With small samples, wide intervals and large p-values are expected. Report them instead of treating a one-task difference as conclusive.

## Publication rules

The [patch-mutation matrix](patch-targeted-mutation-testing.md) fixes all settings
except the off/report/enforce policy. Release cards propagate sensitivity counts
and completed trials from retained selected-result review rounds. Stable assertion
failures alone count as kills; these totals are not semantic correctness or final
unique test strength. Invalid configured evidence withholds quality publication.

The [agent-cycle matrix](agent-cycle-control.md) changes only the off/shadow/enforce
policy. Release cards preserve `loop_guard_status_counts`, including returned
losing-branch receipts. Counts are not unique loops or evidence of cost savings;
stopping a stage may simply move inference into another repair stage. Keep the
existing resolution, regression, complete-usage, and price requirements.

The [file-level minimization comparison](execution-guided-patch-minimization.md)
retains optimization statuses, accepted file-removal counts, and completed trial
counts in release JSON/cards. These are diagnostics, not semantic-equivalence
proofs or net final savings. The frozen two-arm smoke matrix changes only that flag.

[Spectrum fault-localization comparisons](spectrum-fault-localization.md) expose
selected-context case counts and probe-status counts in release JSON and, when
receipts are present, the Markdown card. These are exposure diagnostics, not
accuracy scores. The two-arm frozen smoke matrix changes only that feature flag.

[Provider-held, interrupted or retried runs](provider-recovery.md) are operational
evidence, not complete quality comparisons. The release builder rejects them; matrix
execution stops after a provider hold rather than requesting the remaining variants.

- Generated cards label their score source as the AgentForge local harness and explicitly state that they are not official SWE-bench leaderboard scores.
- Submit `predictions.jsonl` to the official evaluator before making an official claim.
- Publish the official evaluator artifact alongside the local release rather than editing `release.json`.
- Freeze configuration choices on validation tasks, then publish the untouched test selection.
- Report every infrastructure failure, price assumption, image digest, date, model ID, sample size, confidence interval, and fingerprint.
- Do not compare matrices that use different task fingerprints or case sets.

This separation makes the artifact useful even before an official run: the code demonstrates benchmark design, experiment governance, statistical reasoning, and honest model reporting without manufacturing a performance claim.
