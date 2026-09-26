# Coding-agent benchmark release engineering

## Why this exists

Single benchmark runs are easy to cherry-pick and hard to compare. AgentForge therefore treats benchmark publication as a frozen experiment release:

1. Select and lock exact case IDs before tuning.
2. Declare every model, workflow, retrieval strategy, and price assumption in a matrix.
3. Run every variant on the identical selection.
4. Compare outcomes at the paired case level.
5. Publish metrics, uncertainty, cost, latency, failures, and artifact hashes together.

The release layer consumes the existing sandbox evaluator. It does not change the definition of a resolved task and does not hide infrastructure failures from the denominator.

## Credential-free validation

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

- Generated cards label their score source as the AgentForge local harness and explicitly state that they are not official SWE-bench leaderboard scores.
- Submit `predictions.jsonl` to the official evaluator before making an official claim.
- Publish the official evaluator artifact alongside the local release rather than editing `release.json`.
- Freeze configuration choices on validation tasks, then publish the untouched test selection.
- Report every infrastructure failure, price assumption, image digest, date, model ID, sample size, confidence interval, and fingerprint.
- Do not compare matrices that use different task fingerprints or case sets.

This separation makes the artifact useful even before an official run: the code demonstrates benchmark design, experiment governance, statistical reasoning, and honest model reporting without manufacturing a performance claim.
