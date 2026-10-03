# Private reviewed holdouts

Use this directory for authorized, redacted evaluation cases that must not be committed or used to tune the system. Its [.gitignore](.gitignore) ignores everything except itself and this README. This is a local privacy convention, not encryption, access control, or an automatic train/test boundary.

Keep provider keys, personal data, raw user conversations, and proprietary source text out of datasets. Use access-controlled storage for genuinely sensitive review material. CSV review sheets, exported reports, and traces can also contain inputs, labels, or reviewer metadata; keep them private even if the original JSONL is ignored. Do not use `git add -f` to bypass these rules.

## Case format and label review

For the general experiment runner, each JSONL line must match `EvalCase` in [evals/platform.py](../../platform.py): a unique `id`, `input`, `expected` behavior, optional `tags`, a `train` / `validation` / `test` split, and explicit review provenance. Other evaluation families have different schemas; do not assume this format works for coding benchmarks or signed runtime cohorts.

Unreviewed cases default to `synthetic_seed`. Use `human_reviewed` only after an actual review. From the repository root, export and import a label-review sheet with:

```bash
python -m evals.review_dataset export --dataset evals/datasets/private/holdout.jsonl --output evals/datasets/private/review.csv
python -m evals.review_dataset import --dataset evals/datasets/private/holdout.jsonl --review-file evals/datasets/private/review.csv --output evals/datasets/private/reviewed-holdout.jsonl
```

Use new output names and preserve the original. Approved rows require a reviewer and receive review audit metadata. Import does not remove rejected or unreviewed cases: their original records remain in the output. Inspect and deliberately select the intended reviewed cohort before reporting results; neither this folder nor a filename proves that every case is human-reviewed. Plain JSONL/CSV review metadata is not the signed independent-review protocol used by the runtime learning studies.

An experiment config must explicitly select its dataset and splits. The general runner does not discover this folder automatically or prevent repeated holdout tuning. Establish representative families and a frozen split before adjusting prompts, retrieval weights, thresholds, or learned policies. Keep near-duplicates and related tasks in the same family, tune on training/validation only, and publish test results with dataset/configuration fingerprints and honest provenance. See [evaluation strategy](../../../docs/evaluation.md).

## Runtime studies use a different protocol

Prospective calibration, reviewed preferences, independent workflow supervision, verifier shadows, and step-ensemble outcome studies use consented signed metadata in private runtime stores. They do not load arbitrary JSONL labels from this directory. In particular:

- task-family IDs are assigned before execution, not reconstructed after seeing outcomes;
- explicit step correctness reviews are separate from final-answer correctness/safety reviews; missing or ambiguous step labels are not inferred from terminal credit;
- cohorts bind chronological cutoffs, embargoes, complete eligible pools, review availability, and live source lineage;
- the shared `data/execution-replay/holdout-ledger.sqlite3` reserves fresh test families before scoring across studies;
- changing a candidate or policy requires fresh families; resetting, copying, or forking the ledger to reuse exposed evidence defeats the protocol.

Exact completed retries revalidate source lineage and reuse signed cached evidence. Synthetic drills exercise these rules but do not provide human-reviewed quality evidence or authorize deployment. A passing verifier/step-ensemble outcome study is for owner review only and does not activate a serving model.

Follow the [prospective validation](../../../docs/prospective-ai-validation.md), [independent process supervision](../../../docs/reviewed-process-supervision.md), [reviewed step ensemble](../../../docs/reviewed-verifier-uncertainty.md), and [final-outcome validation](../../../docs/ensemble-outcome-validation.md) guides for the correct schemas, keys, commands, and holdout rules.

## Retention

Protect private files, runtime databases, signing keys, backups, and holdout exposure history together. Tenant deletion removes the relevant private runtime rows, not external JSONL/CSV exports, saved reports, or the separate opaque exposure ledger. Handle exported files and ledger retention through an explicit policy; do not describe deletion as model unlearning or silently erase exposure history to retune on the same tasks.
