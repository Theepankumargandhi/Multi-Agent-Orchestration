# Query-aware context packing

Finding the right file is not enough if compression drops the guard or return that answers
the question. The opt-in `balanced_v1` packer addresses that gap without changing retrieval
ranking or calling another model. The default remains `legacy`.

Set `CODE_CONTEXT_PACKING_POLICY=balanced_v1` to try it deliberately, or pass
`packing_policy="balanced_v1"` to context selection. Unknown policies fail closed.

## What changes

The packer combines query matches with declaration, guard, and exit windows. Higher-ranked
files receive larger shares of the available context. It considers at most 64 windows and
emits only complete numbered source lines that fit the approximate character-based token
budget. An oversized line is omitted, never truncated into a misleading prefix.

Overlapping windows within a file share their line positions. Identical text in different
files is retained so one file cannot impersonate another file's evidence. Receipts include
the packing policy, contiguous source-line ranges, retained/omitted line counts, and a
fingerprint. The span evaluator checks those fields against the actual emitted source.

These are excerpts, not executable programs: gaps remain explicit, trailing whitespace is
normalized as in the existing renderer, and syntax completeness is not guaranteed. The
token budget remains an estimate, not a provider tokenizer measurement. No evaluation
labels enter the packer and no model is downloaded.

## Review the isolated comparison

```bash
python -m evals.retrieval_span_evaluation \
  --plan evals/experiments/context_packing_plan.json \
  --output data/evaluations/context-packing/my-study-v1.json
```

Both arms use `hybrid_rerank` with the same top-k and budgets; only packing differs. On the
authored 12-query/six-family fixture, complete-span recall rises from 83.3% to 100% at
256, 512, and 1024 approximate tokens. File recall stays 100%. The long rendering function's
final assembly lines now survive compression.

The family-bootstrap 95% interval for the paired span delta is `[0.0, 0.5]`: this small toy
study does not establish a positive generalizable gain. Its synthetic labels keep the study
held. CI uploads a review card but never activates the packer or promotes learned weights.
Use independently reviewed, unseen repositories and downstream patch tests before making
claims about real coding-agent quality.

See [evidence-span evaluation](evidence-span-retrieval.md) for label integrity, review gates,
and output protection. Reports use fresh filenames and are never silently overwritten.

The [query robustness study](retrieval-robustness.md) adds paired wording/noise controls,
worst-case source coverage, and clean-to-stress regression gates without changing defaults.
