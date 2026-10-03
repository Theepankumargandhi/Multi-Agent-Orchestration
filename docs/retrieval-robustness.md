# Retrieval robustness under query stress

A clean query can hide a brittle retriever. This study checks whether required source lines
survive whitespace changes, polite wording, explicitly irrelevant terms, and quoted
instruction-like noise. Only the query changes; retrieval never sees the scoring labels.

Each variant is paired with its clean query at fixed approximate budgets. Reports include
family-macro coverage, span drops, paired family-bootstrap 95% intervals, and regressed case
IDs. Worst-case coverage takes the minimum per case before family averaging. Variants do
not count as additional independent families. The fixed plan specifies a primary budget,
allowed drop, and worst-case recall floor; an empty-context control verifies failure detection.

```bash
python -m evals.retrieval_robustness \
  --plan evals/experiments/retrieval_robustness_plan.json \
  --output data/evaluations/retrieval-robustness/my-study-v1.json
```

JSON and Markdown outputs use fresh exclusive filenames and cannot overwrite inputs or
indexed source. Raw queries and snippets are omitted, but paths and IDs can be private
metadata. Fingerprints detect changes; they do not authenticate reviewers. CI uploads the
diagnostic report without enabling models. `--require-gate` returns 1 because generated
variants always remain held; invalid inputs return 2. A normal valid run returns 0 even
when numerical gates fail: inspect `numerical_gate_passed` and reasons, not just exit status.

On the authored 12-query/six-family fixture, both packers retain their clean span recall
under all four transformations: legacy 83.3%, balanced 100%, at 256/512/1024 tokens.
This does not establish resilience on unseen repositories. Generated variants reset reviewer
provenance to synthetic; review of an original question does not approve changed intent.

Instruction-like noise tests retrieval sensitivity, not LLM prompt-injection resistance.
Distractors may change intent and need review. No model answers, patch execution, traffic
lift, or provider tokenizer measurements are included. Independent repositories and
downstream outcomes are needed before promotion.

See [context packing](query-aware-context-packing.md) and
[source-bound evaluation](evidence-span-retrieval.md) for underlying contracts.
