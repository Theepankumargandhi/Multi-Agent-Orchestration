# Patch regression corpus contribution guide

This directory is the public contribution contract, not a populated real-world
benchmark. The currently runnable dataset is
[`../patch_regression_controls.json`](../patch_regression_controls.json): four
clearly labelled authored implementation controls. Independently reviewed real
cases admitted so far: **0**. Target ten carefully reviewed cases before making
any real-corpus performance claim.

See [the benchmark guide](../../../docs/patch-regression-benchmark.md) for commands.
The [public upstream development pilot](../../../docs/real-patch-validation.md)
is populated separately. Its four pinned forward changes are source-verified
replays, not independently reviewed benchmark admissions. Review kits remain
pending; they do not increase the admitted count above zero.

Keep proposed snapshots, patches, owner specifications, withheld reproducers and
review kits under ignored `repositories/` or `data/regression-benchmark/` paths.
Only publish code and license evidence that you have permission to redistribute.

A real-case manifest entry needs these fields:

| Field | Requirement |
|---|---|
| `case_id`, `family`, `split` | Stable identity; no development/holdout family overlap |
| `evidence_kind` | `historical_real_patch`, never a relabelled authored fixture |
| `repository`, `patch`, `spec` | Bounded, confined local paths below the selected root |
| `expected` | Proposed `regression` or `correct`; independent review must confirm |
| `license_file`, `license_identifier` | Exact licensed snapshot and redistribution basis |
| `upstream_url`, `upstream_commit` | Origin repository and full pinned revision |
| `patch_origin` | `upstream_commit`, `historical_reversal` or `model_generated`; disclose which |
| `review` | Initially null for intake; required and artifact-bound at freeze |

Before approving a case, the reviewer should reproduce both original and patched
existing tests passing, confirm the owner intended preservation, inspect the
withheld reproducer and review equivalent/correct negative controls. A candidate
that fails existing tests is retained as ineligible, not quietly substituted to
improve the detection score. Report all exclusions and inconclusives.

The review kit intentionally leaves reviewer identity and evidence URL blank.
Submitter and reviewer must differ. Local attestations do not prove identity or
that a supplied upstream commit really contains the snapshot; independently
check the upstream content before publishing. A historical fix reversed to create
a regression is not evidence that a model made that mistake. Likewise, an agent
generating a patch is not evidence that a maintainer accepted it.

The generator supports owner-declared integer or bounded structured JSON domains
and explicit JSON-call contracts, not arbitrary tests or all application behaviors. Public
controls are not a secret holdout, and a ten-case sample would still be a small
descriptive study rather than an official benchmark leaderboard result.
