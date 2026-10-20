# Portable verification and pilot audit — October 16, 2026

This was a local implementation check, not an official benchmark submission,
live-model coding study or external maintainer trial. The implementation adds
three usable workflows: [external patch verification](portable-patch-verification.md),
[frozen regression comparisons](patch-regression-benchmark.md), and
[consented migration pilots](maintainer-migration-pilots.md).

## What ran

| Check | Observed result |
|---|---|
| New and related regression suites | 293 passed; zero failures, errors or skips |
| Authored patch case/strategy expectations | 12/12 passed with actual Docker |
| Authored migration pilot lifecycle expectations | 6/6 passed with actual Docker migration verification |
| External prediction import through the normal CLI | Passing equivalent-refactor fixture; nine behavioral cases; no counterexamples |
| Installed CLI entry points | All three installed and their help commands ran |
| Lint and Python compilation | Passed locally |
| Live provider requests | 0 in the new execution controls |
| Independently reviewed real benchmark cases | 0; external intake/review remains pending |
| Real maintainer contacts/acceptances | 0; lifecycle consent/review were explicitly synthetic |

The selected tests cover new admission, exact hunk application, generated domains,
receipt binding, repeated Docker execution, skips/import errors, family leakage,
review provenance, consent withdrawal and tampering. Related coverage includes
the existing migration/recovery/checkpoint, delivery, counterfactual, verified-PR,
tournament and adaptive-supervisor workflows. This is not a full-repository test
run or a claim that every specialist was exercised with a live model.

## The actual comparison

The four authored patches form two families: a zero-quantity fee boundary and an
integer-versus-float serialization change. Each family has a deliberately
regressing candidate and a behavior-preserving refactor. Existing owner tests
pass for both: the discount example demonstrates why Python equality alone can
overlook a JSON-visible type change.

| Strategy | Regressions detected | Correct-patch false alarms | Inconclusive comparisons |
|---|---|---|---|
| Existing tests | 0/2 | 0/2 | 0/4 in the tests-only alarm baseline |
| Explicit owner contracts | 1/2 | 0/2 | 0/4 |
| Contracts plus generated owner-domain inputs | 2/2 | 0/2 | 0/4 |

The tests-only *verifier* still reports behavioral evidence as inconclusive; the
benchmark intentionally interprets ordinary tests staying green as no alarm,
not as a semantic verification pass. All arms have their own repeated fresh
Docker execution. The sample is four engineered controls, so these figures
measure implementation sensitivity, not real-world detection accuracy or AI
superiority. Infrastructure cost was not measured; zero provider cost is not a
claim that execution was free.

The prediction-import check used `examples/patch-verification.preds.json`, whose
producer label explicitly says it is authored, not a model. It validates batch
format interoperability, not a live mini-SWE-agent run or authenticated producer.

## Pilot lifecycle evidence

The pilot used the existing authored Pydantic field-constraint snapshot, the
actual pinned old/target/codemod environments and generated counterexample
verification. Its migration was review-ready without inference. The lifecycle
then rejected an implicit second execution, recorded a synthetic accepted review
without counting it as real acceptance, blocked an unconsented public summary and
blocked use after consent withdrawal. Nothing contacted a maintainer, created a
PR, pushed, merged or deployed.

The wrapper does not fix the separate live-provider limitations documented in
the [October 4 audit](live-upgrade-audit-2026-10-04.md). Real-world usefulness still
needs licensed external tasks, genuine consent, independent reviewers and
maintainer decisions.

## Retrieval artifact maintenance

The new source made the existing fusion artifact stale. A temporary prospective
index included intended public source names without staging them; exact
reproduction and the five-query holdout comparison then completed. The real Git
index stayed unchanged. The refreshed artifact contains 44 training pairs with
88.64% pairwise training accuracy, not a downstream coding-quality metric.

Baseline and candidate both retain recall/MRR 1.00. Candidate NDCG is 0.929456
versus baseline 0.959387, so the unchanged promotion gate rejects the candidate.
Serving remains on the fixed baseline. See [code intelligence](code-intelligence.md)
for the distinction between artifact reproducibility and model activation. The
ordinary Git-root exact check must be rerun after the intended new source files
are tracked; it otherwise sees a narrower corpus than the prospective refresh.

## Local evidence and reproduction

Private receipts are under these ignored paths; they are not public attachments:

```text
data/evaluations/portable-verification-20261016/selected-final.xml
data/regression-benchmark/controls-20261016-final-b/controls.json
data/maintainer-pilots/controls-20261016-final-b/controls.json
data/patch-verify/prediction-cli-20261016-final-b/report.json
data/evaluations/portable-verification-20261016/retrieval-refresh-receipt.json
data/evaluations/portable-verification-20261016/final-evidence.json
```

Reproduce with the commands in the three workflow guides, using new private
output directories. The checked-in CI job runs authored controls without provider
credentials; no GitHub CI run was dispatched here. No changes were committed or
pushed. Existing unrelated worktree edits were preserved.

On this Windows host, restricted-process asyncio initially stalled while creating
its internal loopback socket pair. The test process was interrupted and rerun
with local loopback/Docker access; the final result above is the completed run,
not an inference from a partial log. The installed Anaconda Scripts directory
is not on PATH, so module commands or full executable paths work without changing
the user's shell configuration.
