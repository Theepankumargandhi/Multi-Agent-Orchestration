# Structured regression discovery: local verification record

Work started October 16; final local checks completed October 17, 2026. This
record covers the structured-input extension to AgentForge Verify and its
shared sandbox changes. It is not a new full-repository audit or a claim that
all possible bugs have been eliminated.

## What changed

The portable verifier now generates bounded nested JSON inputs within an
owner-declared preservation domain. When the patch behaves differently from the
original, it confirms one failing input alone, tries smaller domain-valid
inputs, and independently confirms the selected smaller input again. A private
replay specification pins the source snapshot, exact patch and existing Docker
image ID.

Optional AI domain suggestions are separate from the execution oracle. They
require explicit caller authorization, owner review and unchanged patch/source
bindings before use. Existing contracts remain intact. There are no automatic
provider calls or self-approved preservation claims.

The shared Docker command runner also changed from unbounded output capture to
concurrent bounded-tail draining. Two additional negative controls caught and
fixed an underestimated schema-size bound for escaped Unicode and negative
numbers. This is a conservative admission bound, not an estimate of an average
generated input.

## Actual Docker results

All six authored control expectations passed. Every arm completed four initial
trials with the same two owner tests passing. Expected inconclusives were checked
for their specific reason; unavailable Docker or missing trials cannot satisfy
an expected-inconclusive control.

| Authored control | Strategy | Observed result |
|---|---|---|
| Zero-quantity regression | Existing tests only | Tests green; behavior inconclusive |
| Same regression | One explicit ordinary cart | Declared checks passed; bug missed |
| Same regression | Generated JSON carts | Preservation violation found and reduced |
| Equivalent arithmetic patch | Generated JSON carts | Declared checks passed |
| Intentional change outside the cart domain | Generated JSON carts | Declared checks passed |
| Domain admits an input unsupported by the original | Generated JSON carts | Inconclusive: invalid original scope |

The generated regression's selected input shrank from **91 to 36 canonical
UTF-8 bytes**. The smaller input was confirmed alone in four fresh containers.
It is a bounded greedy result, not a globally minimal counterexample. The
campaign used the existing local image
`sha256:4bc89e72e4cb90f86c2cb0991210e28f66f268632dd30ad37a29dcea2d3da45a`.

The completed six-control artifact fingerprint is
`108c0d4a666fe804451b65ba056230ea63710ed685b7ae98ddf870731682edf6`.
Its verifier/controller fingerprint is
`1ac88689188401ae18bc9d54669e71b02f4420cd6817866b1b6a2f437e66fe86`.
Private receipts were reassessed for complete trials, reference behavior,
isolated confirmation, matching implementation hashes and report integrity.

The independently confirmed input was also replayed through the normal CLI:
one case, four fresh trials, all owner tests green, `regression_found`, and the
original source unchanged. No submitted module was imported on the host. The
separate structured integration test also exercised replay end to end.

Earlier attempts are not counted as clean runs. One exposed an incorrect hunk
line in the authored intentional-change fixture. Another was interrupted and
left incomplete execution evidence. The fixture was corrected, incomplete
evidence stayed inconclusive, and the complete controls were rerun. The scoring
tests also explicitly reject missing Docker trials and unrelated inconclusive
reasons.

## Regression and packaging checks

The final focused unit/regression selection passed **196 tests**, with three
integration cases deliberately excluded from that command and exercised
separately. Coverage included schema admission, deterministic generation,
reduction confirmation, review tampering/staleness, model failure/timeout,
replay binding/export, private artifacts, output draining, patch admission,
benchmark/pilot lifecycle, coding-agent execution, verified PRs, provider
reliability and container import closure.

Eight patch/replay/migration profile tests passed without skips. These included
the structured reduction/replay, existing integer regressions, skipped-test and
import-error rejection, scripted migration repair, provider authorization and
Windows newline compatibility. Two additional real-container agent checks
passed: non-root/offline/read-only execution and scripted end-to-end repair with
the original repository unchanged. The remaining scripted Docker checkpoint/
resume test also passed on a fresh retry in 15.06 seconds, without changing its
timeout or assertions. Its earlier 49-minute attempt exceeded the task deadline
and failed; that receipt was retained, not counted as a pass. This record does
not establish the cause of that scheduling/runtime interruption.

Together with the strict prospective-source retrieval check below, the latest
scoped results reconcile to **208 passing test cases across separate commands**:
196 focused regressions, eight patch/migration profile checks, two agent checks,
one checkpoint/resume check and one retrieval-artifact check. This is not one
pristine full-suite run. Live-model behavior was not tested.

The service image built successfully as
`agentforge-service:structured-regression-20261017`. An offline, read-only-root
container running as UID **10001** imported the verifier, optional domain
proposal helper and bounded-output runner successfully. This was a packaging
smoke check, not a public deployment or load test.

Repository-wide Ruff and patch-whitespace checks passed. The public-source scan
parsed Python sources and checked relative documentation links. Exact matching
of the four locally configured secret values found no matches in intended
versionable files. No private runtime or environment path was versionable.
This check does not prove the absence of every possible credential.

## Source-bound retrieval artifacts

The new Python source changes the deterministic retrieval training corpus, so
fusion weights and their matching five-query regression report were refreshed.
Exact reproduction and the unchanged strict regression test passed against a
**disposable prospective-source index**. The real Git index was verified
unchanged; nothing was staged or committed. The ordinary tracked-source check
must still run after the complete intended source is staged in the eventual PR.

The refreshed artifact has 44 training pairs and 86.36% pairwise training
accuracy. Baseline/learned Recall@8 and MRR remain 1.00; NDCG@8 remains
0.9594/0.9348. Promotion was **rejected** for the 0.0246 NDCG regression. The
fixed-weight default was not changed. Reproducibility is not a reason to promote
a worse model.

## Evidence and limits

Raw inputs, receipts, JUnit results and replay specs remain in ignored local
directories, primarily `data/patch-verify/structured-controls-20261017-final/`
and `data/evaluations/structured-patch-20261016/`. Review before sharing; inputs
can be sensitive even when application outputs are omitted.

There were **zero live provider calls**, no paid usage, no upstream repository
writes and no Git push. AI proposal tests used injected controlled clients;
Docker execution itself was real. These are authored examples, not independently
curated real upstream bugs, production detection rates or maintainer acceptance
claims. The original program is only a reference inside owner-approved
preservation scope, not universal ground truth. Incorrect owner scope, bounded
sampling, shared-kernel isolation and unauthenticated local attestations remain
explicit limitations. The earlier dependency-audit deployment constraints are
not cleared by this upgrade.

See [the usage guide](structured-regression-discovery.md) for commands, supported
schemas and execution boundaries.
