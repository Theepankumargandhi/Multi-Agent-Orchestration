# Real-patch validation: October 17, 2026

Structured discovery was run against four actual forward upstream changes from
three projects. All **12 case/strategy expectations matched**, and both detected
regressions were independently replayed from their reduced inputs. This is a
small, public, curator-labelled **development pilot**, not an independently
reviewed benchmark or a claim of production detection accuracy.

## Observed results

Every arm completed four initial fresh-container trials, with its selected
existing tests passing for both original and candidate. Source snapshots stayed
unchanged. Neither tests-only nor the ordinary explicit contracts raised an
alarm for the two documented regressions.

| Actual upstream production change | Existing selected tests per trial | Structured result | Independently replayed input |
|---|---:|---|---|
| [Boltons `af6ab9f`](https://github.com/mahmoud/boltons/commit/af6ab9fb9d5da8094f09e789ce81d65854274410) | 12 | Regression found | `"buses"` |
| [Humanize `fb61d5a`](https://github.com/python-humanize/humanize/commit/fb61d5aae902e4526cd733a90d7a96ecded11b0e) | 62 | Regression found | `0.5` |
| [Humanize `a178851`](https://github.com/python-humanize/humanize/commit/a178851b369e303242749b60badaef697480375a) | 62 | No violation in declared native-number scope | Not applicable |
| [more-itertools `bc8c7ac`](https://github.com/more-itertools/more-itertools/commit/bc8c7acad2bcf9e0dc93f45c79a19d61794da012) | 4 | No violation in declared JSON scope | Not applicable |

For Boltons, the reduced input's observed result changes from `"bus"` to
`"buse"`. For Humanize, `0.5` changes from `"0 Bytes"` to `"0.5 Bytes"`.
These descriptions were checked against the stored execution hashes, without
importing upstream modules on the host. Public follow-up fixes support the
proposed labels; the selected preservation domains deliberately exclude
intentionally changed behavior.

The Boltons input reduced from **11 to 7 canonical JSON UTF-8 bytes**; Humanize
reduced from **18 to 3 bytes**. String sizes include their JSON quotes. Reduction
used one bounded search round, then fresh single-input confirmation. Each final
replay used four additional fresh trials, separate from the search and its
confirmation. Both reproduced the regression with the existing tests still green.
These are bounded greedy reductions, not globally minimal inputs.

| Strategy | Documented regressions detected | Alarms on preservation controls | Inconclusives | Observed mean arm latency |
|---|---:|---:|---:|---:|
| Existing selected tests | 0/2 | 0/2 | 0/4 | 26.50 s |
| Tests plus ordinary explicit contracts | 0/2 | 0/2 | 0/4 | 26.23 s |
| Same checks plus structured generation/reduction | 2/2 | 0/2 | 0/4 | 62.80 s |

These timings include Docker startup, repeated tests/probes and reduction where
applicable; the separate final replays are not included in these arm means.
Provider calls and provider cost were **zero**. Infrastructure cost is unknown.
Four selected public cases cannot establish statistical superiority, a reliable
population false-alarm rate or repository-disjoint generalization.

## What was frozen and what was projected

The public catalog pins full base/candidate Git commit IDs and expected SHA-256
blob bytes. Preparation checked Git object identities and direct first-parent
relationships, copied selected base files and exact license bytes, and built
production-only forward diffs. This was not a hand-written bug mutation or a
reversed fix.

It was also **not the full upstream commit or complete test suite**. Boltons
used the unchanged base strutils test module, Humanize the unchanged filesize
test module, and more-itertools the unchanged `OnlyTests` class. Other commit
paths were explicitly excluded and recorded. Humanize's `src/` layout was mapped
to the import root, with a recorded build-only version constant for metadata
normally generated during packaging. No test assertion or target implementation
was rewritten to make the result pass.

Boltons ran in the separate offline-only Python 3.9 historical image because
its 2018 tests use APIs removed from maintained Python versions. That runtime
is end-of-life and must not be deployed as a service. The other cases used the
normal Python 3.12 sandbox. Frozen execution identities were:

- Historical image: `sha256:33aba66516d7ddc6e5acaecf725b967136dfa58b9127efcda6208f96f9b1bb96`.
- Normal sandbox: `sha256:4bc89e72e4cb90f86c2cb0991210e28f66f268632dd30ad37a29dcea2d3da45a`.

Git caches/snapshots stayed local and ignored. Preparing a new cache requires
explicit download authorization; no source downloads occurred during execution.
The cache download has a command deadline, not a filesystem quota: use a
quota-controlled environment before accepting arbitrary third-party catalogs.
Docker isolation is not a complete hostile-code boundary.

## Implementation and regression validation

The integration added source/license provenance intake, exact production patch
projection, immutable study validation, per-arm receipts, independent replay
confirmation and an input-free summary. It also fixed the benchmark's older
integer-only admission check so structured-only specifications can use the
existing review workflow. Bounded named pytest selectors permit upstream test
classes without copying or rewriting their assertions.

The final targeted regression command passed **213 tests** with one unrelated
provider checkpoint integration deliberately excluded. This covered Git object
and byte corruption, symlinks, wrong origins, review/admission laundering,
image/source/spec/patch drift, receipt binding, selector injection, structured
generation, reduction, replay, private evidence and existing benchmark/pilot
behavior. The selected legacy/structured portable Docker suite passed **5 tests
without skips**. The unchanged strict retrieval-artifact test also passed
against a disposable prospective-source index. These are separate commands,
not a new complete-project test run.

An initial preparation attempt correctly stopped before upstream execution
because an overbroad LF check rejected the original CRLF license. The intake
was corrected to preserve that exact license blob; production patch admission
still enforces its LF profile. A local Git regression test now checks this case.
The completed study was not assembled from that partial attempt.

Repository Ruff, patch-whitespace and relative-link checks passed. The intended
versionable source was scanned for exact matches of the four locally configured
secret values; no matches or private versionable artifact paths were found.
This is not a universal secret-detection guarantee. The new manual GitHub
workflow was added but not dispatched; regular CI's new intake tests are offline.
No project commits, PRs, pushes, maintainer contacts or upstream mutations were made.

## Review status and evidence

**Independently reviewed benchmark admissions remain zero.** All four review
kits intentionally leave the independent reviewer and review evidence URL blank.
The submitter is labelled automated public curation. Source verification and
public bug-fix evidence do not substitute for a separate review of these exact
snapshots, domains, projections, labels and runtime assumptions.

The existing benchmark still rejects these unreviewed cases. A human reviewer
can inspect the private kits and reproduce them before issuing artifact-bound
reviews. A ten-case reviewed corpus and a fresh withheld evaluation remain
separate milestones. No model-generated test quality, maintainer acceptance or
official SWE-bench result is claimed.

Local evidence is under ignored `data/regression-benchmark/upstream-20261017/`:

- `prepared-v2/study.json`: frozen catalog and runtime/source bindings.
- `prepared-v2/<case>/review-kit.json`: pending independent review.
- `results/results.json`: all 12 observations and two final replay receipts.
- `results/<case>-reproducer.spec.json`: source/patch/image-pinned input replay.
- `results/public-summary.json`: summary without raw inputs, code or counterexamples.

Study fingerprint:
`a982b7583f5b9665fd5ccd0637880ca09b770a4988975dc5570befabddd7ad0a`.
Results fingerprint:
`bb9c11eb399c9121959f3d3a7f9bcab9cbabe53df2a9b7eb666d7ba84658515f`.
Verifier/controller fingerprint:
`4d2d0c57b7afaace8b6597e8e63edcb4934c80e1a28136bfc31fa16cc651bb87`.

The new source also required refreshing the deterministic retrieval artifact.
Exact reproduction passed without changing the real Git index. The refreshed
candidate still regresses NDCG by 0.0246 on the five-query regression set and
remains **unpromoted**; the fixed-weight default was not changed. The ordinary
tracked-source gate should run after the complete intended change set is staged.
Earlier deployment/security waivers are not cleared by this validation.

See [the reproduction guide](real-patch-validation.md) for commands and scope.
