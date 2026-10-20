# Release-readiness review — October 16, 2026

AgentForge is ready for a technical code review as an engineering reference
project. This pass fixed concrete deployment, retrieval and reliability problems;
it did not establish that every line is optimal or clear the system for
unrestricted production traffic.

## What was reviewed

The initial inventory contained 592 shareable files and 363 Python files. The
final inventory contains 599 shareable files and 369 Python files, including the
new regression tests and setup utility. All versionable Python files were parsed
for syntax and inventoried structurally. Repository-wide Ruff, relative Markdown
links, whitespace checks and exact configured-secret comparisons were checked.
Ignored runtime data and virtual environments were not treated as source code.

Manual review concentrated on retrieval, the SDK's contracts and adapters,
coding-agent execution boundaries, credential screening, artifact bindings,
container composition, CI and onboarding. This was not a line-by-line manual
attestation of roughly 82,000 Python lines. Formatting changes were scoped to
the small SDK instead of rewriting unrelated orchestration code.

## What changed, and why

The service image omitted `agentforge_sdk`, even though production RAG now
imports its shared retrieval primitives. Several selectively copied dashboards
also omitted internal dependencies. Their Dockerfiles now copy those modules
and check application imports during the build. A regression test checks the
internal COPY/import closure of all seven application/dashboard images. The
code-context dashboard is also included in CI's image-build checks.

The immutable SDK corpus was being tokenized and counted again for every
question. It now owns a reusable BM25 postings index and evidence-ID lookup.
The one-shot service ranker still shares the same scoring implementation.
Independent reference tests check exact scores, repeated query terms, ties,
empty documents and the resulting evidence receipts. No retrieval-quality
improvement is inferred from a speed improvement.

Other changes close small but meaningful failure paths:

- Backend identities are pinned per SDK instance, preventing a new query model
  from being mixed with old cached document vectors or an obsolete selector ID.
- Invalid Unicode questions are rejected before any custom backend invocation.
  File reads themselves are bounded even if an input grows after its size check;
  oversized embedding matrices are rejected before numeric conversion.
- Native parser grammars are explicitly provisioned during setup and the
  service-image build. Runtime indexing never initiates lazy grammar downloads.
  Native setup failures use cached, content-free fallback diagnostics.
- Optional Redis clients have finite connection/read deadlines while retaining
  the existing cache-failure fallback.
- Admission and post-patch verification share credential-pattern definitions,
  including Groq and underscore-form GitHub literals. The policy receipt and
  verifier implementation fingerprint bind those definitions. Pattern matching
  remains a heuristic, not a general secret detector.
- Docker contexts exclude environment variants, private keys and runtime logs.
  CI lint includes the SDK and examples, and coverage includes the SDK package.
- SDK packaging uses short disposable build/install paths, avoiding a real
  Windows path-length failure under this deeply nested checkout.

## Measured optimization

A final local microbenchmark used 160 authored document pages, 640 chunks and
12 queries over five rounds. Both paths produced identical search receipts.

| Warm search path | Median time per query |
|---|---:|
| Original repeated corpus preprocessing | 34.38 ms |
| Reusable corpus index | 0.34 ms |

That is about a 103× warm-query speedup on this synthetic workload. It is not a
claim about LLM latency, real-document answer quality, or every corpus. Indexing
moves work upfront: this measurement retained about 3.27 MB for the index and
peaked at 6.36 MB while constructing it. Construction took about 416 ms with
allocation tracing enabled; that tracing cost is not a production timing result.

## Verification and the failures found

The complete initial local run finished with **1,532 passed, three failed and
one skipped** across 1,536 tests. The failures were investigated rather than
excluded from the final accounting:

1. Fault localization could not load coverage instrumentation from an old local
   sandbox image. Rebuilding from the declared pinned requirements fixed it.
2. A memory-revocation test double lacked the workspace's newer edit-admission
   method. The double was updated; production authorization was not relaxed.
3. The exact retrieval-artifact check enumerated tracked files, while intended
   new source files were still untracked. The candidate was regenerated and the
   unchanged strict test passed against a temporary prospective index. The real
   index was not changed.

Final affected-path runs passed **215 combined regression tests**, **142 sandbox
and coding checks**, **31 cache/research checks**, and **23 parser/retrieval
checks with one Windows directory-symlink skip**. These suites overlap; their
counts must not be added together.

Reconciliation against the final collection of **1,578 tests** found a passing
latest result for **1,577**, with that one explicit platform skip and no missing
test IDs. This combines the initial full run with final scoped reruns. It is
**not** a claim that a second pristine full run passed in one invocation, or that
GitHub CI has run on a committed revision.

Additional checks passed:

- Repository-wide Ruff, syntax inventory, whitespace and relative documentation
  links; no exact matches for the four configured secret values in public files.
- All ten authored SDK workflow controls and a fresh 17,241-byte SDK wheel
  installed outside the checkout, with network/provider/platform imports blocked.
  This reuses the interpreter's Pydantic dependency, not an empty machine.
- Actual builds of the four changed runtime images: service, coding-evaluation,
  code-context and security dashboards. The coding sandbox was rebuilt too.
- Final service imports and native parsing with networking disabled, running as
  UID 10001; `pip check` reported no broken requirements inside that image.
- Exact retrieval-candidate reproduction. Its five-query holdout still rejects
  promotion because learned NDCG regresses; the fixed baseline remains serving.

No live model benchmark, provider spending, package publication, commit or push
was performed. Private XML, JSON, wheel and timing evidence is retained under
the ignored `data/evaluations/release-readiness-20261016/` directory.

## What still needs care before release

The [Chroma dependency waiver](security/audit-exceptions.md) is overdue for owner
review. It was not silently extended. The checked advisories still report
unpatched issues; embedded-only use is a scoped mitigation, not blanket security
clearance. This pass did not perform a complete fresh vulnerability audit of
every transitive dependency.

The host Anaconda environment still has missing/mismatched service dependencies,
and the existing `.venv` points at a missing interpreter. Use a fresh project
environment following [CONTRIBUTING](../CONTRIBUTING.md), or the verified service
image. Do not interpret the clean container dependency check as repairing the
host environment.

The real Git index deliberately still excludes some new files. Stage the
reviewed public source together before rerunning the ordinary strict artifact
check and CI; testing only the older tracked subset will correctly report drift.
Do not stage environment files, runtime stores or private evaluation artifacts.

Large orchestration modules remain candidates for gradual decomposition with
behavior-preserving tests. Public untrusted execution also needs a stronger
worker boundary and host-side streaming output caps: the current Docker command
runner truncates output after subprocess capture, not before host allocation.
These are meaningful production-hardening tasks, not reasons to claim that a
cosmetic rewrite makes the code universally optimized.

Finally, a current-file secret scan does not audit all Git history or revoke
credentials previously disclosed elsewhere. Rotate any exposed credentials and
review the actual staged diff before sharing.
