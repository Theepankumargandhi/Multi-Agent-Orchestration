# Document SDK integration audit — October 16, 2026

This iteration packages the existing **text-evidence** pipeline for developers.
The user selected this scope instead of building a multimodal foundation. No
vision/OCR, chart understanding or generated-summary capability is claimed.

The [SDK guide](document-sdk.md) contains installation, Python/async usage,
existing-store adapters, privacy boundaries and runnable examples. Its default
answers are literal excerpts with checkable provenance, not model completions.

## What the implementation checks cover

- Shared production chunking, BM25, rank fusion and heuristic reranking rather
  than a second set of ranking formulas.
- Typed immutable pages, evidence, citations and result receipts, with bounded
  corpus/context sizes and document/version/page identities.
- Exact quote membership, receipt integrity, abstention and rejection of forged
  IDs, nonliteral quotes, duplicate selections and tampered citations.
- Explicit external-backend consent, lazy per-instance embeddings, labelled
  degraded lexical fallback, no automatic retries and unknown custom-backend cost.
- Read-only existing SQLite, already-fetched Chroma records, explicit text files
  and optional embedded-PDF text, including an unsupported blank-page hold.
- A localhost FastAPI integration that returns typed citations and does not echo
  invalid private request inputs in validation errors.
- Independent SDK import without the service, provider stack, tracing or `.env`.
- Actual wheel build/install followed by a separate application using the installed
  SDK with network sockets and full-platform/provider imports blocked.

The packaging control also checks license and `py.typed` metadata and that the
wheel contains only SDK files and its own distribution metadata. Pydantic is the
only required dependency; PDF extraction and FastAPI integration are optional.
No package/model downloads or publishing are performed by that control.

## Evidence interpretation

| Local check | Observed result |
|---|---|
| Focused SDK and HTTP tests | 50 passed |
| Related RAG/incident/grounding/coding suites | 191 passed, one Windows directory-symlink capability skip |
| Authored technical-report workflow controls | 10/10 passed |
| Final wheel build/install outside checkout | Passed; installed module used with network and platform imports blocked |
| Actual localhost HTTP checks | Structured page citation, abstention and privacy-safe invalid-input response passed |
| Lint, compilation and workflow syntax | Passed locally; no GitHub job dispatched |
| Live model requests / package publication | 0 / not published |

The focused tests are included in the related-suite total; these are not 241
distinct tests. Separate wheel-install and real HTTP controls exercise the
developer path beyond in-process test doubles. The final SDK-only wheel is
16,272 bytes in this environment; that excludes Pydantic and optional dependencies
and is a packaging observation, not a memory/performance benchmark.

The ten workflow controls use authored technical reports and labelled selectors.
They validate sensitivity, abstention and developer integration, not real-world
QA accuracy or a model improvement. A passing quote check means text provenance,
not relevance, truth, freshness or authenticated source ownership. Page labels
come from declared metadata or the PDF loader's one-based page enumeration.

The complete shared service remains a different product boundary. The SDK does
not inherit every LangGraph safety, grounding, approval, memory or provider gate.
Its optional Python adapters are trusted application code, not sandboxed plugins.
Localhost demo HTTP is not authenticated production hosting. The standalone wheel
is a prerelease build from the full checkout, not a published package-index release
or independently verified source archive.

Local receipts are kept under ignored `data/evaluations/document-sdk-20261016/`.
No live provider evaluation or external user adoption is claimed. Results from
the SDK and related regression suites are recorded there, including any explicit
platform-dependent skips; they are not represented as a complete repository run.

The existing learned retrieval candidate must be refreshed when shared source
changes. Reproducibility and holdout quality remain separate checks, and no new
artifact is automatically activated. See [code intelligence](code-intelligence.md)
for the tracked-source/prospective-index distinction before staging or pushing.

The final shared-core source refresh passed exact reproduction against a temporary
prospective index, without changing the real Git index. It trained 44 pairs with
84.09% training accuracy. Five-query baseline/candidate recall and MRR remain 1.00,
but candidate NDCG 0.934830 is below baseline 0.959387, so promotion stays rejected
and serving remains on the fixed baseline. Incident-replay runtime fingerprints
now also bind `agentforge_sdk/_retrieval.py`; old runtime evidence is not silently
reinterpreted against new ranking code.

Private local evidence:

```text
data/evaluations/document-sdk-20261016/selected-release.xml
data/evaluations/document-sdk-20261016/controls-final.json
data/evaluations/document-sdk-20261016/package-final-b/report.json
data/evaluations/document-sdk-20261016/http.json
data/evaluations/document-sdk-20261016/retrieval-refresh-receipt.json
```

The temporary localhost server was stopped after the checks. The full project
editable install was refreshed so its new SDK namespace/demo command resolves;
the standalone package check used a separate private installation target. No
changes were staged, committed or pushed, and no private document was uploaded.
