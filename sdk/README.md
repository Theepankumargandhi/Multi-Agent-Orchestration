# AgentForge document SDK

A small Python interface to the existing text-RAG chunking, BM25 ranking, rank
fusion and heuristic reranking. Default answers are literal evidence excerpts
with document/page/version references and checkable quote offsets. They are not
model-generated summaries or a correctness guarantee.

From the repository root:

```bash
python -m pip install ./sdk
python examples/document_sdk_quickstart.py
```

The SDK-only wheel needs Pydantic, not the agent service, LangGraph, Docker,
Chroma or provider SDKs. Optional embedded-PDF text extraction is available with
`python -m pip install './sdk[pdf]'`. This release packages the shared source
from the parent repository; build/install from this checkout or use the built
wheel. Do not copy this `sdk/` directory on its own or publish its source archive.
No package or model is uploaded automatically. Version 0.2 adds an optional
spatial path with `python -m pip install './sdk[multimodal]'`: native PDF tables,
explicit local Tesseract OCR, page/region/cell citations and replayable Decimal
calculations. Tesseract itself and language data are caller-installed.

Version 0.3 adds page corrections and manually reviewed scan table grids through
`ReviewedRAG`, retaining original machine extraction and per-citation review
origin. HMAC-bound approvals expire and can be revoked in-process. See
`docs/reviewed-document-evidence.md`; the application provisions reviewer keys
and authenticates the review action. The SDK does not manufacture human approval.

Version 0.4 adds `DocumentRevision`: extraction-level version comparison, verified
answer/calculation dependency impact and explicit calculation replay. New files
and review approvals require new receipts; uncertain alignment never silently
carries evidence forward. See `docs/document-version-impact.md` for scope and
controls. This is not general semantic or pixel-level PDF diffing.

Version 0.5 adds `RevalidationBatch` for saved question/answer and calculation
receipts: assessment by default, opt-in fresh retrieval and tool replay, a
batch-wide answer-attempt budget, JSON checkpoints and a content-minimal review
queue. Resume rechecks evidence and current review authority, without rerunning
completed answer backends. Storage and worker coordination remain the
application's responsibility; there is no exactly-once or spend-cap claim. See
`docs/document-revalidation.md`.

Version 0.6 adds `EvidenceSet`: exact-label numerical fact reconciliation across
explicitly scoped documents, original label/value citations, and calculations
that refuse unresolved facts. Reports never settle disagreements by majority,
guess a reporting period or convert currencies. Reviewed citations retain their
approval lineage. See `docs/evidence-reconciliation.md` for the deliberately
limited numeric grammar and source-trust boundaries.

Version 0.7 adds `DocumentReasoningAgent`: an explicit planner can select literal
quotes, look up registered facts, or request bounded Decimal calculations.
Unknown candidates and invalid plans cannot execute; conflicts and provider
failures abstain. Only invalid plans or missing candidates may trigger bounded
retrieval recovery. Local retrieval stays local, external planning requires
consent, and unknown provider usage/cost is not reported as zero. See
`docs/document-reasoning-agent.md` for the opt-in Groq adapter and the frozen
real-document pilot. Provenance verification does not prove that a quote answers
the question; human answer-quality review remains separate.

Version 0.8 adds `NumericClaim`, `check_numeric_claim` and `verify_numeric_claim`
for exact Decimal assertions over verified `EvidenceSet` facts/calculations.
Matching values and units are supported, different comparable numbers refuted,
and unusable evidence or unit mismatch unresolved. Display labels are not parsed;
there is no rounding tolerance, scope inference or release approval. The web
service adds optional resumable batches and private reports separately. See
`docs/numeric-claim-audits.md` for SDK usage and the boundaries.

Version 0.9 adds `EvidenceSet.source_ids` and `without_sources([...])` for separate
source-omission diagnostics. Minimum artifact count, full-coverage policy and
retained review authority stay unchanged. Original evidence is not modified;
support appearing after omitting dissent does not resolve the original conflict.
The service's bounded matrix and resumable checkpoints are separate from the SDK.
See `docs/evidence-fragility-stress-tests.md`.

Version 0.10 adds portable numeric replay bundles with `ClaimReplayRecipe`,
`build_claim_replay_bundle`, `load_claim_replay_bundle` and
`verify_claim_replay_bundle`. `python -m agentforge_sdk.bundle_cli bundle.json`
recomputes every saved check using the base SDK alone. It reads no original
documents and calls no parser or provider. Bundles contain full private extraction
text, including uncited regions; live reviewed authority is refused, not stripped.
Exit 0 means replay consistency, not universal support or source authenticity.
See `docs/portable-claim-replay.md` for separate pinning and `--require-supported`.

```python
from agentforge_sdk import DocumentPage, DocumentRAG

rag = DocumentRAG([
    DocumentPage(document_id="atlas-report", source="atlas-report.pdf", page=2,
                 version="r1", text="Atlas p95 latency is 180 ms at 100 requests per second.")
])
answer = rag.answer("What is Atlas p95 latency?", top_k=1)
assert answer.status == "evidence_found"
assert all(rag.verify_citation(item) for item in answer.citations)
```

The full guide at `docs/document-sdk.md` covers existing SQLite/Chroma records,
PDF limitations, explicit custom backends, async usage, privacy and reproducible
controls. The separate `docs/multimodal-evidence-sdk.md` guide covers spatial
evidence and its strict limits: no chart understanding, scanned table inference
or generated summaries. No OCR, numeric tool, unrestricted agent tool or network
processing runs by default. There is no telemetry, `.env` loading, sample
document seeding or automatic persistence.
