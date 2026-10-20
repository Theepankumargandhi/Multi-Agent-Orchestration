# Reviewed document evidence: execution audit

This audit covers SDK 0.3's correction/review path. It does not claim a fresh
complete-platform pass, genuine human approval or model-learned scanned-table
extraction. The [original multimodal audit](multimodal-sdk-audit-2026-10-17.md)
remains a separate historical record.

## Verified implementation

- **145 standalone SDK tests passed** in the no-network Python 3.11 container.
  This includes text retrieval, actual local OCR, spatial citations and the new
  review workflow. One production-service parity test was deliberately
  deselected because the lightweight runtime does not install the agent stack.
- **35 focused review tests passed** separately on Windows. They overlap with
  the container suite and should not be added to it as unique cases.
- **16/16 review integration controls passed**, including actual Tesseract OCR
  followed by authored full-page annotations and authored approvals.
- Existing **25/25 multimodal controls** and **10/10 text controls** passed again.
  They are authored implementation controls, not external-document QA scores.
- An actual **0.3.0 wheel** was built, installed into a disposable target and
  exercised from an application outside the checkout. Network sockets and
  platform/provider imports were blocked. Text, native PDF, actual scanned-PDF
  OCR, reviewed grid lookup, reviewed arithmetic and revocation all passed.
- The authored scan and its citation overlays were visually inspected.
  The source is an image; no
  user document was uploaded or modified.
- Whole-repository Ruff and whitespace checks passed. The unchanged strict
  retrieval artifact regression passed after retraining, exact reproduction and
  holdout validation. All four commands exited zero; only a disposable Git index
  was used, and the real index was unchanged.
- The final syntax/privacy/link scan covered 645 prospective public files and
  399 Python modules, finding no configured-secret matches, private versionable
  paths or broken relative Markdown links. Four configured secret values were
  compared internally without printing them; this is not a universal secret audit.

The container ran as UID 65532 with a read-only source mount, no network,
capabilities dropped, bounded memory/CPU/process count and temporary storage.
The pre-existing OCR verification image was reused; there was no new package or
model download for runtime checks. A FastAPI/Starlette deprecation warning
remained a warning, not a test failure.

## What the integration control demonstrates

The authored scan contains a three-by-three table. Actual Tesseract extraction
returned readable text but no structured table grid. The original SDK therefore
could not calculate from non-existent cells. Explicit complete-page annotations
added a reviewed grid without rewriting that machine extraction.

`Requests After` then returned separate `Requests`, `After` and `125` citations,
all marked `reviewer_supplied`. Selecting the `100` and `125` cells produced
`25%` change through the existing Decimal calculator. The result retained the
source file hash, original page fingerprint, draft, approval and review bundle.
This is a controlled before/after workflow demonstration, not a model accuracy
comparison. The public test-only key and reviewer are explicitly authored.

The controls also denied wrong signing keys, rejected decisions, altered
provenance, expired use and revoked receipts. Unit tests additionally covered
in-flight revocation, source replacement/removal, rollback, callback failures,
stale pages, incomplete/overlapping manual grids, partial-document holds and
mixed machine/reviewer citation origins. A revocation-capacity edge case was
hardened to disable the authority rather than leave an unrevocable lease live.

Early test iterations caught missing test imports and a misplaced rollback-test
assertion; these were corrected before the final green runs. They are not
presented as clean first-attempt passes. No application-source failure was hidden
by deleting a test or weakening the existing retrieval promotion gate.

## Evidence and limits

Private source images, bundles, receipts, wheel logs and JUnit results are under
`data/evaluations/evidence-review-20261017/`. The review integration report
fingerprint is
`a885ec2398e941ce30349f384c610465514d9da29482ff12df2a92c436b95564`.
The installed wheel is 35,124 bytes, SHA-256
`f8575cf5441939bd003ba8e283e32dcfa458d4d1c4d8e657c7fe42cd45001114`.
No API calls, charges, package publication, staging, commit or Git push occurred.
The refreshed learned retrieval candidate still fails its ranking promotion gate
(NDCG delta `-0.033689`), so the default fixed-weight baseline remains unchanged.
Reproducibility passing is not presented as a learned-ranking quality gain.

HMAC is a shared-key integrity control, not proof of human identity or a legal
signature. Whole-page review is an application attestation, not verified visual
completeness. The review capability must remain behind authenticated application
actions; it is not an agent approval tool. Already-returned data is not erased
by revocation. Revocations and clock state are process-local and non-durable.

No chart/handwriting understanding, automatic scanned-table structure learning,
external-user adoption, real-document quality gain or complete multi-tenant
production deployment was established. See the
[review guide](reviewed-document-evidence.md) for API contracts and deployment
boundaries.
