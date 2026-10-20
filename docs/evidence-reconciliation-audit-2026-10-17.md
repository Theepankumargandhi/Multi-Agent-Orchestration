# Numerical evidence reconciliation: local execution audit

This records SDK 0.6's scoped table-fact reconciliation, reviewed literal
citations and calculations over agreed facts. It is not a complete-platform
release certification, real-world factuality benchmark or live-model comparison.
Earlier SDK audits remain historical records.

## What actually ran

Five authored PDFs went through native extraction. The `v1` and `same-content`
artifacts report Requests After as `125`; another report has `150`, with the
table reordered and moved to physical page 2. The application explicitly assigns
the same comparison scope to these controls; the SDK does not infer their period
or authority from the PDF.

The matching pair produced an agreed fact. Reconciled Before `100` and After
`125` produced a verified percent-change result of `25`. Adding the third report
returned both value groups, no winning value, and a calculation abstention. Two
matching values did not overrule the disagreement. Every numeric observation
retained original row-label, column-label and value citations.

**22/22 new actual-PDF controls passed.** They also covered currency mismatch,
duplicate labels, missing facts, raw-file aliases, explicit scope exclusion,
single-artifact policy, and modified receipt rejection. The `same-content`
fixture differs in PDF metadata, not table content: distinct raw hashes here
are deliberately not evidence of independent-source corroboration.

All existing control runners were rerun against SDK 0.6: text **10/10**, spatial/
actual local OCR **25/25**, reviewed evidence **16/16**, version impact **18/18**,
and batch revalidation **20/20**. With the new run, **111/111 authored controls
passed**. These are separate from pytest counts and are not real-document
accuracy measurements.

## Tests and package integration

The final standalone suite passed **251 tests**, with one production-service
parity test deliberately excluded because the lightweight image does not install
the full agent platform. An existing FastAPI/Starlette deprecation warning was
not a failure. The initial container run passed 246 tests before five additional
number-format and mid-lookup revocation cases were added and verified.

The focused Windows review/revision/revalidation/reconciliation run passed
**136 tests** before those final additions. The final Windows reconciliation run
passed **45 tests**. These runs overlap and must not be added as unique tests.
Controls cover majority disagreement, alias inflation, exact scope, equivalent
currency/sign/decimal forms, unsupported grammars, sparse/multiple tables,
duplicate labels, bounds, wrong/altered receipts, unit mismatch, zero denominators,
and calculation stability under a caller's low Decimal precision.

Reviewed tests preserve approval-origin wrappers and reject expired, revoked or
changed-source evidence. A later source revoking an earlier reviewed observation
during lookup caused the earlier values/citations to be discarded, even when
two other artifacts agreed. No authenticated human review is claimed: approvals
and annotations in these controls are authored test capabilities.

The SDK-only **0.6.0 wheel** was built and installed outside the checkout with
network sockets and platform/provider imports blocked. It passed text evidence,
native PDF math, actual scanned-PDF Tesseract OCR, reviewed citation lookup and
revocation, version replay, batch resume, cross-document agreement, conflict-held
calculation and reconciled Decimal arithmetic checks. No package was published
or dependencies downloaded. The 51,723-byte wheel SHA-256 is
`f5a24ff7b41bd97fb355534045b8debce21b91dce7cbf6bf3568da126795807b`.

## Visual and runtime evidence

Rendered previews were inspected for the old report and revised
physical page 2. Headings, row/column labels and `125`/`150` values were legible
and unclipped, consistent with the cited observations. Poppler emitted unused
Symbol/ArialUnicode font warnings; the inspected English content rendered
correctly. This checks the authored fixture, not general layout accuracy.

Runtime checks reused `agentforge-document-sdk:multimodal-20261017`: Python 3.11.16,
Tesseract 5.5.0, pdfplumber 0.11.10, pypdfium2 5.14.0, Pillow 12.3.0 and ReportLab
5.0.1. Containers ran without network, as UID 65532, with read-only source,
dropped capabilities and bounded CPU, memory, process count and temporary
storage. Only the private evidence directory was writable.

JUnit reports, extracted receipts, PDFs, renderings and wheel logs remain in
`data/evaluations/evidence-reconciliation-20261017/`. The reconciliation-control
report fingerprint is
`df497311353655c4e3d909d0a27d390898c33a1794668999dc91be72a28c9592`.

The retrieval artifact was refreshed for the changed source without weakening
the promotion gate. Training, exact reproduction, five-query holdout validation,
and the unchanged strict artifact test exited zero through a disposable
prospective index; the real Git index was unchanged. Artifact fingerprint:
`eb7fe73867f386419a21be34fb2db20323cfedf68f81435d86fc02a9e7097d9d`.
The learned candidate still regresses NDCG by `0.033689`, so promotion stays
rejected and the fixed baseline remains active.

Final whole-repository Ruff and whitespace checks passed. A syntax/link/privacy
scan covered 662 prospective public files and 410 Python modules, with no
configured-secret matches, private versionable paths or broken relative Markdown
links. Four configured values were compared internally without printing them;
this is not a universal secret audit.

## Limits and next review

This compares caller-scoped numeric table cells, not arbitrary prose claims,
charts, synonyms, reporting periods or semantic contradictions. First-row and
first-column labels are conventions. Scope, metric dimensions, scale and
publisher authority remain application responsibilities. Raw-file diversity
does not prove independent evidence; agreeing sources can all be wrong.
Receipts are integrity bindings, not signed approval or truth certificates.
Reviewed authorization retains existing process-local revocation limits.

The human review should check three outcomes in the [guide](evidence-reconciliation.md):
agreement permits the explicit tool, disagreement blocks it without majority
voting, and missing/ambiguous/unauthorized evidence remains unresolved.
No provider request, model download, API charge, automatic approval, deployment,
Git staging, commit or push was performed. Unrelated platform features were not
all rerun in this iteration.
