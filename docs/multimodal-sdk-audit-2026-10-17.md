# Multimodal evidence SDK: local implementation audit

This records the optional SDK 0.2 spatial path, not a claim that every platform
subsystem was revalidated. The original text API and its historical
[October 16 audit](document-sdk-audit-2026-10-16.md) remain separate.

## What actually ran

- **110 standalone tests passed** in Python 3.11, including actual local OCR on
  PNG, JPEG, scanned PDF and a mixed native/scanned document. One production-core
  parity test was deliberately deselected because this environment contains no
  agent platform. The container had no network, ran as UID 65532, and mounted
  source read-only with bounded temporary storage, memory, CPU and process count.
- **25/25 authored workflow controls passed** with Tesseract 5.5.0. These checked
  native table geometry, literal lookup with separate axis/value citations,
  Decimal arithmetic, physical page references, OCR, abstention and tamper holds.
- A separate Windows synchronous SDK/evaluation regression run passed **113
  tests**, with four explicit local-Tesseract skips and three async cases
  deselected. The four actual OCR paths were exercised in the container, not
  claimed from those Windows skips.
- The standalone wheel was built and installed into a disposable target. A
  separate application outside the checkout used that installed module for
  text citations, native PDF lookup, cell arithmetic and actual scanned-PDF OCR,
  with network sockets and platform/provider imports blocked.
- The native table and scanned PDF were rendered with Poppler and visually
  inspected, including citation-box overlays. The row-label, column-label and
  value boxes aligned with their table cells; the OCR box enclosed its cited
  line. The table grid, labels and scan text were legible and unclipped.
  These observations cover the authored fixtures, not general document-layout
  accuracy.
- Whole-repository Ruff and whitespace checks passed during implementation.
- The unchanged strict source-artifact regression passed after retraining,
  exact reproduction and holdout validation all exited successfully. A
  disposable index included intended public source; the real Git index was
  unchanged. No staging, commit or push was performed.
- The final syntax/privacy/link scan covered 639 prospective public files,
  including 395 Python modules. It found no configured-secret matches, private
  versionable paths or broken relative Markdown links. Four configured secret
  values were compared internally and never emitted. This is a scoped check,
  not a claim to have proved that no unknown secret can exist.

Runtime versions were Python 3.11.16, pdfplumber 0.11.10, pypdfium2 5.14.0,
Pillow 12.3.0 and ReportLab 5.0.1. ReportLab/pypdf generate test fixtures; they
are not new mandatory runtime dependencies. Installing/building the test image
downloaded public packages and Tesseract data. **Runtime verification** then ran
offline. No private document, cloud model call, API charge or package publication
was involved.

## Concrete results

The native PDF contains 15 addressed table cells. A `Revenue Q2` lookup returns
three separately verifiable quotes: `Revenue`, `Q2`, and `USD 150.75`.
`USD 120.50 + USD 150.75` produces `271.25 USD`. Requests rising from `100` to
`125` produce `25%` change. These are authored expected results, not financial
analysis or a model-generated numerical answer.

The actual OCR path reads `Recovery time is 42 seconds.` from the PNG, JPEG and
image-only PDF, and cites physical page 2 for the mixed document. Native pages
do not invoke OCR. Comma-ambiguous values, mixed units and zero denominators
abstain. Missing OCR, low-confidence words, blank pages, merged grids and
unsupported orientation hold the document; no partially extracted document is
silently declared fully answerable.

Early execution found and fixed two integration defects: eager evaluation-package
imports pulled LangChain into the SDK-only environment, and PDFium page handles
needed explicit cleanup rather than an unsupported context manager. Evaluation
exports now load lazily while preserving their public names. PDFium access is
serialized and page/bitmap handles are closed deterministically.

Two wider Windows test attempts stalled and were explicitly stopped; they are
not counted as passing. Linux container results above include the SDK async
wrapper. The final handoff identifies any additional Windows regression results
separately, without presenting interrupted runs as a clean full-suite pass. One
source-artifact validation subprocess also exceeded its initial 180-second
deadline; a longer, separately recorded retry was used rather than treating a
partially written report as a passing command.

The final installed wheel is 29,570 bytes, SHA-256
`fc33ad7f130c5535f88920609646b0315b3b5239c4000af50ab749d35d2fa257`.
The source-bound retrieval candidate still fails its ranking promotion gate
(NDCG delta `-0.029931`); the default fixed-weight baseline remains unchanged.
This is not an improvement claim for the learned ranker.

The SDK container emitted one existing FastAPI/Starlette HTTP-client
deprecation warning, not a test failure. No complete-platform green-suite claim
is made from these focused SDK/evaluation checks.

Private receipts, fixtures, wheel checks and previews are under
`data/evaluations/multimodal-20261017/`. They are ignored, not product source or
public evaluation data. The final workflow receipt fingerprint is
`20dc2df8f773e180144ca2b3faaacbb4e00ca4cf8f54924f5c1078d8b7728360`.

## What this does not establish

This is a small authored corpus with English printed text and simple ruled
tables. It does not measure external-document QA quality, handwritten OCR,
chart understanding, scanned table reconstruction, multilingual accuracy,
real-user adoption or downstream business correctness. The low-confidence
negative control is an explicitly labelled adapter double, not an empirical
calibration study. Tesseract can confidently misread or miss text.

Integrity receipts are not signatures, and this in-process parser is not a
hostile-upload sandbox. The test container is not a complete multi-tenant
production ingestion service. See [the SDK guide](multimodal-evidence-sdk.md)
for coordinate conventions, extraction limits, exact arithmetic semantics,
deployment boundaries and reproducible commands.
