# Saved-evidence revalidation: local execution audit

SDK 0.5 adds bounded revalidation of saved spatial answers and calculations.
This is a local implementation audit, not a complete-platform certification,
live-model benchmark or real-document accuracy claim. Previous audits remain
historical records.

## Actual document workflow

Five authored PDFs went through native extraction. The old report has Requests
going from `100` to `125`, giving `25` percent change. The revision inserts a cover
page, reorders rows and changes the final value to `150`.

The batch refreshed a saved latency question against the revised corpus, verified
its new receipt and citations on physical page 2, then replayed request growth
as `50`. A saved revenue sum stayed unchanged. Only request growth entered the
changed-result queue. The application reconstructed the batch between steps and
resumed from JSON; completed answer work was not called again. Queue output
contained no excerpts or questions.

**20/20 batch workflow controls passed**, including assessment-only defaults,
a zero answer budget that still permits tool replay, ambiguous-label blocking,
mixed-currency abstention, wrong-target rejection and altered counters. These
are authored expectations, not an external quality benchmark.

## Verification scope

**206 standalone SDK tests passed** in the offline Python 3.11 container. The
suite covers text retrieval, HTTP integration, actual local OCR,
spatial citations, reviewed evidence, version impact and batch revalidation.
One production-service parity test is deliberately excluded: the image does not
install the full platform. The existing FastAPI/Starlette deprecation warning is
not a failure. Final JUnit results are retained privately.

**96 focused Windows tests passed** for reviewed evidence, revision impact and
batch revalidation. These overlap with the container suite, not additional
unique tests. Revocation, expiry and changed source bytes invalidated saved
reviewed checkpoints without executing the target answer backend.

The existing control runners were rerun on SDK 0.5: text **10/10**, multimodal
**25/25**, reviewed evidence **16/16**, individual revision/replay **18/18**.
With the new run, this is **89/89 authored controls**, separate from pytest counts.
Review approvals are authored test capabilities; no human document review or
production identity integration is claimed.

Batch unit controls additionally cover JSON round trips, changed questions,
changed configuration, unknown custom-backend billing, no-backend checkpoint
verification, failure accounting without retries, altered answers/summaries,
source authority, safe queues and new evidence for a previously unanswered query.

The **installed 0.5.0 wheel** passed from a disposable application outside the
checkout with sockets and platform/provider imports blocked. Text evidence,
native PDF math, actual scanned-PDF OCR, reviewed lineage/revocation, version
replay, batch resume and changed-result queue checks all passed. Its 46,315-byte
wheel has SHA-256
`b3c6c9e58d932661d8d1ad6468915219004958d5700fed8a321f32eb45bf4eb1`.
No dependencies were downloaded and no package was published.

The old page and revised physical page 2 were rendered with Poppler and visually
inspected. Headings, labels and values were legible and
unclipped; the reordered Requests row agreed with the revalidation evidence.
Unused Symbol/ArialUnicode font warnings did not affect the inspected English
content. This supports the fixture, not general layout or OCR accuracy.

## Environment and evidence

Runtime checks reused `agentforge-document-sdk:multimodal-20261017`: Python 3.11.16,
Tesseract 5.5.0, pdfplumber 0.11.10, pypdfium2 5.14.0, Pillow 12.3.0 and ReportLab
5.0.1. The container had no network, a read-only source mount, UID 65532, dropped
capabilities, and bounded CPU, memory, process count and temporary storage. Only
the private evidence directory was writable. There were no provider requests,
model downloads or API charges.

Private reports, PDFs, checkpoints, renderings, wheel logs and JUnit results are
under `data/evaluations/document-revalidation-20261017/`. The batch-control report
fingerprint is
`0aded93e1f860dc6a6899a4b704bda5a01d07163c03785e1a65d3ff676bc7c21`.

The first full suite caught a hard-coded import/version assertion expecting
0.4.0; 203 other tests passed. The assertion was updated and the suite rerun.
The first focused run also caught a test adapter using `external` instead of
the required `external_processing` declaration. That fixture was corrected.
Neither failed run is presented as passing verification.

Source-bound retrieval training, exact reproduction, five-query holdout validation
and the unchanged strict checked-in-artifact test exited zero through a disposable
prospective index. The actual Git index was unchanged. The refreshed artifact
fingerprint is `8b898219333d201c6d27232c036ebac002bb4797edaed0413de370e58c8a5f52`.
It still regresses NDCG by `0.033689`: promotion remains rejected and the fixed
baseline stays active. Reproduction is not evidence of improved AI quality.

Whole-repository Ruff and whitespace checks passed. The final syntax/link/privacy
scan covered 656 prospective public files, including 406 Python modules. It found
no configured-secret matches, private versionable paths or broken relative
Markdown links. Four configured values were compared internally without printing
them; this check is not a universal secret audit.

## Boundaries

Checkpoints are not signed execution logs. Protected storage, authenticated
reviews and concurrent-worker coordination belong to the application. An SDK
answer attempt is not a provider-request count or spending cap; unknown billing
remains `None`. A crash before checkpoint persistence or replay of an older
checkpoint can repeat external work. No exactly-once guarantee, durable worker
journal, background monitor, automatic approval or deployment is included.

Existing extraction limits apply: complex tables, OCR omissions, chart meaning
and semantic contradictions can remain unsupported or undetected. Current review
authority is enforced, not transferred. This iteration does not claim real-user
adoption, live-model gains or that all unrelated project code was retested. No
Git staging, commit or push was performed. The [integration guide](document-revalidation.md)
describes exact API behavior and reproducible commands.
