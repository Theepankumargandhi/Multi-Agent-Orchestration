# Document-version impact: local execution audit

This records SDK 0.4's extraction diff, dependency impact and calculation replay.
It is not a fresh complete-platform pass or a general document-understanding
accuracy claim. Earlier SDK audits remain separate historical records.

## Actual execution

- **171 standalone SDK tests passed** in the offline Python 3.11 container,
  covering text, actual local OCR, spatial evidence, review and revision paths.
  One production-service parity test was deliberately deselected; the lightweight
  runtime does not install the full agent platform. The existing FastAPI/
  Starlette deprecation warning was not a failure.
- **61 focused Windows tests passed** for review and version changes. These
  overlap with the container suite and are not additional unique tests.
- **18/18 revision workflow controls passed** on five authored PDFs with actual
  extraction and Decimal replay. Existing text, multimodal and review control
  runners also passed **10/10**, **25/25** and **16/16**, respectively.
- The **installed 0.4.0 wheel** passed from a disposable application outside the
  checkout, with network sockets and platform/provider imports blocked. It
  exercised text, native PDFs, actual scanned-PDF OCR, reviewed evidence and the
  new page/row alignment and calculation replay.
- The before page and revised physical page 2 were rendered with Poppler and
  visually inspected, including original dependency and new operand boxes.
  They were legible, unclipped and aligned to the expected cells. This checks
  the authored fixtures, not general layout accuracy.

The container reused the existing OCR verification image. It ran as UID 65532,
with network disabled, a read-only checkout, capabilities dropped, and bounded
memory, CPU, process count and temporary storage. No new runtime dependency or
model download was required for this upgrade's verification.

## What changed in the fixture

The first report has Requests moving from `100` to `125`; the old verified
percent-change result is `25`. The second inserts a cover page, reorders the
table rows and changes the final value to `150`. Replay follows the row/column
labels, preserves operand order and returns `50`, with new citations on physical
page 2. The old receipt is not accepted by the new corpus.

A separate unit-conflict PDF changes one currency to EUR while the other remains
USD. The new tool receipt abstains; it does not invent a conversion. Duplicate
row labels hold replay. A metadata-only new version still needs new receipts,
even though the extracted values match. These are authored workflow expectations,
not held-out quality measurements.

Unit controls additionally covered column reordering, duplicate pages/literals,
budget exhaustion, unavailable extraction, parser-profile drift, tampered
receipts, raw UTF-8 text hashes, changed uncited content, reviewed approval
replacement and target revocation. An unchanged cited quote does not become a
claim that the best answer is unchanged.

## Reproducibility and evidence

Private PDFs, manifests, reports, overlays, wheel logs and JUnit results are
under `data/evaluations/document-changes-20261017/`. The final revision-control
report fingerprint is
`5f84b16428d14167edfcd45c701f68fa50bdf3b240e7a292a2116a517d3cfcad`.
The final wheel is 41,393 bytes, SHA-256
`91eaa78578fe0dbe1af85e3f37c1718def89c9eaab15808569e0f34493e888a5`.

Final review made the new text-hash field use the same raw UTF-8 SHA-256 convention
as the existing SDK. Controls, tests and the installed package were rerun after
that correction. One automatic permission-review timeout prevented a command
from starting; the retry succeeded. It was not presented as a test failure or a
passing execution.

The source-bound retrieval artifact was refreshed without changing its promotion
gate. The learned candidate still regresses NDCG (`-0.033689`), so the default
fixed-weight baseline remains unchanged. Reproduction passing is not a ranking
quality improvement. Retraining, exact reproduction, holdout validation and the
unchanged strict artifact test all exited zero through a disposable index; the
actual Git index was unchanged. The final Ruff/whitespace checks passed. A syntax,
privacy and link scan covered 651 prospective public files and 403 Python modules,
with no configured-secret matches, private versionable paths or broken relative
Markdown links. Four configured values were compared internally without printing
them; this is not a universal secret audit. No Git staging, commit, push, package publication, provider
call or API charge was performed.

## Boundaries

Alignment is deterministic and conservative, not semantically authoritative.
Unseen pixels, chart meaning, confident OCR omissions, complex page/line splits
and unfamiliar table schemas can be missed or remain unmatched. A completed diff
is not proof of complete alignment. Current-use review authorization is checked;
approvals cannot be carried across versions. Process-local revocation and the
underlying hostile-upload limitations remain unchanged.

No real external-user adoption, real-document accuracy gain, visual QA benchmark,
automatic contradiction detection or deployed monitoring service was established.
See the [integration guide](document-version-impact.md) for exact semantics and
reproducible commands.
