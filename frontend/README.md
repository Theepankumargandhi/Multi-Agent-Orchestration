# AgentForge Workbench

The Workbench is a browser interface to the existing FastAPI service. Version 0.11
connects research conversations, owner-scoped coding tasks and local document evidence. It does not replace
the agent runtime or silently enable experimental controls.

## Run locally

Start the existing backend from the repository root, using its normal private
configuration and dependencies:

```bash
python -m uvicorn service.service:app --host 127.0.0.1 --port 8000
```

In another terminal, with Node 24 LTS (verified locally on 24.13.1):

```bash
cd frontend
npm ci --ignore-scripts
npm run dev
```

Open `http://127.0.0.1:5173`. On Windows, use `npm.cmd` if PowerShell blocks the
`npm.ps1` wrapper. The development server forwards `/api/*` to
`http://127.0.0.1:8000`. To use another backend, set `WORKBENCH_API_URL` in the
terminal before starting Vite. It must be an HTTP(S) origin without credentials,
a path, query or fragment. It is a server-side setting, not a browser variable.

The frontend does not load `.env` files. Never put provider keys, auth secrets or
private data in `VITE_*` variables or frontend source. All configuration visible
to a browser is public to that browser's user.

## What is connected

- **Overview:** configured model count, explicit repository allowlist and feature
  flags. These are configuration facts, not provider or sandbox health checks.
- **Research:** login/register, saved conversations, POST-based streamed answers,
  actual tool activity, and explicit web approval in the same conversation.
- **Coding:** bounded repair submissions, durable owner-scoped history, lifecycle
  polling, test results, verification reasons, dossiers and approved patch access.
- **Documents:** opt-in private PDF/image uploads, isolated extraction and local OCR,
  source-page inspection, clickable literal citations, selected-cell arithmetic and
  immutable owner review. See the [document setup and boundaries](../docs/document-workbench.md).
  The [version-impact workflow](../docs/document-workbench-versions.md) adds explicit
  version lineage, extraction changes, bounded local revalidation and a fresh-result review queue.
  The [reconciliation workspace](../docs/document-workbench-reconciliation.md) freezes
  explicitly comparable source versions, checks literal numeric facts and withholds
  calculations on unresolved evidence. Owner reviews cannot choose a winning value.
  [Numeric claim audits](../docs/numeric-claim-audits.md) check explicitly mapped
  expected values and units, resume two claims per step, and export private evidence
  only after an acknowledgement. Display labels are not semantically verified.
  [Change-aware replay](../docs/change-aware-claim-replay.md) reruns a completed
  audit on an explicitly comparable target, preserving assertions and showing
  before/after support without copying reviews.
  [Evidence fragility](../docs/evidence-fragility-stress-tests.md) tests every
  single-source omission under the unchanged policy. Its matrix shows support
  loss and support appearing after omission as diagnostics, never source approval.
  [Portable replay bundles](../docs/portable-claim-replay.md) export a complete
  audit and private extraction snapshots for independent SDK recomputation,
  behind a separate full-snapshot acknowledgement. Replay is not source authentication.
  [Source-backed replay](../docs/source-backed-replay.md) re-extracts one stored
  original at a time and compares its full manifest with the frozen audit source,
  with explicit execution/OCR consent and a runtime slot shared with uploads.
  [PDF cell concordance](../docs/pdf-cell-concordance.md) adds an explicit PDFium
  read of cited native headers/values at their physical locations. The worker
  receives no expected text; disagreements and uncheckable cells stay visible.
  This shares the same runtime slot and does not change claims or owner reviews.
  [Rendered-cell OCR](../docs/pixel-evidence-concordance.md) adds separate explicit
  local-OCR consent, fixed raster/fringe/quality budgets and visible disagreement
  or uncheckable cells. It never supplies expected text to OCR, changes original
  extraction settings or turns matching pixels into approval.
- **Evaluations:** separate dashboards; private reports and investigation controls
  are not exposed here. Broader SDK document workflows remain separate.

Model requests start paused each session. Enable them explicitly, then submit a
research question or coding task. Document evidence uses no provider and does not
need that checkbox. The checkbox gates new UI requests; it is not a backend
billing limit and does not cancel work already submitted. There are no automatic
model requests, mutation retries, fake live-agent animations or hidden reasoning
displays. Configured model IDs are not validated against a provider inventory.

## Enable browser coding submissions

The backend must use `ENABLE_USER_AUTH=true`. Configure the existing coding
sandbox/provider, place approved repositories below `CODE_AGENT_REPOSITORY_ROOT`,
then set these in the backend's ignored configuration:

```dotenv
CODE_AGENT_ENABLED=true
WORKBENCH_REPOSITORIES=example,team/approved
```

The allowlist defaults to empty. It contains **relative IDs**, not host paths or
repository URLs. No directory is enumerated, cloned or uploaded by the UI. The
Workbench adapter enforces the allowlist and configured model IDs on submission;
it fixes the test command to `python -m pytest -q`. Browser clients cannot choose
shell arguments, sandbox images, execution policies, credentials or user IDs.

This allowlist is server-wide, not a new per-repository tenant ACL. The existing
direct `/code/tasks` API keeps its existing root/policy validation. Use separate
deployments or add proper repository authorization before offering untrusted
users different private source repositories.

Coding history is scoped to the authenticated owner and limited to fifty headers
per request. Detail and artifact access use the existing owner-checked APIs.
`awaiting_approval` exposes execution evidence, not the patch. **Approve patch
access** unlocks read-only viewing/download; it does not apply, merge or publish
anything. Rejection can delete the stored patch artifacts. Read the decision
before clicking.

## Session and failure behavior

Bearer tokens live only in React memory. Refresh and sign-out discard them;
there is no local/session storage persistence. Expiry and authenticated HTTP 401
responses return the user to sign-in. Sign-out aborts outstanding browser fetches,
but it does not revoke the issued token on the server before its expiry.

An uncertain coding submission is not retried automatically. An unchanged manual
retry within the mounted form reuses its idempotency key. After navigation,
refresh or any uncertain result, inspect history before submitting again.
Stopping a chat stream stops **viewing**, not guaranteed server cancellation.
Interrupted streams are not displayed as completed answers.

All output is rendered as text; untrusted HTML/Markdown is not executed. Tool
excerpts, patches and dossiers may contain private content: protect your account
and screen. The browser never receives a service-wide auth secret.

Document originals, previews and saved questions live in the backend's private
SQLite library, not browser storage or static assets. Uploads default to no OCR
and require a separately enabled Docker runtime. Delete removes the document and
its saved results/reviews and dependent comparisons; parent versions with children
cannot be deleted first. Frozen evidence sets also block source deletion: explicitly
delete a set and its evidence/reviews first, preserving the source documents.
Inspect the confirmation before using deletion. Broader SDK planning and reviewed
correction authority are not exposed by this UI.

## Docker and checks

From the repository root, the optional Compose profile adds the non-root static
frontend and same-origin API proxy:

```bash
docker compose --profile workbench up --build workbench
```

Open `http://127.0.0.1:8080`. This starts the existing backend dependencies as
needed; it can download/build substantial images. The browser port is loopback
only. Existing Streamlit services are preserved. The static image copies only
frontend assets, sets a restrictive CSP and uses no response buffering for SSE.
This is not an internet-ready TLS deployment; the development server must not be
publicly exposed. `npm run preview` serves assets only, without an API proxy.

```bash
npm run build
npm test
```

Tests use authored HTTP responses with no model calls. Backend controls live in
`tests/test_workbench.py`, `tests/test_document_workbench.py`,
`tests/test_document_workbench_versions.py`, `tests/test_document_workbench_reconciliation.py`,
`tests/test_document_claim_audits.py`, `tests/test_document_claim_replay.py` and
`tests/test_document_claim_stress.py`, `tests/test_document_claim_bundle.py` and
`tests/test_document_source_replay.py`, `tests/test_document_cell_concordance.py`
`tests/test_document_cell_worker.py`, `tests/test_document_pixel_concordance.py`
and `tests/test_document_pixel_worker.py`. The document tests
use authored extraction by default; set `AF_RUN_DOCUMENT_DOCKER=1` to exercise the
dedicated local parser image, including real PDF/OCR, version revalidation,
cross-report conflicts, guarded arithmetic, resumable audits, changed-evidence
replay, source-omission diagnostics, portable offline recomputation and cleanup.
The source-backed check also exercises fresh original extraction and coherent
snapshot corruption detection. It does not authenticate a publisher.
The second-engine control additionally tests physical page-two cells, wrong
localization despite matching text elsewhere, and rotated-geometry abstention.
Pixel controls exercise real visible-versus-hidden text, fixed boundary guards,
confidence withholding and bounded OCR scheduling without provider calls.
See the repository's
[Workbench guide](../docs/workbench.md) for architecture and verification limits.
