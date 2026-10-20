# Document reasoning: implementation and execution audit

SDK 0.7 connects natural-language planning to the document evidence tools. The
implementation and local control checks passed. The real-document comparison is
a limited pilot, not proof of improved answer accuracy. In particular, financial
questions frequently need table structure that the pilot does not infer.

## What changed

`DocumentReasoningAgent` retrieves literal text and an application-approved fact
catalog, then asks an explicit planner for a small typed plan. Only quote,
registered lookup, bounded calculation and abstention are allowed. Unknown
candidate IDs cannot execute. Invalid plans can trigger bounded retrieval
expansion; conflicts, unit failures and provider exceptions do not trigger
retry-until-success behavior.

The optional Groq adapter uses a fixed HTTPS endpoint, explicit model selection,
one shared request counter, no automatic retries and content-free errors. It
requires external-processing consent. Missing token usage and unknown cost stay
null. The default SDK import still needs only Pydantic and makes no network call.
The [guide](document-reasoning-agent.md) explains the API and trust boundaries.

## Local verification

These checks ran in the existing Python 3.11.16 SDK container, without networking,
as a non-root user with a read-only checkout and a private writable evidence
directory. They are local results, not a GitHub Actions run or a full-platform
release certification.

| Check | Observed result |
|---|---|
| Complete standalone SDK suite | 293 passed; one production-service parity test deliberately deselected |
| Focused reasoning/provider/pilot suite on the Windows host | 42 passed |
| Existing text evidence controls | 10/10 passed |
| Actual local OCR and spatial/cell controls | 25/25 passed |
| Reviewed correction controls | 16/16 passed |
| Document revision and replay controls | 18/18 passed |
| Batch revalidation controls | 20/20 passed |
| Numeric reconciliation controls | 22/22 passed |
| New reasoning controls with actual PDFs and authored planners | 16/16 passed |
| SDK 0.7 wheel, installed outside the repository | Passed with sockets and platform/provider imports blocked |
| Source-bound retrieval artifact refresh and strict regression test | Passed reproducibility; quality promotion rejected |

The 127 workflow controls use authored documents/plans/approvals, not independent
human labels or live model reasoning. They cover actual extraction, OCR, numeric
tools, revocation, stale evidence, bounded recovery, tamper rejection and installed
package behavior. The final suite retains one existing Starlette/httpx
deprecation warning. CI now includes the new offline tests and reasoning controls;
no live API benchmark was added to CI.

The installed wheel is 60,246 bytes, SHA-256
`7a139eb1a89ba60458d73ecadf8307c0c529f7ed1b742fb73c1478b31fd67ab9`.
It exercised text reasoning, native PDF evidence, real Tesseract OCR, reviewed
grids and revocation, version replay, batch resume and cross-document numeric
reconciliation. No package was published or dependencies downloaded by this check.

PDF visual inspection covered the authored `v1.pdf`, JPMorgan physical page 85,
Nike physical page 62 and the returned 3M quote's physical page 62. The authored
table is readable and its Requests row is 100 to 125. The real pages expose
multi-column prose, financial table headers, units and a chart. Looking at these
pages is visual QA, not a completed human answer-labeling exercise. Poppler
reported missing display fonts for two font names; the inspected pages remained
readable. This does not certify all fonts or all pages.

## Frozen real-document cohort

The pilot used the pinned [FinanceBench source snapshot](https://github.com/patronus-ai/financebench/tree/cc39aeb4afdf33909ee1412188bf89035950c2eb).
The first twenty IDs in SHA-256 ID order were frozen before inspecting model
outputs. Nineteen distinct PDFs downloaded. Selection did not filter on answer,
PDF support or model success. The public [cohort record](../evals/experiments/document_reasoning_pilot_cohort.json)
contains IDs and hashes only. Full source files, gold labels, requests and
receipts remain under ignored `data/evaluations/document-reasoning-20261017/`.
No right to redistribute the underlying data/PDFs is asserted.

The cohort fingerprint is
`ea745b0883cf910e2cd6e69c9293f041c9d90ac4b3e0c4e8103b0484c33d57ab`.
Question text and retrieved candidate cards enter the model context; gold
answers, evidence-page labels and gold rationales do not. Each variant is given
the task's document. This is provided-document retrieval, not corpus-wide
document discovery. The ordinary RAG baseline may synthesize/reason; the verified
path can release only literal quotes or registered numeric tool results. Their
initial candidates and model settings match, but their output capabilities differ.

The completed offline admission run loaded eleven cases and held nine under the
strict whole-PDF text profile. Its planner deliberately abstains. Its eleven
paired records are not live comparisons and must not be added to a live sample.
The old container lacked FontTools and emitted embedded-font decoding warnings.
The optional PDF extra and Dockerfile now include FontTools; the live host had
FontTools 4.60.1 and pypdf 6.14.2. The old image was not rebuilt for this audit.

An initial preparation attempt encountered upstream metadata using `doc_name`
where an upstream example described `evidence_doc_name`. Intake now validates
either spelling and rejects contradictory aliases. Two host test invocations
used unsuitable temporary paths; rerunning with a valid test directory passed.
Neither harness failure was hidden as a successful run.

## Live provider accounting

The operator confirmed a Free-plan project and authorized at most twenty total
Groq requests, including diagnostics. The key was read privately from ignored
`.env`; no credential is included in commands, public artifacts or this report.

1. One initial benchmark POST using the old configured Llama model failed. The
   earlier adapter retained only `http_error`, so its original status/body is not
   recoverable. Zero paired model comparisons completed in that attempt.
2. One explicit model-inventory GET succeeded. The configured model was absent;
   `qwen/qwen3.8-27b` was present. This supports a configuration diagnosis, but
   does not prove the exact cause of the earlier HTTP response.
3. The operator approved Qwen with at most eighteen additional requests. The new
   run uses non-thinking JSON mode, temperature zero, a maximum of 600 output
   tokens, fifteen-second request-start spacing and no automatic retries.

The Qwen run finished without provider errors. It made sixteen requests, bringing
the cumulative task total to **eighteen of twenty**. Each completed pair used one
baseline request and one planner request; no recovery call was needed. Two
requests remained unused. The runner requires a three-request reserve before
starting another pair, so it did not spend those two on an incomplete comparison.
A Free-plan assertion is not a provider invoice: monetary cost remains unknown
rather than being fabricated as a measured zero.

| Frozen cohort outcome | Count |
|---|---:|
| Completed baseline/verified pairs | 8 |
| Unsupported under strict PDF admission | 9 |
| Not run because of the request reserve | 3 |
| Total selected cases retained | 20 |

On the eight paired cases, ordinary RAG returned three nonempty answers. The
verified path returned **one literal quote and seven model abstentions**: 12.5%
answer coverage on paired cases, not 12.5% accuracy. All eight answer/abstention
receipts passed provenance replay. The baseline and verified path both had zero
pure-number exact matches among three grammar-eligible cases; this diagnostic
does not judge the other answers. Initial retrieval included a labeled evidence
page on four of eight pairs. A page hit does not establish sufficient context.

Actual reported usage was 10,339 input / 508 output tokens for the baseline and
12,435 input / 279 output tokens for the verified planner. There were no missing
token fields in these sixteen successful responses. The measured mean/p95 times
were 5.24/12.57 seconds for baseline and 14.92/15.12 for verified. **These include
deliberate request pacing**, which disproportionately waits before the planner;
they are not an inference-speed or production latency comparison. PDF admission
and indexing are outside those timers. The recorded input preparation across the
eight pairs took 135.49 seconds; it excludes unsupported and budget-held cases.

The one returned quote, on `financebench_id_01858`, pointed to 3M physical page
62. Its text begins mid-word and contains a joined word. The rendered PDF contains
the full statement; the emitted quote is literal to the extracted chunk but is a
readability/context defect. It was retained, not repaired in-place after seeing
the benchmark output. A valid provenance receipt did not prevent that defect.

No human answer labels were created. `answer_accuracy` and
`unsupported_answer_rate` therefore remain null. This pilot does **not** establish
a quality gain over ordinary RAG; its useful findings are low answer coverage,
strict-input exclusions and incomplete context/extraction. The next quality work
should target those failures and compare on a separately frozen cohort, not add
agent personas or tune against these eight observed outputs.

The live report fingerprint is
`e0340887c0216031411a202ca3ff0d743423dbaeb556a4505d07bf6fa1619e83`;
its `cases.json` SHA-256 is
`5d325078c288dd75d9965fd2a6a11a3140b124bc22373c7fdcd6bd3d23b18fe9`.
The original failed attempt, offline admission, provider diagnostic and successful
Qwen run remain in separate private directories. No failed run was overwritten.

## Retrieval artifact maintenance

SDK source changes alter the source-only code-retrieval corpus. A temporary
prospective index included intended non-ignored source files without staging or
changing the real Git index. Training, exact reproduction, five-query holdout
validation and the strict checked-in-artifact test all passed. The refreshed
artifact is
`1733a609b606f810463c56f21d8f2a64a12aeb1cc96af72ecc79abcac84445cf`.

Training used 44 hard-negative pairs; training pairwise accuracy was 88.64%.
Both rankers retained recall/MRR 1.00, but candidate NDCG was 0.9257 versus
baseline 0.9594. Promotion remains rejected for the 0.0337 regression. The fixed
serving baseline and thresholds were not changed to make the upgrade look better.

## What still needs review

Citation/tool verification proves lineage and arithmetic, not relevance,
question interpretation, source truth or overall factuality. The current
financial pilot has no automatically inferred numeric registry. Unsupported PDFs
remain visible, and the public dataset may have appeared in model training.

Reviewers should inspect private `cases.json`, the source PDF and receipt, then
complete `review-template.json` into a separate labels file and run the documented
`score-reviews` command. Intake binds labels to predictions, rejects incomplete,
foreign or duplicated labels, and does not count abstention as a correct answer.
Reviewer identity is operator-attested, not authenticated by JSON.

Final static checks parsed all **417 versionable Python modules**, passed
repository-wide Ruff and `git diff --check`, and found no broken relative
Markdown links, versionable private paths or matches for the four configured
secret values across **672 versionable files**. This is not a full Git-history
secret audit or a proof that arbitrary undisclosed secrets do not exist.

No changes were committed, pushed, deployed or published during this audit.
