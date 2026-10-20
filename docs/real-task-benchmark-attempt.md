# Real-task benchmark attempt — October 3, 2026

## Latest follow-up — October 4, 2026: Docker works; live gaps remain

The [full implementation audit](live-upgrade-audit-2026-10-04.md) supersedes the
readiness-only status below. Docker execution, counterfactual/minimization/mutation
drills and authenticated local HTTP checks succeeded. Eight accepted Groq calls
produced a smoke candidate; independent replay caught a patch-framing defect.
After an explicitly recorded framing-only repair, its generated code passed
fresh owner assertions and an additional boundary grid.

The Requests `1963` base checkout now has a compatible image and a frozen local
oracle that reproduces the bug, but inference was held before any real-issue
model call by the pilot's token reservation. A compact adaptive follow-up failed
closed on a schema rejection. This is not an official SWE-bench score or a
successful all-specialist benchmark. Three rejected requests returned no usable
usage; the audit records all eleven admitted calls rather than manufacturing
complete zero-cost accounting. No serving model or paid fallback was activated.

## Earlier readiness check — October 4, 2026: Groq works, Docker unavailable

The owner chose a Groq pilot, confirmed a Free-plan account, and authorized storing
the supplied credential in the ignored `.env`. The value was never printed or added
to versioned files. Because it was pasted into chat, it should be revoked after use.

A read-only models request succeeded. Neither `llama-3.3-70b-versatile` nor
`llama-3.1-8b-instant` appeared in that account's returned catalogue. One bounded
inference-readiness request used the listed `openai/gpt-oss-20b` model instead,
with strict JSON-schema output, low reasoning effort, a 1,024-token completion cap,
a thirty-second timeout, and zero SDK retries. It returned a valid `CodeAction`
for an authored boundary correction; the generated Python parsed successfully.
Actual provider usage was **289 prompt + 113 completion = 402 tokens**, including
57 reported reasoning tokens within completion usage. End-to-end request/validation
time was about 624 ms, not a coding-task latency measurement.

No generated code executed. This is a single readiness result, not a solved issue,
quality benchmark, tool-use study, or verified-PR run. The probe used the native Groq
SDK, not the project's older name-prefix provider selector. That selector and its
optional Groq integration dependency still need review before running the complete
workflow with the new model identifier; no silent OpenAI fallback was used.

Docker engine inspection failed because the `dockerDesktopLinuxEngine` named pipe
was absent. A non-destructive background launch was attempted and inspection failed
again. No reset, image/container/volume deletion, or host-execution substitute was
used. **Docker is the current execution blocker**, despite the earlier successful
restart below. The real SWE-bench pilot remains unexecuted.

Private, Git-ignored evidence is in `data/evaluations/groq-pilot-20261004/`:
`inference-readiness.json`, `generated-action.json`, and the single-request helper.
The helper refuses a repeat when its receipt exists. Model output is retained locally
without the key. Strict output support was checked against the
[official Groq structured-output documentation](https://console.groq.com/docs/structured-outputs).

## Earlier status after Docker restart

The owner restarted Docker, and the repeat checks succeeded. Docker Engine `29.5.3`
responded, and the existing sandbox image was present:

```text
sha256:301d860262ba8f6dc0f66bca2309bb54875979a0790df207d0833928f1ffa60f
```

The focused readiness suite then passed **all 11 tests with no skips** on Python 3.11.14,
including the coding-agent integration test in a real isolated Docker container. That
integration test uses a scripted action model and an authored fixture; it demonstrates
execution, patch generation, verification, cleanup, and source preservation, not
live-model success on real issues. The existing image was used, not rebuilt.

A single live inference-readiness request was then attempted against
`gpt-4o-mini-2024-07-18`, with a 16-token completion limit, a 20-second timeout, and
zero SDK retries. The API rejected it:

```text
HTTP 429
provider_error_type: insufficient_quota
provider_error_code: credit_balance_exhausted
```

**Docker is no longer the blocker. API inference quota is the current blocker.** No
successful model completion, real-issue coding-agent benchmark, or official grading
result was produced. The provider returned no usage, so the receipt leaves usage and
estimated cost unavailable rather than manufacturing a zero-cost measurement. No
additional inference retries or matrix runs were launched after this response.

The [official API error guide](https://developers.openai.com/api/docs/guides/error-codes)
distinguishes quota exhaustion from ordinary request-rate limiting. The operator must
restore usable API billing/quota for the project associated with the configured key,
or deliberately authorize another provider/model and a separately frozen experiment.
Do not paste credentials into chat. No billing settings, private environment values,
production defaults, or serving policies were changed.

The real-task dataset remains prepared but unexecuted. Once inference is available,
the per-task Requests checkouts, compatible images, fixed test commands, spending limit,
and independent official evaluation still need preparation; the smoke image alone does
not make the real-task environment ready.

Retry evidence, also Git-ignored in the same local artifact directory:

- `preflight-retry-01.json`: engine/image/model-metadata readiness;
- `readiness-docker-retry-v1.xml`: 11 passing tests, zero skips;
- `inference-probe-retry-01.json`: the one rejected capped API request;
- `inference_probe.py`: private helper for that probe, not a coding-quality evaluator.

## Initial attempt before Docker restart

The live benchmark could not get past the Docker readiness gate. Provider model metadata
access worked, the benchmark harness passed its focused checks, and three real Requests
tasks were selected and locked. **No coding task, model inference, or official grading
run executed.** There is no resolution rate, cost saving, or workflow winner to publish.

This report separates preparation from execution so that a setup failure cannot become
a misleading benchmark result. The local date was October 3 (America/New_York); the
final receipts were written shortly after midnight on October 4 UTC.

## What actually ran

| Check | Observed result | What it establishes |
|---|---|---|
| Existing adaptive-supervisor matrix validation | Passed: three variants, five frozen cases | Smoke configuration and dataset lock agree |
| Private provider configuration | Credential present; value never printed | A key is available to the process |
| OpenAI model metadata request | Access to `gpt-4o-mini-2024-07-18` succeeded | Metadata access, not inference quota or output quality |
| Benchmark/release readiness tests | 10 passed, one skipped on Python 3.11.14 | Harness contracts pass; Docker integration did not run |
| Real-task pilot import/validation | Three cases; matching content fingerprint | Real issue metadata is prepared, not solved |
| Docker engine and sandbox image inspection | Failed: engine pipe unavailable | Task execution cannot start; image availability is unknown |

The readiness tests cover the SWE-bench adapter, dataset fingerprints, scoring, portable
artifacts, dashboard boundaries, and paired releases. Their model/sandbox doubles are
not real coding results. The skipped test was the real-container coding integration
check in `tests/test_code_agent.py`.

Earlier readiness attempts encountered a denied default temporary directory and a
failure writing a nested artifact under a long workspace-based path. Those unsuccessful
receipts were retained. The final invocation used a fresh short temporary path, disabled
third-party pytest plugin autoload, and explicitly enabled `pytest_asyncio.plugin`.
That run passed; it was not a full repository test run.

## Why execution stopped

Starting the installed Docker Desktop outside the restricted shell did launch its
backend, but its log reported:

```text
starting services: initializing Inference manager:
listening on unix://.../Docker/run/dockerInference:
The file cannot be accessed by the system.
```

The exact inference path was a reparse-point file. After the owner approved a targeted
repair, a backup rename, native removal, and reparse-metadata operations were attempted
on that file alone. Windows refused them, including error 1920. An error message was
treated as failure even when a native utility returned exit code zero.

The socket was not successfully renamed or deleted. No factory reset, image/container/
volume deletion, Docker settings change, operating-system reinstall, or WSL reset was
performed. Docker was unavailable at the final readiness check. Repository tests were
not moved onto the host as a substitute for isolation.

## The real-task pilot

The source is the official [SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite),
at revision `b0dde1093fe417d83b7184254edf8199c1f0dff5`.
Selection was deterministic: the first three Requests instances in lexicographic
instance-ID order, without selecting by repair outcome.

- `psf__requests-1963`
- `psf__requests-2148`
- `psf__requests-2317`

Only instance ID, repository, base commit, issue statement, version, test IDs, and
environment-setup commit were projected into retained source records. No gold solution
patch, hidden test patch, hint text, or evaluation script was retained or sent to a
provider. The downloaded Parquet payload was parsed in memory, not saved as an agent
input. Each case has a separate checkout name to prevent different base commits from
sharing a mutable repository directory.

The pilot dataset fingerprint is:

```text
8eb9e69c4dc7741125a5e25ff97b0e80a51f8473ca2c327ddafefe212af65544
```

These public test-split records have deliberately been assigned to the local
**development/validation pilot**, not presented as an untouched generalization holdout.
They are a small, single-repository sample, and public benchmark training contamination
cannot be ruled out. The provisional test command still needs environment review;
locking task selection does not mean the execution configuration is ready.

## Finish the study on a working Docker host

1. Restore Docker startup without discarding existing data. Confirm that `docker info`
   returns a server response, not just that the desktop application is open.
2. Build/inspect the project's smoke sandbox and run a real-container readiness check.
   Keep the five authored smoke cases labelled as smoke evidence.
3. Prepare one Requests checkout per selected instance at its recorded base commit.
   Verify the revision and build compatible, trusted dependency images. Freeze image
   digests, test commands, policies, model, prices, and a spending limit before inference.
4. Execute verified PR, fixed three-team tournament, and adaptive supervisor on the
   identical pilot. Monitor every case and preserve losing attempts and infrastructure
   errors rather than discarding them. Do not activate the reviewed shadow advisor.
5. Independently grade generated predictions with the
   [official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/).
   Hidden evaluator tests must remain outside agent-visible checkouts. Existing local
   tests passing is not an official resolved result.
6. Document resolution, regressions, calls, usage completeness, latency, failures,
   paired comparisons, and cost estimates where accounting is complete. A later,
   independent held-out selection is needed before making comparative quality claims.

The existing matrix is still a smoke template with zero-price placeholders. Official
[OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini)
was checked for the pinned model and lists standard input/output rates of $0.15/$0.60
per million tokens. Supply applicable prices when freezing a live run; calculated
estimates are not invoices, and metadata access does not prove billable inference will
succeed. The matrix runner also needs credentials loaded into its process: unlike the
evaluation CLI, its entry point does not itself call `load_dotenv()`.

See [benchmark release engineering](benchmark-release.md) for the supported execution
and release commands. No production defaults, release approval, or serving model were
changed by this attempt.

## Local evidence

Generated artifacts remain Git-ignored under `data/evaluations/live-benchmark-20261003/`:

- `preflight-01.json` and `preflight-final.json`: redacted readiness receipts;
- `readiness-tests.xml` and `readiness-tests-v2.xml` through `readiness-tests-v4.xml`:
  unsuccessful and final successful focused-test receipts;
- `requests-pilot.jsonl` and `requests-pilot.lock.json`: selected issues and task lock;
- `requests-pilot-source.json`: upstream revision, source digest, and selection rule;
- `preflight.py` and `prepare_pilot.py`: private preparation helpers, not production APIs.

The private environment remains ignored. No credentials are included in this report.
Benchmark scores remain unavailable, not zero. The initial Docker failure is historical;
the current API-quota blocker and successful container checks are recorded above.
