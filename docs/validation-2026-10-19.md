# Validation record: October 19, 2026

This record covers the working-tree update containing the document SDK,
Workbench, coding-agent workflows and their evaluation controls. It separates
completed checks from the remaining merge gates.

## Completed checks

| Area | Result |
| --- | --- |
| Frontend | 69 tests across 12 files passed; formatting, TypeScript and production build passed |
| Host-side integrations | 132 tests passed, including real PDF/OCR, native and rendered-cell checks, patch verification and dependency-migration controls |
| Offline AI evaluations | All 69 selected CI commands passed across the original batch and one fresh-output rerun |
| Standalone SDK | Version 0.10.0 wheel built and installed outside the checkout; text evidence, scanned-PDF OCR, arithmetic, review revocation, revision comparison and reconciliation checks passed |
| Retrieval reproducibility | Training, exact artifact reproduction and holdout evaluation passed with the provisioned native parsers |
| Python lint | Repository-wide Ruff passed |
| Frontend dependency audit | No reported vulnerabilities in the lockfile audit |
| Publication checks | No configured-secret matches, private versionable paths or broken relative Markdown links |

The offline controls exercise orchestration, recovery, budgets, memory isolation,
grounding, uncertainty, preference and verifier gates, planning, retrieval and
document evidence. They use authored inputs and scripted model responses where
specified; no live provider requests were made.

The first offline preference command stopped because its default report path
already contained different evidence. A fresh output path passed without changing
the gate or overwriting the earlier report. The first full-suite collection also
identified missing dashboard dependencies in the validation controller. Those
dependencies were installed before starting the coverage run.

The SDK installation check blocked network access and platform/provider imports.
Its wheel was 67,301 bytes, with SHA-256
`ab99b11ed0496c3ce0f41b98eced510bd1ab3224e3436ab3ce2c344b5aa214bb`.
It verified that the installed package, rather than the source checkout, supplied
the document implementation.

## Retrieval decision

The five-query holdout reports baseline/candidate recall of 1.00/0.90, MRR of
1.00/1.00 and NDCG of 0.9594/0.8820. The learned candidate remains rejected;
reproducibility does not override the regression gate or change the serving
baseline. Source-bound checks used a disposable prospective index to include
the new source files before staging.

## Document-controller CI import fix

The isolated document job initially stopped with twelve collection errors:
importing a document adapter also imported `service.service` through the package
initializer, which then required `dotenv` and the rest of the application stack.
The package now loads the application only when a caller explicitly requests
`app`. The existing `from service import app` entry point is preserved.

Four fresh-process regression tests check the lightweight import boundary,
document adapters, cached application export and unknown attributes. They block
application/provider and optional parser imports explicitly. CI now runs these
checks using the document controller's existing dependency list.

The clean-controller run also exposed an authority-test fixture that unnecessarily
called the optional image/PDF extractor. That unit test now uses an authored,
source-hash-bound manifest with the real approval authority. It verifies a
supported numeric receipt before checking that the portable bundle builder
rejects the live reviewed source. The security assertion remains active.

The host document selection passed all **300 tests**, including the actual
Docker PDF, OCR, replay and cell integrations. The final fixture/import selection
passed **29 tests** in both the clean Python 3.11 controller and the host
environment. Existing service-hardening and Workbench contracts passed **30
tests**. The strict source-bound retrieval regression passed after the initializer
change; its learned candidate remains rejected, with serving defaults unchanged.
These counts describe separate commands, not a single combined test run.

The complete selection also passed in the clean Python 3.11 controller:
**290 passed and 10 skipped**. Nine skips were the explicit Docker integrations,
which ran on the host; the remaining skip was an optional-parser SDK fixture,
also covered by the host selection. The controller confirmed that `dotenv`,
LangChain, LangGraph and model-provider packages were absent. No application
dependencies were added to make the document job pass.

## Migration-recovery CI image setup

The metamorphic job built only `agentforge-upgradepilot-sessions:local`, while
the recovery fixture inherits `agentforge-upgradepilot:local`. A fresh runner
therefore failed at image inspection before the study could be frozen.

The job now gives one pinned build both aliases and checks that they resolve to
the same immutable image ID before either evaluation starts. A three-case
workflow regression test checks the required aliases for the metamorphic,
main-test and portable-patch jobs. Image pinning, local-only resolution and
recovery admission rules are unchanged.

The workflow regression and existing study/recovery selection passed **36
tests**. The unchanged strict retrieval-artifact check also passed; this CI-only
change did not require retraining or activate a model.

The corrected build and alias preflight were then executed locally. The actual
Docker recovery evaluator passed **all six controls** in 175.87 seconds:
explicit interrupted-run recovery, retaining uncertain usage, reusing completed
results, detecting receipt tampering, adopting a saved receipt, and withholding a
retry when its budget was exhausted. It used twelve scripted specialist calls
and zero live provider requests. Some storage fault windows deliberately replay
existing controller receipts; they are not additional model executions.

## Remaining merge gates

The earlier broad backend coverage run finished with **2,191 passed, 36 skipped
and two failures**. It overlapped working-tree updates rather than testing one
frozen final commit. Its source-bound retrieval failure was followed by an
artifact refresh and passing strict reruns. The Docker-output UTF-8 regression
test still needs review. These targeted fixes are not a clean full-suite pass;
the remaining test and remote CI results must be resolved before merging.

The unwaived service dependency audit reported 12 entries across two packages,
representing eight distinct advisories after duplicate IDs were collapsed:

- `cryptography` 46.0.7: four distinct advisories. The declared `<47` upper bound
  prevents installing the currently reported fixes. For example, the
  [PKCS#7 decryption advisory](https://github.com/advisories/GHSA-g6cj-pr64-35w5)
  lists 50.0.0 as patched, and the
  [wheel OpenSSL advisory](https://github.com/advisories/GHSA-537c-gmf6-5ccf)
  covers the bundled-library issue.
- ChromaDB 1.5.9: four distinct advisories, with no fixed version listed in this
  audit. The existing embedded-only deployment constraint and overdue exception
  review remain relevant; they are not a clean audit result. See the
  [ChromaDB code-injection advisory](https://github.com/advisories/GHSA-36p7-vc44-83pf)
  and the [exception record](security/audit-exceptions.md).

No vulnerability ignores, quality thresholds or serving defaults were relaxed.
These dependency findings are separate from the
[publication secret check](publication-security-check-2026-10-19.md): the latter
found no configured credentials or private runtime data in the checked content.

Detailed command logs, JUnit output, dependency reports and generated document
fixtures remain in ignored local storage. Public documentation contains only
the project-level results and operating limits.
