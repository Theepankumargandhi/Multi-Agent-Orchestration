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

## Remaining merge gates

The full backend coverage run was still in progress when this PR snapshot was
prepared. Its completed result and remote CI status must be reviewed before
merging. The integration selection above is not a substitute for that full run.

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
