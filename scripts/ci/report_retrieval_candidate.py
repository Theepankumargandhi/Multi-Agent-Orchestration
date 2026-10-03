"""Publish experimental retrieval evidence without confusing a hold with code failure."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code_agent.retrieval_backends import LearnedFusionScorer  # noqa: E402
from code_agent.retrieval_learning import (  # noqa: E402
    FusionArtifact,
    FusionValidationReport,
    fusion_promotion_reasons,
)


def candidate_summary(report_path: Path, artifact_path: Path) -> tuple[bool, str]:
    report = FusionValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    artifact = FusionArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    scorer = LearnedFusionScorer.load(artifact_path)
    if report.schema_version != "1.0" or report.artifact_fingerprint != artifact.artifact_fingerprint:
        raise ValueError("candidate report does not bind the reviewed artifact")
    base, learned = report.base, report.learned
    if (base.index_fingerprint != report.index_fingerprint
            or learned.index_fingerprint != report.index_fingerprint
            or report.index_fingerprint != artifact.index_fingerprint
            or base.dataset_fingerprint != report.dataset_fingerprint
            or learned.dataset_fingerprint != report.dataset_fingerprint
            or base.total != learned.total or base.total < 1
            or base.top_k != learned.top_k or base.max_tokens != learned.max_tokens
            or base.strategy != learned.strategy
            or base.fusion_backend != "fixed-weight-v1" or learned.fusion_backend != scorer.name):
        raise ValueError("candidate report has inconsistent comparison provenance")
    if [(case.case_id, case.relevant_paths) for case in base.outcomes] != [
        (case.case_id, case.relevant_paths) for case in learned.outcomes
    ]:
        raise ValueError("candidate report compares different cases")
    for actual, expected in (
        (report.recall_delta, learned.recall_at_k - base.recall_at_k),
        (report.mrr_delta, learned.mrr - base.mrr),
        (report.ndcg_delta, learned.ndcg_at_k - base.ndcg_at_k),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("candidate report has inconsistent metric deltas")
    reasons = fusion_promotion_reasons(base, learned, artifact.pairwise_accuracy)
    if report.promotion_reasons != reasons or report.promotion_approved != (not reasons):
        raise ValueError("candidate report contradicts the strict promotion policy")
    decision = "PASS: eligible for owner review" if not reasons else "HOLD: not approved for serving"
    lines = [
        "## Experimental retrieval candidate", "", f"Decision: **{decision}**", "",
        "| Metric | Fixed-weight baseline | Learned candidate |", "|---|---:|---:|",
        f"| Recall@{base.top_k} | {base.recall_at_k:.6f} | {learned.recall_at_k:.6f} |",
        f"| MRR | {base.mrr:.6f} | {learned.mrr:.6f} |",
        f"| NDCG@{base.top_k} | {base.ndcg_at_k:.6f} | {learned.ndcg_at_k:.6f} |", "",
        f"Artifact: `{artifact.artifact_fingerprint}`", "",
        "Code CI does not approve or activate this model. The default fixed-weight baseline is unchanged.",
        "Run the separate Retrieval Candidate Promotion workflow (or --require-promotion CLI) before owner activation.",
        "This repository-specific regression set is not evidence of general model quality.",
    ]
    lines.extend(f"- Hold reason: {reason}" for reason in reasons)
    return not reasons, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        approved, summary = candidate_summary(args.report, args.artifact)
    except (OSError, ValueError, KeyError):
        print("Candidate evidence is missing, invalid, or inconsistent.", file=sys.stderr)
        return 2
    print(summary)
    if not approved:
        print("::warning title=Experimental retrieval candidate held::Quality gate rejected the candidate; baseline remains the default.")
    if summary_path := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
