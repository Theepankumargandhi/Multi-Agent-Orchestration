"""Measure evidence that survives code-context packing, not only retrieved filenames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from code_agent.context_evaluation import DirectoryWorkspace, index_fingerprint
from code_agent.intelligence import CodeIntelligenceIndex, ContextPack
from code_agent.models import CodeContextReceipt
from code_agent.retrieval_backends import RetrievalConfig

DEFAULT_ROOT = Path(__file__).parent / "fixtures" / "retrieval_spans"
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "retrieval_spans.jsonl"


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def span_digest(lines: list[str], start: int, end: int) -> str:
    return hashlib.sha256("\n".join(lines[start - 1:end]).encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceSpan(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def confined(self):
        path = PurePosixPath(self.path)
        if (path.is_absolute() or ".." in path.parts or "\\" in self.path
                or ":" in self.path or path.as_posix() != self.path
                or any(ord(character) < 32 for character in self.path)):
            raise ValueError("evidence paths must be canonical repository-relative paths")
        if self.end_line < self.start_line or self.end_line - self.start_line >= 500:
            raise ValueError("evidence span is reversed or too large")
        return self


class SpanCase(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    family_id: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=5, max_length=4000)
    variant: str = Field(default="original", min_length=1, max_length=80)
    required_spans: list[EvidenceSpan] = Field(min_length=1, max_length=20)
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"
    reviewer: str = Field(default="", max_length=120)
    reviewed_at: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def distinct_labels(self):
        if not self.query.strip() or not self.id.strip() or not self.family_id.strip() or not self.variant.strip():
            raise ValueError("case identifiers, variants, and queries must not be blank")
        seen = set()
        for span in self.required_spans:
            positions = {(span.path, line) for line in range(span.start_line, span.end_line + 1)}
            if positions & seen:
                raise ValueError("required spans must not overlap")
            seen.update(positions)
        if self.review_status == "human_reviewed":
            from datetime import datetime
            if not self.reviewer.strip():
                raise ValueError("human-reviewed labels require a reviewer")
            try:
                reviewed = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("human-reviewed labels require an ISO review timestamp") from exc
            if reviewed.tzinfo is None:
                raise ValueError("review timestamp must include a timezone")
        return self


class SpanPlan(StrictModel):
    baseline: Literal["lexical", "lexical_graph", "hybrid", "hybrid_rerank"] = "lexical_graph"
    candidate: Literal["lexical", "lexical_graph", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    baseline_packing: Literal["legacy", "balanced_v1"] = "legacy"
    candidate_packing: Literal["legacy", "balanced_v1"] = "legacy"
    top_k: int = Field(default=4, ge=1, le=30)
    token_budgets: list[int] = Field(default_factory=lambda: [256, 512, 1024], min_length=1, max_length=12)
    primary_budget: int = Field(default=512, ge=64, le=32000)
    minimum_families: int = Field(default=6, ge=2)
    minimum_span_recall: float = Field(default=0.8, ge=0, le=1)
    maximum_slice_regression: float = Field(default=0.05, ge=0, le=1)
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)

    @model_validator(mode="after")
    def fixed_primary(self):
        if (self.primary_budget not in self.token_budgets
                or len(set(self.token_budgets)) != len(self.token_budgets)
                or any(not 64 <= budget <= 32000 for budget in self.token_budgets)):
            raise ValueError("budgets must be unique, bounded, and include the fixed primary budget")
        return self


def load_cases(path: Path) -> list[SpanCase]:
    cases = [SpanCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_cases(cases)
    return cases


def validate_cases(cases: list[SpanCase]) -> None:
    if not cases or len(cases) > 1000 or len({case.id for case in cases}) != len(cases):
        raise ValueError("span dataset must be nonempty with unique case IDs")
    queries: dict[str, str] = {}
    anchors: dict[tuple[str, int], str] = {}
    for case in cases:
        query = " ".join(case.query.lower().split())
        if query in queries:
            raise ValueError("duplicate normalized queries cannot inflate comparisons")
        queries[query] = case.family_id
        for span in case.required_spans:
            for line in range(span.start_line, span.end_line + 1):
                key = (span.path, line)
                if key in anchors and anchors[key] != case.family_id:
                    raise ValueError("the same evidence target or overlapping spans cannot inflate independent family counts")
                anchors[key] = case.family_id


def validate_targets(index: CodeIntelligenceIndex, cases: list[SpanCase]) -> None:
    for case in cases:
        for span in case.required_spans:
            if span.path not in index.files:
                raise ValueError(f"target is absent from the indexed corpus: {case.id}")
            lines = index.files[span.path].content.splitlines()
            if span.end_line > len(lines) or span.sha256 != span_digest(lines, span.start_line, span.end_line):
                raise ValueError(f"stale source-bound evidence label: {case.id}")
            if not any(line.strip() for line in lines[span.start_line - 1:span.end_line]):
                raise ValueError("blank-only evidence spans are not meaningful targets")


def receipt_fingerprint(receipt: CodeContextReceipt) -> str:
    payload = {
        "query": receipt.query, "plan": receipt.query_plan, "strategy": receipt.strategy,
        "selected": [(item.path, item.sha256, item.snippet_sha256, item.rank) for item in receipt.selected_files],
    }
    if receipt.packing_policy != "legacy":
        payload.update(packing_policy=receipt.packing_policy, source_line_ranges=receipt.source_line_ranges,
                       packing_diagnostics=receipt.packing_diagnostics)
    return fingerprint(payload)


def visible_lines(index: CodeIntelligenceIndex, pack: ContextPack) -> set[tuple[str, int]]:
    """Validate emitted sections against receipts and count only complete source lines."""
    if pack.receipt.context_chars != len(pack.prompt_context):
        raise ValueError("context length contradicts its receipt")
    if (pack.receipt.fingerprint != receipt_fingerprint(pack.receipt)
            or pack.receipt.estimated_tokens != max(1, math.ceil(len(pack.prompt_context) / 4))):
        raise ValueError("context fingerprint or token estimate contradicts its provenance receipt")
    if len({item.path for item in pack.receipt.selected_files}) != len(pack.receipt.selected_files):
        raise ValueError("duplicate selected paths contradict context provenance")
    sections = pack.prompt_context.split("\n\n")[1:]
    if len(sections) != len(pack.receipt.selected_files):
        raise ValueError("context sections contradict selected-file receipts")
    visible = set()
    for section, item in zip(sections, pack.receipt.selected_files, strict=True):
        document = index.files.get(item.path)
        if document is None or document.sha256 != item.sha256:
            raise ValueError("context source provenance mismatch")
        header, separator, snippet = section.partition("\n")
        expected_header = (f"## {item.path} | {document.language} | "
                           f"symbols: {', '.join(document.symbols[:12]) or 'none'}")
        if not separator or header != expected_header or hashlib.sha256(snippet.encode()).hexdigest() != item.snippet_sha256:
            raise ValueError("emitted snippet does not match its provenance receipt")
        source = document.content.splitlines()
        for line in snippet.splitlines():
            match = re.fullmatch(r"\s*(\d+):(?: (.*))?", line)
            if not match:
                continue  # Compression gap markers carry no evidence.
            number, body = int(match[1]), match[2] or ""
            if not 1 <= number <= len(source):
                raise ValueError("emitted line number is outside its source")
            expected = source[number - 1].rstrip()
            if body == expected:
                visible.add((item.path, number))
            elif not expected.startswith(body):
                raise ValueError("emitted line contradicts its source")
            # A truncated prefix is not a complete evidence line.
    if pack.receipt.packing_policy == "balanced_v1":
        ranges = {}
        retained = 0
        omitted = 0
        for item in pack.receipt.selected_files:
            numbers = sorted(number for path, number in visible if path == item.path)
            grouped: list[list[int]] = []
            for number in numbers:
                if grouped and number == grouped[-1][1] + 1:
                    grouped[-1][1] = number
                else:
                    grouped.append([number, number])
            ranges[item.path] = grouped
            retained += len(numbers)
            omitted += len(index.files[item.path].content.splitlines()) - len(numbers)
        if (ranges != pack.receipt.source_line_ranges or pack.receipt.packing_diagnostics != {
            "retained_source_lines": retained, "omitted_source_lines": omitted, "partial_source_lines": 0,
        }):
            raise ValueError("balanced packing audit contradicts actual emitted source lines")
    return visible


def score_pack(index: CodeIntelligenceIndex, case: SpanCase, pack: ContextPack) -> dict:
    if pack.receipt.query != case.query:
        raise ValueError("context receipt belongs to a different query")
    visible = visible_lines(index, pack)
    required = set()
    complete = 0
    missing_spans = []
    for span in case.required_spans:
        positions = {(span.path, line) for line in range(span.start_line, span.end_line + 1)}
        required.update(positions)
        complete += positions <= visible
        if not positions <= visible:
            missing_spans.append({
                "path": span.path, "start_line": span.start_line, "end_line": span.end_line,
                "missing_lines": sorted(line for path, line in positions - visible),
            })
    paths = {span.path for span in case.required_spans}
    selected = {item.path for item in pack.receipt.selected_files}
    hits = len(required & visible)
    failure = "none"
    if complete < len(case.required_spans):
        failure = "file_miss" if not paths <= selected else "compression_loss" if not hits else "partial_evidence"
    return {
        "case_id": case.id, "family_id": case.family_id, "variant": case.variant,
        "file_recall": len(paths & selected) / len(paths),
        "line_recall": hits / len(required), "span_recall": complete / len(case.required_spans),
        "complete_evidence": float(complete == len(case.required_spans)),
        "context_precision": hits / max(1, len(visible)),
        "estimated_tokens": pack.receipt.estimated_tokens,
        "failure": failure, "context_fingerprint": pack.receipt.fingerprint,
        "missing_spans": missing_spans,
        "packing_policy": pack.receipt.packing_policy,
    }


METRICS = ("file_recall", "line_recall", "span_recall", "complete_evidence", "context_precision", "estimated_tokens")


def family_aggregate(rows: list[dict]) -> dict[str, float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
    if not groups:
        raise ValueError("cannot aggregate an empty comparison")
    return {metric: math.fsum(
        math.fsum(row[metric] for row in group) / len(group) for group in groups.values()
    ) / len(groups) for metric in METRICS}


def paired_family_interval(base: list[dict], candidate: list[dict], *, seed: str, samples: int) -> list[float]:
    if not 100 <= samples <= 10000:
        raise ValueError("bootstrap sample count must be bounded")
    if [(row["case_id"], row["family_id"]) for row in base] != [
        (row["case_id"], row["family_id"]) for row in candidate
    ]:
        raise ValueError("paired comparisons require identical ordered cases and families")
    groups: dict[str, list[float]] = defaultdict(list)
    for left, right in zip(base, candidate, strict=True):
        groups[left["family_id"]].append(right["span_recall"] - left["span_recall"])
    deltas = [math.fsum(groups[family]) / len(groups[family]) for family in sorted(groups)]
    if not deltas:
        raise ValueError("paired comparison requires families")
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples))
    return [means[int((samples - 1) * 0.025)], means[int((samples - 1) * 0.975)]]


def evaluate(index: CodeIntelligenceIndex, cases: list[SpanCase], plan: SpanPlan) -> dict:
    validate_cases(cases)
    validate_targets(index, cases)
    # Labels/queries are hashed, not included in the report or passed into retrieval.
    protocol = {
        "dataset_fingerprint": fingerprint([case.model_dump(mode="json") for case in cases]),
        "index_fingerprint": index_fingerprint(index), "plan": plan.model_dump(mode="json"),
        "embedding_backend": index.embedder.name if index.embedder else "none",
        "reranker_backend": index.reranker.name if index.reranker else "none",
        "fusion_backend": index.fusion_scorer.name if index.fusion_scorer else "fixed-weight-v1",
        "parser_backends": dict(sorted(index.stats.parser_backends.items())),
    }
    protocol_fingerprint = fingerprint(protocol)
    comparisons = []
    primary = None
    for budget in plan.token_budgets:
        rows = {}
        for name, strategy in (("baseline", plan.baseline), ("candidate", plan.candidate)):
            packing = plan.baseline_packing if name == "baseline" else plan.candidate_packing
            rows[name] = []
            for case in cases:
                pack = index.select(case.query, top_k=plan.top_k, max_tokens=budget, strategy=strategy,
                                    packing_policy=packing)
                if pack.receipt.strategy != strategy:
                    raise ValueError("context receipt belongs to a different retrieval strategy")
                if pack.receipt.packing_policy != packing:
                    raise ValueError("context receipt belongs to a different packing policy")
                rows[name].append(score_pack(index, case, pack))
        baseline, candidate = family_aggregate(rows["baseline"]), family_aggregate(rows["candidate"])
        slices = {}
        for variant in sorted({case.variant for case in cases}):
            selected = {case.id for case in cases if case.variant == variant}
            left = [row for row in rows["baseline"] if row["case_id"] in selected]
            right = [row for row in rows["candidate"] if row["case_id"] in selected]
            slices[variant] = {
                "families": len({row["family_id"] for row in left}),
                "baseline": family_aggregate(left), "candidate": family_aggregate(right),
            }
        point = {
            "token_budget": budget, "baseline": baseline, "candidate": candidate,
            "span_recall_delta": candidate["span_recall"] - baseline["span_recall"],
            "paired_family_bootstrap_95_ci": paired_family_interval(
                rows["baseline"], rows["candidate"], seed=protocol_fingerprint + str(budget),
                samples=plan.bootstrap_samples,
            ),
            "slices": slices, "outcomes": rows,
        }
        comparisons.append(point)
        if budget == plan.primary_budget:
            primary = point
    assert primary is not None
    reasons = []
    families = len({case.family_id for case in cases})
    if families < plan.minimum_families:
        reasons.append("too few independent labelled task families")
    if primary["candidate"]["span_recall"] < plan.minimum_span_recall:
        reasons.append("primary-budget complete-span recall is below the fixed floor")
    if primary["paired_family_bootstrap_95_ci"][0] < 0:
        reasons.append("primary-budget paired family interval permits regression")
    for variant, values in primary["slices"].items():
        if values["families"] < 2:
            reasons.append(f"insufficient family support for primary slice: {variant}")
        elif values["candidate"]["span_recall"] + plan.maximum_slice_regression < values["baseline"]["span_recall"]:
            reasons.append(f"primary-budget query slice regressed: {variant}")
    if any(row["estimated_tokens"] > point["token_budget"]
           for point in comparisons for rows in point["outcomes"].values() for row in rows):
        reasons.append("a context pack exceeded its approximate token budget")
    quality_gate_passed = not reasons
    if any(case.review_status != "human_reviewed" for case in cases):
        reasons.append("synthetic or unreviewed labels cannot establish owner-review readiness")
    report = {
        "schema_version": "1.0", "protocol": protocol, "protocol_fingerprint": protocol_fingerprint,
        "cases": len(cases), "families": families, "comparisons": comparisons,
        "quality_gate_passed": quality_gate_passed,
        "decision": "held" if reasons else "ready_for_owner_review", "reasons": reasons,
        "production_activation": False,
        "limitations": [
            "File recall is not evidence-span coverage or downstream patch success.",
            "Intervals resample task families, not independent repositories; small-family bootstrap is approximate.",
            "Line labels and reviewer provenance are owner-supplied, not externally authenticated.",
            "Tokens use the context pack's character-based estimate, not a model tokenizer.",
            "No test partition is consumed here; use a fresh preregistered family set for a real experiment.",
        ],
    }
    report["report_fingerprint"] = fingerprint(report)
    return report


def verify_report(report: dict) -> bool:
    try:
        return report["report_fingerprint"] == fingerprint({key: value for key, value in report.items() if key != "report_fingerprint"})
    except (KeyError, TypeError, ValueError):
        return False


def markdown_report(report: dict) -> str:
    primary_budget = report["protocol"]["plan"]["primary_budget"]
    primary = next(point for point in report["comparisons"] if point["token_budget"] == primary_budget)
    lines = ["# Evidence-span retrieval comparison", "", f"Decision: {report['decision']}", "",
             f"{report['cases']} queries in {report['families']} task families. Fixed primary budget: {primary_budget} approximate tokens.", "",
             f"At that budget, candidate file recall is {primary['candidate']['file_recall']:.1%}, "
             f"but complete-span recall is {primary['candidate']['span_recall']:.1%}.", "",
             "| Approximate token budget | Baseline span recall | Candidate span recall | Paired family 95% interval |",
             "|---:|---:|---:|---:|"]
    for point in report["comparisons"]:
        low, high = point["paired_family_bootstrap_95_ci"]
        lines.append(f"| {point['token_budget']} | {point['baseline']['span_recall']:.3f} | "
                     f"{point['candidate']['span_recall']:.3f} | [{low:+.3f}, {high:+.3f}] |")
    failures = [row for row in primary["outcomes"]["candidate"] if row["failure"] != "none"]
    if failures:
        lines.extend(["", "## Primary-budget evidence gaps (first 20)", "",
                      "| Case | Failure | Missing source spans |", "|---|---|---|"])
        for row in failures[:20]:
            spans = ", ".join(f"{span['path']}:{span['start_line']}-{span['end_line']}" for span in row["missing_spans"])
            case_id, spans = (value.replace("|", "\\|").replace("\n", " ") for value in (row["case_id"], spans))
            lines.append(f"| {case_id} | {row['failure']} | {spans} |")
    lines.extend(["", "## Review notes", ""] + [f"- {reason}" for reason in report["reasons"]])
    lines.extend(["", "No model was trained, activated, or deployed. JSON includes per-case compression failures and slices.",
                  "Reports contain no raw queries or source snippets. Treat paths and case IDs as potentially private metadata.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--plan", type=Path, help="Fixed comparison policy JSON, supplied before inspecting outcomes")
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/retrieval-spans/report.json"))
    parser.add_argument("--require-gate", action="store_true", help="Fail if not ready for owner review; synthetic defaults remain held")
    args = parser.parse_args(argv)
    try:
        plan = SpanPlan.model_validate_json(args.plan.read_text(encoding="utf-8")) if args.plan else SpanPlan()
        cases = load_cases(args.dataset)
        target = args.output.resolve()
        card = target.with_suffix(".md")
        root = args.repository_root.resolve()
        if (target == card or any(path in {args.dataset.resolve(), args.plan.resolve() if args.plan else None}
                                  or (path.is_relative_to(root)
                                      and path.relative_to(root).parts[:2] != ("data", "evaluations"))
                                  for path in (target, card))):
            raise ValueError("outputs must not overwrite inputs or enter the indexed corpus")
        if target.exists() or card.exists():
            raise ValueError("use a fresh output path; comparison evidence is not overwritten")
        index = CodeIntelligenceIndex.build(DirectoryWorkspace(root), config=RetrievalConfig(prefer_tree_sitter=False))
        report = evaluate(index, cases, plan)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with card.open("x", encoding="utf-8") as handle:
            handle.write(markdown_report(report))
        print(json.dumps({"decision": report["decision"], "families": report["families"],
                          "report_fingerprint": report["report_fingerprint"], "reasons": report["reasons"]}))
        if report["decision"] == "held":
            print("::notice title=Evidence-span comparison held::Review the evidence report; this check does not approve or activate a model.")
        return 1 if args.require_gate and report["decision"] != "ready_for_owner_review" else 0
    except (OSError, ValueError):
        print("Span comparison failed: invalid inputs, stale labels, or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
