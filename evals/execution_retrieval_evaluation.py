"""Authored multilingual localization controls, not coding resolution benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent.execution_retrieval import ExecutionRetriever
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.models import SandboxPolicy, ToolObservation
from code_agent.retrieval_backends import RetrievalConfig


class _Workspace:
    def __init__(self, files):
        self.files = files

    def list_files(self, limit=500):
        return sorted(self.files)[:limit]

    def read_file(self, path):
        return self.files[path]


def evaluate() -> dict:
    rows = []
    for suffix, content in [
        ("py", "def normalize(value):\n    return value\n"),
        ("ts", "export function normalize(value: number) { return value; }\n"),
        ("js", "export function normalize(value) { return value; }\n"),
        ("java", "class Core { int normalize(int value) { return value; } }\n"),
        ("go", "package service\nfunc normalize(value int) int { return value }\n"),
        ("rs", "fn normalize(value: i32) -> i32 { value }\n"),
    ]:
        path = f"service/core.{suffix}"
        files = {path: content, **{f"support/helper_{number}.{suffix}": content for number in range(10)}}
        workspace = _Workspace(files)
        index = CodeIntelligenceIndex.build(workspace, config=RetrievalConfig(prefer_tree_sitter=False))
        static = index.select("Fix the incorrect result", top_k=6, max_tokens=1024, packing_policy="balanced_v1")
        observation = ToolObservation(iteration=0, action="test", ok=False, summary="failed",
                                      output=f"{path}:2: incorrect result")
        retriever = ExecutionRetriever(SandboxPolicy(execution_retrieval_enabled=True), "implementation")
        retriever.refresh(workspace, [observation])
        receipt = retriever.receipts[-1]
        rows.append({
            "language": suffix, "expected_path": path,
            "static_target_recalled": path in {item.path for item in static.receipt.selected_files},
            "execution_target_recalled": path in receipt.selected_file_sha256,
            "execution_estimated_tokens": receipt.estimated_tokens,
            "execution_status": receipt.status, "index_sha256": receipt.index_sha256,
        })
    return {"schema_version": "1.0", "evidence_kind": "authored-localization-controls",
            "coding_resolution_measured": False, "provider_calls": 0,
            "total": len(rows), "static_file_recall": sum(row["static_target_recalled"] for row in rows) / len(rows),
            "execution_file_recall": sum(row["execution_target_recalled"] for row in rows) / len(rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate()
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if any(row["execution_status"] != "selected" or not row["execution_target_recalled"]
           or row["execution_estimated_tokens"] > 1024 for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
