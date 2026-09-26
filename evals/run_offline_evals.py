"""Credential-free routing regression evaluation for CI and portfolio demos."""

import argparse
import asyncio
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.research_assistant import intent_router_agent


async def evaluate(dataset_path: Path) -> dict:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for row in rows:
        query = row["input"]
        decision = await intent_router_agent(
            {"messages": [HumanMessage(content=query)], "query": query},
            {"configurable": {"model": "offline-eval"}},
        )
        actual = decision["route"]
        results.append({**row, "actual_route": actual, "passed": actual == row["expected_route"]})
    passed = sum(bool(row["passed"]) for row in results)
    return {
        "metric": "routing_accuracy",
        "score": passed / len(results) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("routing_dataset.jsonl"))
    parser.add_argument("--min-score", type=float, default=0.95)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.dataset))
    print(json.dumps(report, indent=2))
    return 0 if report["score"] >= args.min_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
