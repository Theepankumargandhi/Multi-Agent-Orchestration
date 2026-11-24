"""Live answer evaluation: ask the real agent 50 questions and score what it answers.

Unlike the offline gates, this runs the whole graph against the real model provider, so it
measures answers, not just routing or policy mechanics. Each run builds a temporary document
store with this repository's docs and one uploaded report (a fictional organisation, so no model
can know the answers from training), then asks every question in evals/datasets/answer_eval.jsonl.

    python -m evals.answer_evaluation --output data/evaluations/answers/latest.json

It reads GROQ_API_KEY from the environment or .env and makes about 150 model calls; --delay
spaces the questions out to stay inside Groq's free-tier rate limits.

Scoring is deterministic and conservative, so it can be read and argued with:
- answer: every expected fact (a regex group) appears, and nothing was refused or paused;
- abstain: the agent said it lacks the evidence (grounding abstention or a plain admission);
- correct_or_abstain: a false premise is corrected or the agent declines;
- blocked: the safety check refused the request.
Each citation is checked against the store: a cited source and page that does not exist counts
as fabricated.
"""

import argparse
import asyncio
import importlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "answer_eval.jsonl"
REPORT_FIXTURE = ROOT / "evals" / "fixtures" / "answer_eval" / "tidewater_annual_report.json"
OWNER = "answer-eval"
CATEGORY_ORDER = ["project_docs", "uploaded_document", "unanswerable", "false_premise", "calculator", "safety"]
CATEGORY_LABELS = {
    "project_docs": "Project docs",
    "uploaded_document": "Uploaded report",
    "unanswerable": "Not in the documents",
    "false_premise": "False premise",
    "calculator": "Calculator",
    "safety": "Safety",
}

# Same marker forms as the UI's citation chips: 【source=report.pdf, page=3】 or 【report.pdf, page 3】.
_MARKER_RE = re.compile(r"【([^】\n]{1,300})】")
_CITATION_RE = re.compile(r"^\s*(?:source\s*=\s*)?([^\s,;]+)\s*,\s*(?:page|p\.)\s*[=:]?\s*(\d{1,5})\s*$", re.I)
_ABSTAIN_RE = re.compile(
    r"(?:don't|do not) have enough (?:verified )?(?:evidence|information)"
    r"|\bnot (?:found|mentioned|stated|specified|included|provided|available|covered|disclosed|listed|given)\b"
    r"|\bno (?:information|mention|details|data|figures?|evidence|record)\b"
    r"|(?:doesn't|does not|don't|do not) (?:say|mention|specify|state|provide|include|contain|list|give|disclose|report)"
    r"|(?:isn't|is not|aren't|are not) (?:mentioned|specified|stated|provided|available|included|disclosed|listed)"
    r"|(?:couldn't|could not|cannot|can't|unable to) (?:find|determine|locate|answer|confirm|verify)"
    r"|no (?:relevant )?documents"
)


def configure_environment(store_dir: Path) -> None:
    """Point the agent at a temporary store; the agent reads these settings when imported."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        {
            "RAG_EMBEDDING_BACKEND": "local",
            "USE_CHROMA_RAG": "true",
            "CHROMA_PERSIST_DIR": str(store_dir / "chroma_db"),
            "LOCAL_RAG_DB_PATH": str(store_dir / "local_rag.db"),
            "GRAPH_RAG_ENABLED": "false",
            "GROUNDING_VERIFICATION_ENABLED": "true",
            "CACHE_USE_REDIS": "false",
            "AGENT_MEMORY_ENABLED": "false",
            "MCP_TOOLS_ENABLED": "false",
            "LANGSMITH_TRACING": "false",
            "ANONYMIZED_TELEMETRY": "false",
        }
    )


def _pdf(pages: list[list[str]]) -> bytes:
    """A small valid PDF with one text line per list item."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    count = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    font = 3 + 2 * count
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()]
    for i, lines in enumerate(pages):
        body = " T* ".join(f"({escape(line)}) Tj" for line in lines)
        stream = f"BT /F1 11 Tf 16 TL 60 740 Td {body} ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def build_store() -> dict[str, Any]:
    from agent import local_rag

    os.chdir(ROOT)  # sources are cited by repository-relative path, as after scripts/ingestion
    docs = ["README.md", *sorted(path.as_posix() for path in Path("docs").rglob("*.md"))]
    shared = local_rag.ingest_markdown_to_chroma(docs)
    if not shared.get("ok"):
        raise RuntimeError(f"Could not index the project docs: {shared.get('message')}")
    report = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))
    upload = local_rag.ingest_upload(OWNER, report["file_name"], _pdf(report["pages"]))
    return {"docs": len(docs), "doc_chunks": shared.get("chunk_count"), "upload": upload["file_name"],
            "upload_pages": upload["pages"]}


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 2,340 -> 2340
    return re.sub(r"\s+", " ", text).lower()


def citations(text: str) -> list[tuple[str, int]]:
    found = []
    for marker in _MARKER_RE.findall(text):
        for part in marker.split(";"):
            match = _CITATION_RE.match(part)
            if match:
                found.append((match.group(1), int(match.group(2))))
    return found


def score(case: dict, outcome: dict) -> dict:
    from agent import local_rag

    text = normalize(outcome.get("response", ""))
    facts_ok = all(any(re.search(pattern, text) for pattern in group) for group in case["facts"])
    forbidden = any(re.search(pattern, text) for pattern in case["forbid"])
    abstained = outcome.get("grounding_action") == "abstain" or bool(_ABSTAIN_RE.search(text))
    blocked = bool(outcome.get("safety_blocked"))
    paused = bool(outcome.get("paused_for_web_approval"))
    expect = case["expect"]
    if outcome.get("error"):
        passed = False
    elif expect == "answer":
        passed = facts_ok and not forbidden and not blocked and not paused
    elif expect == "abstain":
        passed = abstained and not blocked and not paused
    elif expect == "correct_or_abstain":
        passed = (facts_ok or abstained) and not forbidden and not blocked and not paused
    elif expect == "blocked":
        passed = blocked
    else:
        raise ValueError(f"Unknown expectation {expect!r} in {case['id']}")

    cited = []
    for source, page in citations(outcome.get("response", "")):
        exists = local_rag.find_passage(source, page, OWNER) is not None
        cited.append({"source": source, "page": page, "exists": exists})
    expected_sources = case["sources"]
    return {
        "passed": passed,
        "facts_found": facts_ok,
        "abstained": abstained,
        "blocked": blocked,
        "paused_for_web_approval": paused,
        "citations": cited,
        "cites_expected_source": bool(expected_sources)
        and any(any(item["source"].endswith(expected) for expected in expected_sources) for item in cited),
    }


async def ask(graph, model: str, question: str) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "model": model, "user_id": OWNER}}
    started = time.perf_counter()
    await graph.ainvoke({"messages": [("user", question)]}, config)
    seconds = time.perf_counter() - started
    snapshot = await graph.aget_state(config)
    values = snapshot.values
    return {
        "response": values.get("final_response") or "",
        "route": values.get("route", ""),
        "grounding_action": values.get("grounding_action", ""),
        "safety_blocked": bool(values.get("safety_blocked")),
        "paused_for_web_approval": bool(snapshot.next),
        "seconds": round(seconds, 2),
    }


async def run(cases: list[dict], model: str | None, delay: float) -> tuple[list[dict], str]:
    from langgraph.checkpoint.memory import MemorySaver

    # agent/__init__.py exports the compiled graph under the module's own name.
    research = importlib.import_module("agent.research_assistant")

    model = model or research.GROQ_CHAT_MODEL
    graph = research.build_research_assistant(checkpointer=MemorySaver())
    results = []
    for index, case in enumerate(cases, 1):
        outcome: dict[str, Any] = {}
        for attempt in range(2):
            try:
                outcome = await ask(graph, model, case["question"])
                break
            except Exception as error:  # provider errors are recorded, not hidden
                outcome = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
                if attempt == 0:
                    await asyncio.sleep(30)
        result = {**case, **outcome, **score(case, outcome)}
        results.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{index:>2}/{len(cases)}] {mark} {case['id']:<16} {outcome.get('seconds', '-')}s "
              f"route={outcome.get('route', '-')}", flush=True)
        if index < len(cases):
            await asyncio.sleep(delay)
    return results, model


def summarize(results: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    document_answers = [
        r for r in results
        if r["category"] in {"project_docs", "uploaded_document"} and not r["abstained"] and not r.get("error")
    ]
    all_citations = [c for r in results for c in r["citations"]]
    latencies = sorted(r["seconds"] for r in results if "seconds" in r)
    p95 = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))] if latencies else None
    return {
        "questions": len(results),
        "passed": sum(r["passed"] for r in results),
        "categories": {
            category: {"questions": len(items), "passed": sum(r["passed"] for r in items)}
            for category, items in sorted(by_category.items(), key=lambda kv: CATEGORY_ORDER.index(kv[0]))
        },
        "document_answers": len(document_answers),
        "document_answers_with_citation": sum(bool(r["citations"]) for r in document_answers),
        "document_answers_citing_expected_source": sum(r["cites_expected_source"] for r in document_answers),
        "citations": len(all_citations),
        "fabricated_citations": sum(not c["exists"] for c in all_citations),
        "paused_for_web_approval": sum(r["paused_for_web_approval"] for r in results),
        "errors": sum(bool(r.get("error")) for r in results),
        "latency_seconds": {
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": p95,
        },
    }


def markdown(summary: dict, results: list[dict], model: str) -> str:
    lines = [
        f"Model: `{model}` · {summary['questions']} questions · "
        f"**{summary['passed']} passed ({summary['passed'] / summary['questions']:.0%})**",
        "",
        "| Category | Passed |",
        "|---|---|",
    ]
    for category, counts in summary["categories"].items():
        lines.append(f"| {CATEGORY_LABELS[category]} | {counts['passed']} / {counts['questions']} |")
    answers = summary["document_answers"]
    lines += [
        "",
        f"- Document answers with at least one citation: {summary['document_answers_with_citation']} / {answers}",
        f"- Document answers citing the expected file: {summary['document_answers_citing_expected_source']} / {answers}",
        f"- Citations checked: {summary['citations']}; fabricated (no such file or page): "
        f"{summary['fabricated_citations']}",
        f"- Paused for web-search approval: {summary['paused_for_web_approval']}; errors: {summary['errors']}",
        f"- Latency: median {summary['latency_seconds']['median']} s, p95 {summary['latency_seconds']['p95']} s",
        "",
        "Failures:",
        "",
    ]
    for result in results:
        if not result["passed"]:
            answer = (result.get("error") or result.get("response", "")).replace("\n", " ")[:220]
            lines.append(f"- `{result['id']}` (route `{result.get('route', '-')}`): {result['question']}  \n"
                         f"  → {answer}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/evaluations/answers/latest.json")
    parser.add_argument("--markdown", help="Also write the summary as Markdown to this path.")
    parser.add_argument("--only", help="Comma-separated question ids or categories to run.")
    parser.add_argument("--model", help="Chat model id (default: the configured Groq chat model).")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between questions.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        cases = [case for case in cases if case["id"] in wanted or case["category"] in wanted]

    with tempfile.TemporaryDirectory(prefix="citeright-answer-eval-", ignore_cleanup_errors=True) as store_dir:
        configure_environment(Path(store_dir))
        if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
            sys.exit("Set GROQ_API_KEY (or OPENAI_API_KEY) in the environment or .env.")
        store = build_store()
        print(f"Indexed {store['docs']} docs ({store['doc_chunks']} chunks) and uploaded "
              f"{store['upload']} ({store['upload_pages']} pages).", flush=True)
        results, model = asyncio.run(run(cases, args.model, args.delay))
        summary = summarize(results)
        from agent import local_rag

        local_rag._reset_chroma_store_cache()  # release the files before the folder is removed

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": model, "summary": summary, "results": results}, indent=2), encoding="utf-8")
    report = markdown(summary, results, model)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
