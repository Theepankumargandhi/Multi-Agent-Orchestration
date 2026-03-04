import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines: list[str]) -> bytes:
    content_ops: list[str] = ["BT", "/F1 11 Tf", "50 760 Td"]
    for i, line in enumerate(lines):
        if i > 0:
            content_ops.append("0 -14 Td")
        content_ops.append(f"({_escape_pdf_text(line)}) Tj")
    content_ops.append("ET")
    content_stream = "\n".join(content_ops) + "\n"
    content_bytes = content_stream.encode("latin-1", errors="replace")

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content_bytes)} >>\nstream\n{content_stream}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1", errors="replace")

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return pdf


def _synthetic_documents() -> dict[str, list[str]]:
    return {
        "agent_architecture_relations.pdf": [
            "Project Atlas runs FastAPIService and StreamlitClient.",
            "FastAPIService calls IntentRouterAgent for every user query.",
            "IntentRouterAgent routes math queries to MathAgent.",
            "IntentRouterAgent routes relationship queries to KnowledgeGraphAgent.",
            "KnowledgeGraphAgent works with RAGAgent to build final context.",
            "RAGAgent reads ChromaStore and LocalSQLiteStore for evidence.",
            "WebSearchAgent uses MCPBridge and MCPToolServer.",
            "MCPToolServer exposes web_search and calculator tools.",
            "ConversationStore writes messages to PostgreSQL.",
            "CheckpointStore writes graph state to PostgreSQL.",
            "ResponseAgent combines WebSearchAgent, RAGAgent, and MathAgent outputs.",
            "EvaluationAgent reviews response quality after ResponseAgent.",
        ],
        "incident_response_graph.pdf": [
            "IncidentManager depends on AlertService and EvidenceStore.",
            "AlertService notifies SafetyAgent and EvaluationAgent.",
            "SafetyAgent blocks unsafe requests before ResponseAgent.",
            "ResponseAgent consumes context from WebSearchAgent and RAGAgent.",
            "EvaluationAgent reviews final_response and run metadata.",
            "RedisCache accelerates WebSearchAgent and RAGAgent.",
            "PostgreSQL stores users and conversation history per user.",
            "UserAuthService issues bearer tokens for API requests.",
            "AuthMiddleware verifies token and sets request user_id.",
            "ConversationStore links thread_id, role, and user_id.",
        ],
        "product_roadmap_links.pdf": [
            "Roadmap2026 includes FeatureMCP, FeatureKG, and FeatureLocalRAG.",
            "FeatureMCP integrates MCPBridge with MCPToolServer.",
            "FeatureKG integrates KnowledgeGraphAgent with RAGAgent.",
            "FeatureLocalRAG combines ChromaStore and BM25Reranker.",
            "FeatureUserAuth links UserID, PasswordHash, and TokenTTL.",
            "FeatureObservability links LangSmithTracing and EvaluationAgent.",
            "FeatureWebFreshness links RecencyGuardAgent and WebSearchAgent.",
            "FeatureStreaming links StreamEndpoint and TokenEvents.",
            "FeatureHybridRoute links IntentRouterAgent and QueryRewriterAgent.",
            "FeatureSafety links LlamaGuard and SafetyAgent.",
        ],
    }


def generate_synthetic_pdfs(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, lines in _synthetic_documents().items():
        file_path = out_dir / filename
        file_path.write_bytes(_build_simple_pdf(lines))
        written.append(file_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PDFs for KG/RAG testing.")
    parser.add_argument(
        "--out-dir",
        default="graph_rag_docs/synthetic_kg",
        help="Output directory for generated PDF files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    files = generate_synthetic_pdfs(out_dir)
    print(f"Generated {len(files)} synthetic PDFs in: {out_dir}")
    for file_path in files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
