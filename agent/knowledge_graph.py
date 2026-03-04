import re
from itertools import combinations
from typing import Any

import networkx as nx

from agent.graph_rag import search_graph_knowledge


_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_SOURCE_EQ_RE = re.compile(r"source=([^,\)\s]+)")
_SOURCE_PAREN_RE = re.compile(r"\(.*?source=([^\)\s]+).*?\)")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "while",
    "where",
    "what",
    "when",
    "who",
    "why",
    "how",
    "about",
    "using",
    "used",
    "agent",
    "agents",
    "source",
    "page",
    "score",
    "fused",
    "vector",
    "local",
    "knowledge",
    "retrieval",
}


def _parse_rag_lines(rag_notes: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw in (rag_notes or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue

        source = "unknown"
        match = _SOURCE_EQ_RE.search(line) or _SOURCE_PAREN_RE.search(line)
        if match:
            source = match.group(1).strip()

        snippet = line[2:].strip()
        if ": " in snippet:
            snippet = snippet.split(": ", 1)[1].strip()
        if not snippet:
            continue
        records.append({"source": source, "snippet": snippet})
    return records


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for token in _ENTITY_TOKEN_RE.findall((text or "").lower()):
        if len(token) < 3:
            continue
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            entities.append(token)
    return entities


def _build_graph(records: list[dict[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    for record in records:
        source = record.get("source", "unknown")
        entities = _extract_entities(record.get("snippet", ""))
        if len(entities) < 2:
            continue

        # Cap per-record entities to keep graph sparse and predictable.
        entities = entities[:8]
        for entity in entities:
            if not graph.has_node(entity):
                graph.add_node(entity, frequency=0)
            graph.nodes[entity]["frequency"] = int(graph.nodes[entity].get("frequency", 0)) + 1

        for left, right in combinations(entities, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] = int(graph[left][right].get("weight", 0)) + 1
                graph[left][right]["sources"].add(source)
            else:
                graph.add_edge(left, right, weight=1, sources={source})
    return graph


def _select_relationships(graph: nx.Graph, query: str, limit: int) -> list[tuple[str, str, int, list[str]]]:
    if graph.number_of_edges() == 0:
        return []

    query_entities = [e for e in _extract_entities(query) if graph.has_node(e)]
    ranked: list[tuple[str, str, int, list[str]]] = []

    for left, right, data in graph.edges(data=True):
        weight = int(data.get("weight", 0))
        if weight <= 0:
            continue
        sources = sorted(list(data.get("sources", set())))
        if query_entities and left not in query_entities and right not in query_entities:
            continue
        ranked.append((left, right, weight, sources))

    if not ranked:
        for left, right, data in graph.edges(data=True):
            weight = int(data.get("weight", 0))
            if weight <= 0:
                continue
            sources = sorted(list(data.get("sources", set())))
            ranked.append((left, right, weight, sources))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[: max(1, limit)]


def query_knowledge_graph(
    query: str,
    limit: int = 4,
    return_meta: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    rag_result = search_graph_knowledge(query, limit=max(5, limit + 2), return_meta=True)
    rag_notes = ""
    rag_meta: dict[str, Any] = {}
    if isinstance(rag_result, tuple):
        rag_notes, rag_meta = rag_result
    else:
        rag_notes = str(rag_result or "")

    clean_notes = (rag_notes or "").strip()
    if not clean_notes:
        text = "Knowledge graph could not run because no local evidence was available."
        return (text, {"source": "kg_empty"}) if return_meta else text

    records = _parse_rag_lines(clean_notes)
    if not records:
        text = "Knowledge graph found no structured local relationship candidates for this query."
        meta = {"source": "kg_no_records", "rag_meta": rag_meta}
        return (text, meta) if return_meta else text

    graph = _build_graph(records)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        text = "Knowledge graph found entities but no strong relationship edges in local evidence."
        meta = {"source": "kg_no_edges", "rag_meta": rag_meta}
        return (text, meta) if return_meta else text

    relations = _select_relationships(graph, query=query, limit=limit)
    if not relations:
        text = "Knowledge graph did not find direct relationship edges for the requested entities."
        meta = {"source": "kg_no_relations", "rag_meta": rag_meta}
        return (text, meta) if return_meta else text

    top_entities = sorted(
        graph.nodes(data=True),
        key=lambda item: int(item[1].get("frequency", 0)),
        reverse=True,
    )[:6]
    entity_line = ", ".join([name for name, _ in top_entities]) or "none"

    lines = [
        f"Detected graph entities: {entity_line}.",
        "Relationship candidates from local evidence:",
    ]
    for left, right, weight, sources in relations:
        source_text = ", ".join(sources[:2]) if sources else "unknown"
        lines.append(f"- {left} <-> {right} (co-occurrence={weight}, source={source_text})")

    text = "\n".join(lines).strip()
    meta = {
        "source": "kg_live",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "rag_meta": rag_meta,
    }
    return (text, meta) if return_meta else text
