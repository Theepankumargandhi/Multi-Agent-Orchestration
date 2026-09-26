import re
from typing import Any

import networkx as nx

from agent.graph_rag import search_graph_knowledge

_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_SOURCE_EQ_RE = re.compile(r"source=([^,\)\s]+)")
_SOURCE_PAREN_RE = re.compile(r"\(.*?source=([^\)\s]+).*?\)")
_RELATION_RE = re.compile(
    r"(?P<subject>[A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*)?)\s+"
    r"(?P<predicate>depends on|works with|consumes context from|routes|calls|uses|reads|writes|"
    r"exposes|integrates|links|notifies|combines|includes|runs)\s+"
    r"(?P<object>[^.;]+)",
    flags=re.IGNORECASE,
)
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


def _normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip(" ,:-")).lower()


def _extract_triples(text: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for match in _RELATION_RE.finditer(text or ""):
        subject = _normalize_entity(match.group("subject"))
        predicate = re.sub(r"\s+", "_", match.group("predicate").strip().lower())
        raw_objects = re.sub(r"\b(?:the|a|an)\b", "", match.group("object"), flags=re.IGNORECASE)
        for raw_object in re.split(r"\s*(?:,|\band\b)\s*", raw_objects):
            obj = _normalize_entity(raw_object)
            if subject and obj and subject != obj and len(obj) <= 80:
                triples.append((subject, predicate, obj))
    return triples


def _build_graph(records: list[dict[str, str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for record in records:
        source = record.get("source", "unknown")
        triples = _extract_triples(record.get("snippet", ""))
        for subject, predicate, obj in triples:
            for entity in (subject, obj):
                if not graph.has_node(entity):
                    graph.add_node(entity, frequency=0)
                graph.nodes[entity]["frequency"] = int(graph.nodes[entity].get("frequency", 0)) + 1
            existing = graph.get_edge_data(subject, obj, key=predicate)
            if existing:
                existing["weight"] = int(existing.get("weight", 0)) + 1
                existing["sources"].add(source)
            else:
                graph.add_edge(subject, obj, key=predicate, predicate=predicate, weight=1, sources={source})
    return graph


def _select_relationships(
    graph: nx.MultiDiGraph,
    query: str,
    limit: int,
) -> list[tuple[str, str, str, int, list[str]]]:
    if graph.number_of_edges() == 0:
        return []

    query_text = (query or "").lower()
    query_entities = [node for node in graph.nodes if node in query_text]
    ranked: list[tuple[str, str, str, int, list[str]]] = []

    if len(query_entities) >= 2:
        undirected = graph.to_undirected()
        try:
            path = nx.shortest_path(undirected, query_entities[0], query_entities[1])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
        for left, right in zip(path, path[1:], strict=False):
            edge_map = graph.get_edge_data(left, right) or graph.get_edge_data(right, left) or {}
            for predicate, data in edge_map.items():
                ranked.append(
                    (left, right, str(data.get("predicate") or predicate), int(data.get("weight", 1)), sorted(data.get("sources", set())))
                )

    for left, right, predicate, data in graph.edges(keys=True, data=True):
        weight = int(data.get("weight", 0))
        if weight <= 0:
            continue
        sources = sorted(list(data.get("sources", set())))
        if query_entities and left not in query_entities and right not in query_entities:
            continue
        candidate = (left, right, str(data.get("predicate") or predicate), weight, sources)
        if candidate not in ranked:
            ranked.append(candidate)

    if not ranked:
        for left, right, predicate, data in graph.edges(keys=True, data=True):
            weight = int(data.get("weight", 0))
            if weight <= 0:
                continue
            sources = sorted(list(data.get("sources", set())))
            ranked.append((left, right, str(data.get("predicate") or predicate), weight, sources))

    ranked.sort(key=lambda item: item[3], reverse=True)
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
    for left, right, predicate, weight, sources in relations:
        source_text = ", ".join(sources[:2]) if sources else "unknown"
        lines.append(f"- {left} --{predicate}--> {right} (evidence_count={weight}, source={source_text})")

    text = "\n".join(lines).strip()
    meta = {
        "source": "kg_live",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "rag_meta": rag_meta,
    }
    return (text, meta) if return_meta else text
