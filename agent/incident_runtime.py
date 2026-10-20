"""Consented request-local capture and replay of the application's bounded local-RAG corpus."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from agent.incident_models import (
    IncidentBundle,
    IncidentDocument,
    IncidentPipeline,
    InvestigationPlan,
    RuntimeRagSnapshot,
    fingerprint,
    runtime_fingerprint,
)

_capture: ContextVar[RuntimeCapture | None] = ContextVar("private_rag_incident_capture", default=None)


def active_capture():
    return _capture.get()


def record_retrieval(payload: dict):
    capture = active_capture()
    if capture:
        capture.retrieval(payload)


def hold_capture(reason: str):
    capture = active_capture()
    if capture:
        capture.hold(reason)


@dataclass
class RuntimeCapture:
    owner: str
    family: str
    call_id: str
    store: object
    parent_id: str | None = None
    payload: dict | None = None
    system: str = ""
    generation_input: str = ""
    generation: dict | None = None
    reason: str = ""
    reranker: str = "unknown"
    reranker_usage: dict = field(default_factory=dict)
    _finished: bool = field(default=False, init=False)

    @contextmanager
    def scope(self):
        from langsmith import tracing_context
        token = _capture.set(self)
        try:
            with tracing_context(enabled=False):
                yield self
        finally:
            _capture.reset(token)

    def hold(self, reason):
        if not self.reason:
            self.reason = reason

    def retrieval(self, payload):
        if self.payload is not None:
            self.hold("multiple_retrievals_not_supported")
            return
        # Validate before retaining private content; never truncate a corpus silently.
        try:
            docs = [IncidentDocument.model_validate(doc) for doc in payload["documents"]]
            if not 1 <= len(docs) <= 20 or sum(len(doc.text.encode()) for doc in docs) > 50000:
                raise ValueError("bounded corpus required")
            snapshot = RuntimeRagSnapshot.model_validate(payload["snapshot"])
            if snapshot.corpus_fingerprint != fingerprint([doc.model_dump() for doc in docs]):
                raise ValueError("corpus mismatch")
            self.payload = payload
        except (KeyError, TypeError, ValueError):
            self.hold("snapshot_bounds_or_integrity_hold")

    def before_generation(self, system, user, state):
        if (state.get("route") != "rag" or state.get("memory_context") or state.get("evidence_quality_report")):
            self.hold("mixed_or_adjudicated_context_not_supported")
        if self.system:
            self.hold("multiple_generations_not_supported")
        self.system, self.generation_input = system, user

    def model_response(self, system, user, response, model):
        if self.system != system or self.generation_input != user:
            return  # Router/rewrite calls are not generation evidence.
        usage = getattr(response, "usage_metadata", None) or {}
        def count(name):
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else None
        self.generation = {"answer": str(response.content or "").strip(), "model": model,
                           "input_tokens": count("input_tokens"), "output_tokens": count("output_tokens")}

    def finish(self, state, answer):
        if self._finished:
            return {"status": "held", "reason": "capture_already_finalized"}
        self._finished = True
        try:
            if self.reason:
                return {"status": "held", "reason": self.reason}
            if not self.payload or not self.generation or state.get("route") != "rag":
                return {"status": "held", "reason": "no_supported_rag_generation"}
            if self.generation["answer"] != answer.strip():
                return {"status": "held", "reason": "post_generation_answer_changed"}
            snapshot = RuntimeRagSnapshot.model_validate({**self.payload["snapshot"],
                "generation_input": self.generation_input, "generation_answer": self.generation["answer"],
                "input_tokens": self.generation["input_tokens"], "output_tokens": self.generation["output_tokens"],
                "capture_call_id": self.call_id})
            bundle = IncidentBundle(origin="runtime", family=self.family, query=self.payload["query"], observed_answer=answer,
                documents=self.payload["documents"], retrieved_ids=self.payload["retrieved_ids"], ranked_ids=self.payload["ranked_ids"],
                pipeline=IncidentPipeline(packing="production", top_k=self.payload["limit"], context_chars=6000),
                model=self.generation["model"], prompt=self.system, application_revision=runtime_fingerprint(), runtime_rag=snapshot)
            identifier = self.store.create(self.owner, bundle, InvestigationPlan(), consent=True,
                ttl_seconds=int(os.getenv("INCIDENT_CAPTURE_TTL_SECONDS", "86400")), parent_id=self.parent_id)
            return {"status": "captured", "id": identifier, "bundle_fingerprint": fingerprint(bundle.model_dump()),
                    "review_required": True}
        except Exception:
            # Never expose captured payloads, paths, validation messages or credentials.
            return {"status": "held", "reason": "capture_configuration_bounds_or_storage_hold"}
        finally:
            self.payload = self.generation = None
            self.system = self.generation_input = ""


def make_capture(user_input, owner, call_id, store_factory):
    """Called at the authenticated request boundary, not from graph state or tracing config."""
    if not user_input.incident_capture_consent:
        return None
    if not owner or not user_input.incident_task_family:
        raise ValueError("runtime capture requires authenticated user mode and a preregistered family")
    if any(os.getenv(flag, "false").lower() != "true" for flag in
           ("INCIDENT_INVESTIGATION_ENABLED", "INCIDENT_RUNTIME_CAPTURE_ENABLED")):
        raise ValueError("runtime capture is disabled")
    return RuntimeCapture(owner, user_input.incident_task_family, str(call_id), store_factory(), parent_id=user_input.incident_parent_id)


def document_payload(text, *, title="", source="", file_name="", page="n/a", identifier=""):
    # IDs are opaque and deterministic; sensitive source labels stay inside the encrypted bundle.
    value = {"text": text, "title": str(title), "source": str(source), "file_name": str(file_name), "page": str(page)}
    return IncidentDocument(document_id="doc-" + hashlib.sha256((identifier + fingerprint(value)).encode()).hexdigest()[:32],
        version_sha256=hashlib.sha256(text.encode()).hexdigest(), **value).model_dump()


def runtime_context(bundle: IncidentBundle, config: IncidentPipeline):
    """Use exactly the same BM25/RRF/heuristic/format helpers as local_rag; no external reads or embedding calls."""
    from agent.local_rag import (
        _bm25_rank_documents,
        _heuristic_rerank,
        _rrf_fuse_rankings,
        format_hybrid_context,
        format_sqlite_context,
    )
    snapshot = bundle.runtime_rag
    docs = bundle.documents
    index = {doc.document_id: number for number, doc in enumerate(docs)}
    if config.retrieval == "captured":
        ranks = {index[row.document_id]: {"fused": row.score, "bm25": 0.0, "vector": 0.0} for row in snapshot.fused_ranking}
        for row in snapshot.lexical_ranking:
            if index[row.document_id] in ranks:
                ranks[index[row.document_id]]["bm25"] = row.score
        for row in snapshot.vector_ranking:
            if index[row.document_id] in ranks:
                ranks[index[row.document_id]]["vector"] = row.score
        retrieved = [index[identifier] for identifier in bundle.retrieved_ids]
    else:
        bm25 = _bm25_rank_documents(bundle.query, [doc.text for doc in docs], snapshot.bm25_top_k)
        vectors = [(index[row.document_id], row.score) for row in snapshot.vector_ranking] if config.retrieval == "hybrid_rrf" else []
        ranks = _rrf_fuse_rankings(vectors, bm25, snapshot.rrf_k)
        retrieved = sorted(ranks, key=lambda number: -ranks[number]["fused"])[:snapshot.candidate_top_k]
    candidates = []
    for number in retrieved:
        doc, scores = docs[number], ranks[number]
        text = doc.text.strip().replace("\n", " ")
        snippet = text[:257] + "..." if len(text) > 260 else text
        if config.packing == "query_window":
            from agent.incident_pipeline import words
            lines = doc.text.splitlines()
            position = max(range(len(lines)), key=lambda line: len(words(bundle.query) & words(lines[line]))) if lines else 0
            snippet = " ".join(lines[position:position + 2])[:260]
        candidates.append({"doc_idx": number, "document_id": doc.document_id, "title": doc.title,
            "source": doc.source, "file_name": doc.file_name, "page": doc.page, "snippet": snippet,
            "content": doc.text, "score": scores["fused"], "fused_score": scores["fused"],
            "vector_score": scores["vector"], "bm25_score": scores["bm25"]})
    if config.reranking == "production_heuristic":
        candidates = _heuristic_rerank(bundle.query, candidates, config.top_k)
    else:
        order = {identifier: number for number, identifier in enumerate(bundle.ranked_ids)}
        candidates.sort(key=lambda item: order.get(item["document_id"], len(order)))
        candidates = candidates[:config.top_k]
    if (config.retrieval, config.reranking, config.packing) == ("captured", "captured", "production"):
        context = snapshot.packed_context
    elif snapshot.backend == "sqlite_overlap":
        context = format_sqlite_context(candidates, windowed=config.packing == "query_window")
    else:
        context = format_hybrid_context(candidates)
    if len(context) > config.context_chars:
        raise ValueError("replayed production context exceeds frozen budget")
    return context, [item["document_id"] for item in candidates]
