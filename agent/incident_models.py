"""Consent-bound reproduction bundles and preregistered one-factor experiments."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Arm = Literal["baseline", "retrieval", "reranking", "packing", "generation", "candidate"]
ARMS = ("baseline", "retrieval", "reranking", "packing", "generation")
COMPONENTS = {"retrieval": "retrieval", "reranking": "reranking", "packing": "packing", "generation": "generation"}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def runtime_fingerprint() -> str:
    root = Path(__file__).parent.parent
    files = ("agent/incident_models.py", "agent/incident_pipeline.py", "agent/incident_investigation.py",
             "agent/local_rag.py", "agent/research_assistant.py", "agent/incident_store.py", "agent/incident_runtime.py", "agent/incident_worker.py",
             "agent/incident_comparison.py", "code_agent/providers.py", "agentforge_sdk/_retrieval.py")
    versions = {}
    for name in ("pydantic", "cryptography", "langchain-core", "langchain-groq", "langchain-openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return fingerprint({"files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}, "versions": versions})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class IncidentDocument(StrictModel):
    document_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    text: str = Field(min_length=1, max_length=8000)
    version_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(default="", max_length=500)
    source: str = Field(default="", max_length=1000)
    file_name: str = Field(default="", max_length=1000)
    page: str = Field(default="n/a", max_length=100)

    @model_validator(mode="after")
    def content_matches_version(self):
        if hashlib.sha256(self.text.encode()).hexdigest() != self.version_sha256:
            raise ValueError("document bytes do not match the declared version")
        return self


class IncidentPipeline(StrictModel):
    retrieval: Literal["captured", "lexical", "bm25", "hybrid_rrf"] = "captured"
    reranking: Literal["captured", "lexical", "production_heuristic"] = "captured"
    packing: Literal["prefix", "query_window", "production"] = "prefix"
    generation: Literal["captured", "evidence_only"] = "captured"
    top_k: int = Field(default=2, ge=1, le=10)
    context_chars: int = Field(default=2000, ge=32, le=6000)


class CapturedRank(StrictModel):
    document_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    score: float


class RuntimeRagSnapshot(StrictModel):
    protocol: Literal["local-rag-runtime-v1"] = "local-rag-runtime-v1"
    backend: Literal["chroma_hybrid", "sqlite_overlap"]
    corpus_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    vector_ranking: list[CapturedRank] = Field(default_factory=list, max_length=20)
    lexical_ranking: list[CapturedRank] = Field(max_length=20)
    fused_ranking: list[CapturedRank] = Field(max_length=20)
    reranker: str = Field(max_length=100)
    embedding_model: str | None = Field(default=None, max_length=100)
    collection: str | None = Field(default=None, max_length=200)
    vector_score_mode: Literal["relevance", "distance", "not_used"] = "not_used"
    rerank_input_tokens: int | None = Field(default=None, ge=0)
    rerank_output_tokens: int | None = Field(default=None, ge=0)
    rrf_k: int = Field(ge=1, le=10000)
    bm25_top_k: int = Field(ge=1, le=10000)
    candidate_top_k: int = Field(ge=1, le=10000)
    packed_context: str = Field(min_length=1, max_length=6000)
    generation_input: str = Field(default="", max_length=12000)
    generation_answer: str = Field(default="", max_length=4000)
    temperature: float = Field(default=0.2, ge=0, le=2)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    capture_call_id: str = Field(default="", max_length=100)


class IncidentBundle(StrictModel):
    origin: Literal["runtime", "operator", "synthetic"]
    family: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=2000)
    observed_answer: str = Field(min_length=1, max_length=4000)
    documents: list[IncidentDocument] = Field(min_length=1, max_length=20)
    retrieved_ids: list[str] = Field(max_length=20)
    ranked_ids: list[str] = Field(max_length=20)
    pipeline: IncidentPipeline = Field(default_factory=IncidentPipeline)
    model: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=2000)
    application_revision: str = Field(min_length=1, max_length=100)
    runtime_rag: RuntimeRagSnapshot | None = None

    @model_validator(mode="after")
    def bounded_snapshot(self):
        ids = [doc.document_id for doc in self.documents]
        if len(ids) != len(set(ids)) or len(self.retrieved_ids) != len(set(self.retrieved_ids)) or len(self.ranked_ids) != len(set(self.ranked_ids)):
            raise ValueError("snapshot identifiers must be unique")
        if not set(self.retrieved_ids) <= set(ids) or not set(self.ranked_ids) <= set(self.retrieved_ids):
            raise ValueError("ranked documents must belong to the captured retrieval pool")
        if sum(len(doc.text.encode()) for doc in self.documents) > 50000:
            raise ValueError("snapshot exceeds document byte budget")
        if self.runtime_rag:
            snapshot = self.runtime_rag
            for ranking in (snapshot.vector_ranking, snapshot.lexical_ranking, snapshot.fused_ranking):
                ranked = [row.document_id for row in ranking]
                if len(ranked) != len(set(ranked)) or not set(ranked) <= set(ids):
                    raise ValueError("runtime ranks must reference unique frozen documents")
            if snapshot.corpus_fingerprint != fingerprint([doc.model_dump() for doc in self.documents]):
                raise ValueError("runtime corpus fingerprint changed")
            if snapshot.generation_input.count(snapshot.packed_context) != 1:
                raise ValueError("captured evidence must occur exactly once in the generation input")
            if self.pipeline.packing != "production" or self.pipeline.retrieval != "captured" or self.pipeline.reranking != "captured":
                raise ValueError("runtime baseline must preserve the captured production stages")
        # Reject obvious credentials instead of silently changing reproduction inputs.
        content = json.dumps(self.model_dump(), ensure_ascii=False)
        if re.search(r"\b(?:gsk_|sk-|ghp_)[A-Za-z0-9_-]{16,}|(?i:bearer)\s+[A-Za-z0-9._-]{16,}|-----BEGIN .*PRIVATE KEY-----", content):
            raise ValueError("remove credentials before consenting to a reproduction bundle")
        return self


class InvestigationPlan(StrictModel):
    protocol: Literal["incident-one-factor-v1"] = "incident-one-factor-v1"
    repeats: int = Field(default=2, ge=2, le=3)
    max_calls: int = Field(default=10, ge=1, le=15)
    max_reserved_tokens: int = Field(default=100000, ge=1000, le=200000)
    max_completion_tokens: int = Field(default=512, ge=256, le=1024)
    timeout_seconds: int = Field(default=20, ge=1, le=30)
    randomization_seed: int = Field(default=47, ge=0, le=2**31 - 1)


class AnswerDraft(StrictModel):
    answer: str = Field(min_length=1, max_length=2000)
    citations: list[str] = Field(default_factory=list, max_length=10)


class TrialReview(StrictModel):
    trial_id: str = Field(min_length=1, max_length=100)
    trial_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["correct", "incorrect", "ambiguous"]
    safe: bool
    rationale: str = Field(min_length=1, max_length=1000)
    abstained: bool | None = None


def intervention(base: IncidentPipeline, arm: str) -> IncidentPipeline:
    changes = {"retrieval": ("captured", "lexical"), "reranking": ("captured", "lexical"),
               "packing": ("prefix", "query_window"), "generation": ("captured", "evidence_only")}
    if arm == "baseline":
        return base.model_copy(deep=True)
    first, second = changes[arm]
    return base.model_copy(update={arm: second if getattr(base, arm) == first else first})


def configuration_for(data: dict, arm: str) -> IncidentPipeline:
    if arm == "candidate":
        return IncidentPipeline.model_validate(data["candidate_config"])
    base = IncidentPipeline.model_validate(data["bundle"]["pipeline"])
    if data["bundle"].get("runtime_rag") and arm != "baseline":
        changes = {"retrieval": "bm25", "reranking": "production_heuristic", "packing": "query_window", "generation": "evidence_only"}
        return base.model_copy(update={arm: changes[arm]})
    return intervention(base, arm)
