"""Query planning, embedding, and reranking backends for code retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,80}")
_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,8}")
_QUOTED = re.compile(r"`([^`]{1,120})`|['\"]([^'\"]{1,120})['\"]")
_TEST_TERMS = {"test", "tests", "pytest", "spec", "regression", "coverage", "fixture", "assert"}
_DEPENDENCY_TERMS = {
    "call", "calls", "caller", "dependency", "dependencies", "depends", "import",
    "imports", "uses", "used", "graph", "neighbor", "flow",
}
_STOPWORDS = {
    "about", "after", "again", "agent", "before", "between", "could", "does", "file",
    "files", "find", "fix", "from", "have", "into", "issue", "project", "repository",
    "that", "their", "there", "these", "this", "what", "when", "where", "which", "with",
}
_SEMANTIC_EXPANSIONS = {
    "auth": ("authentication", "authorization", "token", "identity"),
    "authentication": ("auth", "login", "token", "identity"),
    "database": ("storage", "persistence", "repository", "sql"),
    "durable": ("persistent", "recovery", "lease", "retry"),
    "error": ("exception", "failure", "failed"),
    "job": ("task", "worker", "queue", "execution"),
    "retrieve": ("search", "rank", "context", "relevance"),
    "retrieval": ("search", "rank", "context", "relevance"),
    "security": ("safety", "policy", "sandbox", "secret"),
    "test": ("pytest", "spec", "assert", "regression"),
    "worker": ("job", "queue", "lease", "heartbeat"),
}
FUSION_FEATURES = ("lexical", "semantic", "graph", "rerank")


def tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for match in _WORD.findall(value):
        pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", match).replace("_", " ").split()
        tokens.extend(piece.lower() for piece in pieces if len(piece) > 1)
    return tokens


@dataclass(frozen=True)
class QueryPlan:
    raw: str
    terms: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    test_intent: bool = False
    dependency_intent: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "terms": list(self.terms),
            "symbols": list(self.symbols),
            "files": list(self.files),
            "concepts": list(self.concepts),
            "test_intent": self.test_intent,
            "dependency_intent": self.dependency_intent,
        }

    @property
    def embedding_query(self) -> str:
        facets = [self.raw]
        if self.symbols:
            facets.append("symbols " + " ".join(self.symbols))
        if self.files:
            facets.append("files " + " ".join(self.files))
        if self.concepts:
            facets.append("concepts " + " ".join(self.concepts))
        if self.test_intent:
            facets.append("tests regression validation")
        if self.dependency_intent:
            facets.append("imports calls dependencies")
        return "\n".join(facets)


def decompose_query(query: str) -> QueryPlan:
    raw_terms = tokenize(query)
    files = tuple(dict.fromkeys(_PATH.findall(query)))
    quoted = [next(group for group in match if group) for match in _QUOTED.findall(query)]
    symbol_candidates = quoted + _WORD.findall(query)
    symbols = tuple(
        dict.fromkeys(
            value
            for value in symbol_candidates
            if ("_" in value or re.search(r"[a-z][A-Z]", value)) and value not in files
        )
    )
    concepts = tuple(
        dict.fromkeys(
            term for term in raw_terms
            if term not in _STOPWORDS and term not in _TEST_TERMS and term not in _DEPENDENCY_TERMS
        )
    )[:20]
    return QueryPlan(
        raw=query,
        terms=tuple(dict.fromkeys(raw_terms)),
        symbols=symbols[:20],
        files=files[:20],
        concepts=concepts,
        test_intent=bool(set(raw_terms) & _TEST_TERMS),
        dependency_intent=bool(set(raw_terms) & _DEPENDENCY_TERMS),
    )


class Embedder(Protocol):
    name: str

    def encode_documents(self, values: list[str]) -> list[tuple[float, ...]]: ...

    def encode_query(self, value: str) -> tuple[float, ...]: ...


class HashingSemanticEmbedder:
    """Deterministic dependency-free embedding used for offline CI."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = max(64, dimensions)
        self.name = f"hashing-semantic-{self.dimensions}"

    def encode_documents(self, values: list[str]) -> list[tuple[float, ...]]:
        return [self._encode(value) for value in values]

    def encode_query(self, value: str) -> tuple[float, ...]:
        return self._encode(value, expand=True)

    def _encode(self, value: str, *, expand: bool = False) -> tuple[float, ...]:
        tokens = tokenize(value)
        features = list(tokens)
        features.extend(f"{left}:{right}" for left, right in zip(tokens, tokens[1:]))
        if expand:
            for token in tokens:
                features.extend(_SEMANTIC_EXPANSIONS.get(token, ()))
        vector = [0.0] * self.dimensions
        for feature in features[:50_000]:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[position] += sign
        norm = math.sqrt(math.fsum(item * item for item in vector)) or 1.0
        return tuple(item / norm for item in vector)


class SentenceTransformerEmbedder:
    """Learned local embedding adapter loaded only when explicitly selected."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the code-intelligence optional dependencies for learned embeddings"
            ) from exc
        self.model_id = model
        self.name = f"sentence-transformers:{model}"
        self._model = SentenceTransformer(model)

    def encode_documents(self, values: list[str]) -> list[tuple[float, ...]]:
        encode = getattr(self._model, "encode_document", None) or self._model.encode
        result = encode(values, normalize_embeddings=True, show_progress_bar=False)
        return [tuple(float(item) for item in row) for row in result]

    def encode_query(self, value: str) -> tuple[float, ...]:
        encode = getattr(self._model, "encode_query", None) or self._model.encode
        result = encode(value, normalize_embeddings=True, show_progress_bar=False)
        return tuple(float(item) for item in result)


class Reranker(Protocol):
    name: str

    def score(self, query: QueryPlan, candidates: list[tuple[str, str]]) -> list[float]: ...


class FeatureReranker:
    """Auditable offline reranker over query facets and candidate text."""

    name = "query-facet-reranker-v1"

    def score(self, query: QueryPlan, candidates: list[tuple[str, str]]) -> list[float]:
        scores = []
        query_terms = set(query.terms)
        for path, text in candidates:
            path_terms = set(tokenize(path))
            text_terms = set(tokenize(text))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            path_overlap = len(query_terms & path_terms) / max(1, len(query_terms))
            symbol_bonus = sum(symbol.lower() in text.lower() for symbol in query.symbols)
            file_bonus = sum(file.lower() in path.lower() for file in query.files)
            test_bonus = 1.0 if query.test_intent and _is_test_path(path) else 0.0
            scores.append(
                overlap + 0.75 * path_overlap + 0.35 * symbol_bonus
                + 0.5 * file_bonus + 0.3 * test_bonus
            )
        return _minmax(scores)


class CrossEncoderReranker:
    """Sentence-Transformers cross-encoder adapter loaded on explicit opt-in."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install the code-intelligence optional dependencies for cross-encoder reranking"
            ) from exc
        self.model_id = model
        self.name = f"cross-encoder:{model}"
        self._model = CrossEncoder(model)

    def score(self, query: QueryPlan, candidates: list[tuple[str, str]]) -> list[float]:
        pairs = [(query.embedding_query, f"{path}\n{text[:8000]}") for path, text in candidates]
        values = self._model.predict(pairs, show_progress_bar=False)
        return _minmax([float(item) for item in values])


@dataclass(frozen=True)
class LearnedFusionScorer:
    """Integrity-checked pairwise learning-to-rank artifact used at retrieval time."""

    weights: tuple[float, ...]
    artifact_fingerprint: str
    source: str

    @property
    def name(self) -> str:
        return f"pairwise-logistic:{self.artifact_fingerprint[:12]}"

    def score(self, features: dict[str, float]) -> float:
        logit = math.fsum(
            weight * float(features.get(feature, 0.0))
            for feature, weight in zip(FUSION_FEATURES, self.weights, strict=True)
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
        exp_logit = math.exp(max(logit, -60.0))
        return exp_logit / (1.0 + exp_logit)

    @classmethod
    def load(cls, path: Path) -> "LearnedFusionScorer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported fusion artifact schema version")
        if payload.get("model_type") != "pairwise-logistic-fusion":
            raise ValueError("unsupported fusion artifact model type")
        if tuple(payload.get("feature_names") or ()) != FUSION_FEATURES:
            raise ValueError("fusion artifact feature schema does not match this runtime")
        fingerprint = str(payload.get("artifact_fingerprint") or "")
        unsigned = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
        expected = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if not fingerprint or fingerprint != expected:
            raise ValueError("fusion artifact fingerprint is invalid")
        raw_weights = payload.get("weights") or {}
        weights = tuple(float(raw_weights[name]) for name in FUSION_FEATURES)
        if any(not math.isfinite(value) or abs(value) > 100 for value in weights):
            raise ValueError("fusion artifact contains unsafe weights")
        return cls(weights=weights, artifact_fingerprint=fingerprint, source=path.as_posix())


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_backend: str = "feature"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fusion_artifact: str = ""
    lexical_weight: float = 0.50
    semantic_weight: float = 0.30
    graph_weight: float = 0.20
    rerank_weight: float = 0.25
    candidate_multiplier: int = 4
    prefer_tree_sitter: bool = True
    fallbacks: list[str] = field(default_factory=list, compare=False)

    @classmethod
    def from_environment(cls) -> "RetrievalConfig":
        return cls(
            embedding_backend=os.getenv("CODE_CONTEXT_EMBEDDING_BACKEND", "hashing").strip().lower(),
            embedding_model=os.getenv(
                "CODE_CONTEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker_backend=os.getenv("CODE_CONTEXT_RERANKER_BACKEND", "feature").strip().lower(),
            reranker_model=os.getenv(
                "CODE_CONTEXT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            fusion_artifact=os.getenv("CODE_CONTEXT_FUSION_ARTIFACT", "").strip(),
            prefer_tree_sitter=os.getenv("CODE_CONTEXT_TREE_SITTER", "true").lower()
            not in {"0", "false", "off"},
        )


def build_embedder(config: RetrievalConfig) -> Embedder | None:
    if config.embedding_backend in {"", "none", "off"}:
        return None
    if config.embedding_backend in {"sentence-transformers", "sentence_transformers", "learned"}:
        try:
            return SentenceTransformerEmbedder(config.embedding_model)
        except (OSError, RuntimeError, ValueError) as exc:
            config.fallbacks.append(str(exc))
    return HashingSemanticEmbedder()


def build_reranker(config: RetrievalConfig) -> Reranker | None:
    if config.reranker_backend in {"", "none", "off"}:
        return None
    if config.reranker_backend in {"cross-encoder", "cross_encoder", "learned"}:
        try:
            return CrossEncoderReranker(config.reranker_model)
        except (OSError, RuntimeError, ValueError) as exc:
            config.fallbacks.append(str(exc))
    return FeatureReranker()


def build_fusion_scorer(config: RetrievalConfig) -> LearnedFusionScorer | None:
    if not config.fusion_artifact:
        return None
    try:
        return LearnedFusionScorer.load(Path(config.fusion_artifact))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        config.fallbacks.append(f"fusion artifact failure: {type(exc).__name__}: {exc}")
        return None


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return math.fsum(a * b for a, b in zip(left, right))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(item - low) / (high - low) for item in values]


def _is_test_path(path: str) -> bool:
    name = path.lower()
    return (
        "/test" in name or name.startswith("test") or name.endswith("_test.py")
        or name.endswith(".test.ts") or name.endswith(".spec.ts")
    )
