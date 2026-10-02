import asyncio
import base64
import hashlib
import hmac
import importlib
import inspect
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langsmith import Client as LangsmithClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from agent import build_research_assistant
from agent.execution_replay import digest
from agent.mcp_client import close_mcp_client
from agent.memory import (
    MemoryCandidate,
    MemoryCorrection,
    MemoryOutcome,
    get_memory_store,
)
from agent.tools import perform_web_search
from code_agent.api import (
    CODE_AGENT_ENABLED,
    CODE_AGENT_MANAGER,
)
from code_agent.api import (
    router as code_agent_router,
)
from code_agent.observability import capture_research_agent_trace
from evals.adaptive_router import CostAwareRouter
from evals.contextual_bandit import BanditArtifact, ContextualBanditPolicy
from evals.platform import ExperimentStore
from schema import (
    AuthLoginInput,
    AuthRegisterInput,
    AuthToken,
    ChatMessage,
    Feedback,
    StreamInput,
    UserInput,
    model_dump_compat,
)
from service.persistence_store import open_conversation_store

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph
else:
    CompiledGraph = Any

load_dotenv()

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")
POSTGRES_CHECKPOINT_URI = os.getenv("POSTGRES_CHECKPOINT_URI") or os.getenv("DATABASE_URL", "")
CHECKPOINT_FALLBACK_SQLITE = os.getenv("CHECKPOINT_FALLBACK_SQLITE", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
CHECKPOINT_NAMESPACE = os.getenv("CHECKPOINT_NAMESPACE", "default")
POSTGRES_STORE_URI = os.getenv("POSTGRES_STORE_URI") or POSTGRES_CHECKPOINT_URI
STORE_DB_PATH = os.getenv("STORE_DB_PATH", "store.db")
STORE_FALLBACK_SQLITE = os.getenv("STORE_FALLBACK_SQLITE", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
STORE_NAMESPACE = os.getenv("STORE_NAMESPACE", "default")
ENABLE_USER_AUTH = os.getenv("ENABLE_USER_AUTH", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
USER_AUTH_SECRET = (
    os.getenv("USER_AUTH_SECRET")
    or os.getenv("AUTH_SECRET")
    or "dev-insecure-user-auth-secret-change-me"
)
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
USER_AUTH_TOKEN_TTL_SECONDS = int(os.getenv("USER_AUTH_TOKEN_TTL_SECONDS", "86400"))
PASSWORD_HASH_ITERATIONS = int(os.getenv("PASSWORD_HASH_ITERATIONS", "210000"))
HITL_PENDING_TTL_SECONDS = max(60, int(os.getenv("HITL_PENDING_TTL_SECONDS", "3600")))
MAX_REQUEST_BYTES = max(1024, int(os.getenv("MAX_REQUEST_BYTES", "65536")))
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "60")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
EVAL_RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR", "data/evaluations"))
ENABLE_EVAL_API = os.getenv("ENABLE_EVAL_API", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADAPTIVE_MODEL_ROUTER_ENABLED = os.getenv("ADAPTIVE_MODEL_ROUTER_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADAPTIVE_MODEL_ROUTER_PATH = Path(
    os.getenv("ADAPTIVE_MODEL_ROUTER_PATH", "data/evaluations/cost_router.json")
)
ADAPTIVE_SMALL_MODEL = os.getenv("ADAPTIVE_SMALL_MODEL", "gpt-4o-mini").strip()
ADAPTIVE_STRONG_MODEL = os.getenv("ADAPTIVE_STRONG_MODEL", "").strip()
CONTEXTUAL_BANDIT_ROUTER_ENABLED = os.getenv(
    "CONTEXTUAL_BANDIT_ROUTER_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
CONTEXTUAL_BANDIT_POLICY_PATH = Path(
    os.getenv("CONTEXTUAL_BANDIT_POLICY_PATH", "data/evaluations/bandit/policy.json")
)
CONTEXTUAL_BANDIT_EPSILON = min(
    0.25, max(0.0, float(os.getenv("CONTEXTUAL_BANDIT_EPSILON", "0")))
)
MODEL_GATEWAY_ENABLED = os.getenv("MODEL_GATEWAY_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AGENT_MEMORY_ENABLED = os.getenv("AGENT_MEMORY_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")
_password_hasher = None
_adaptive_router_cache: tuple[float, CostAwareRouter] | None = None
_contextual_bandit_cache: tuple[float, ContextualBanditPolicy] | None = None

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")
logger = logging.getLogger("agent_service")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
AGENT_RUNS_TOTAL = Counter(
    "agent_runs_total", "Agent runs by route and outcome", ["route", "outcome"]
)
AGENT_NODE_DURATION_SECONDS = Histogram(
    "agent_node_duration_seconds", "Agent graph-node duration", ["node"]
)


def _rotate_incompatible_checkpoint_db(db_path: str) -> str:
    """
    Rotate legacy/invalid checkpoint db files so AsyncSqliteSaver can recreate schema.

    If the legacy file is locked (common on Windows), fall back to a fresh runtime
    db path so service startup does not fail.

    Root cause addressed:
    - Existing db has a checkpoints table without required thread_ts column.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        return db_path

    columns = set()
    incompatible = False
    try:
        with sqlite3.connect(str(db_file)) as conn:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints' LIMIT 1"
            ).fetchone()
            if not has_table:
                return db_path
            columns = {row[1] for row in conn.execute("PRAGMA table_info(checkpoints)").fetchall()}
    except sqlite3.DatabaseError:
        # Corrupt or incompatible db format: rotate and recreate.
        incompatible = True

    # LangGraph 0.x used thread_ts; current checkpoint packages use checkpoint_id.
    if not incompatible and ({"thread_ts", "checkpoint_id"} & columns):
        return db_path

    stamp = int(time.time())
    backup_base = db_file.with_name(f"{db_file.name}.legacy-{stamp}")

    try:
        os.replace(str(db_file), str(backup_base))
        for ext in ("-wal", "-shm"):
            sidecar = Path(f"{db_file}{ext}")
            if sidecar.exists():
                os.replace(str(sidecar), f"{backup_base}{ext}")
        return db_path
    except PermissionError:
        runtime_db = db_file.with_name(f"{db_file.stem}.runtime-{stamp}{db_file.suffix}")
        return str(runtime_db)


def _ensure_parent_dir(file_path: str) -> None:
    path = Path(file_path)
    parent = path.parent
    if str(parent) and str(parent) not in {".", ""}:
        parent.mkdir(parents=True, exist_ok=True)


def _resolve_sqlite_saver():
    """
    Resolve sqlite saver class from supported import paths across langgraph versions.

    Returns the saver class when found, otherwise None.
    """
    candidates = [
        ("langgraph.checkpoint.aiosqlite", "AsyncSqliteSaver"),
        ("langgraph.checkpoint.sqlite", "AsyncSqliteSaver"),
        ("langgraph.checkpoint.sqlite.aio", "AsyncSqliteSaver"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        saver_cls = getattr(module, class_name, None)
        if saver_cls is not None:
            return saver_cls
    return None


def _resolve_memory_saver():
    """
    Resolve in-memory saver class for environments without sqlite saver support.
    """
    candidates = [
        ("langgraph.checkpoint.memory", "InMemorySaver"),
        ("langgraph.checkpoint.memory", "MemorySaver"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        saver_cls = getattr(module, class_name, None)
        if saver_cls is not None:
            return saver_cls
    return None


def _resolve_postgres_saver():
    """
    Resolve postgres saver class from supported import paths across langgraph versions.

    Returns:
    - (saver_class, is_async) when found
    - None when not found
    """
    candidates = [
        ("langgraph.checkpoint.postgres", "AsyncPostgresSaver", True),
        ("langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", True),
        ("langgraph.checkpoint.postgres", "PostgresSaver", False),
        ("langgraph.checkpoint.postgres.aio", "PostgresSaver", False),
    ]
    for module_name, class_name, is_async in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        saver_cls = getattr(module, class_name, None)
        if saver_cls is not None:
            return saver_cls, is_async
    return None


def _patch_postgres_saver_signature_compat(saver) -> None:
    """
    Bridge saver method signatures across langgraph/checkpointer version skew.

    Known issue:
    - Some AsyncPostgresSaver versions require `new_versions` in aput/put,
      while older langgraph runtimes do not pass it.
    """
    # Patch async aput(config, checkpoint, metadata[, new_versions])
    aput = getattr(saver, "aput", None)
    if callable(aput):
        try:
            aput_sig = inspect.signature(aput)
            nv_param = aput_sig.parameters.get("new_versions")
            if nv_param is not None and nv_param.default is inspect._empty:
                original_aput = aput

                async def aput_compat(config, checkpoint, metadata, new_versions=None):
                    return await original_aput(
                        config,
                        checkpoint,
                        metadata,
                        new_versions or {},
                    )

                saver.aput = aput_compat
        except Exception:
            pass

    # Patch sync put(config, checkpoint, metadata[, new_versions])
    put = getattr(saver, "put", None)
    if callable(put):
        try:
            put_sig = inspect.signature(put)
            nv_param = put_sig.parameters.get("new_versions")
            if nv_param is not None and nv_param.default is inspect._empty:
                original_put = put

                def put_compat(config, checkpoint, metadata, new_versions=None):
                    return original_put(
                        config,
                        checkpoint,
                        metadata,
                        new_versions or {},
                    )

                saver.put = put_compat
        except Exception:
            pass


async def _ensure_checkpointer_schema(saver) -> None:
    """
    Initialize checkpointer tables if the saver exposes setup methods.

    Supports both async and sync saver variants across langgraph versions.
    """
    for method_name in ("setup", "asetup"):
        method = getattr(saver, method_name, None)
        if callable(method):
            result = method()
            if inspect.isawaitable(result):
                await result
            return


async def _open_checkpointer(stack: AsyncExitStack):
    """
    Open Postgres checkpointer when configured; otherwise use SQLite fallback.
    Returns (saver, backend_label).
    """
    if POSTGRES_CHECKPOINT_URI:
        resolved = _resolve_postgres_saver()
        if resolved is None:
            message = (
                "Postgres checkpointer requested, but Postgres saver import failed. "
                "Install/update deps: pip install langgraph-checkpoint-postgres psycopg[binary]"
            )
            if not CHECKPOINT_FALLBACK_SQLITE:
                raise RuntimeError(message)
            print(f"[service] {message} Falling back to SQLite checkpointer.")
        else:
            saver_cls, is_async = resolved
            try:
                saver_cm = saver_cls.from_conn_string(POSTGRES_CHECKPOINT_URI)
                if is_async:
                    saver = await stack.enter_async_context(saver_cm)
                else:
                    saver = stack.enter_context(saver_cm)
                _patch_postgres_saver_signature_compat(saver)
                await _ensure_checkpointer_schema(saver)
                return saver, "postgres"
            except Exception as e:
                message = f"Failed to open/initialize Postgres checkpointer: {e}"
                if not CHECKPOINT_FALLBACK_SQLITE:
                    raise RuntimeError(message)
                print(f"[service] {message}. Falling back to SQLite checkpointer.")

    sqlite_saver_cls = _resolve_sqlite_saver()
    if sqlite_saver_cls is not None:
        _ensure_parent_dir(CHECKPOINT_DB_PATH)
        resolved_checkpoint_db = _rotate_incompatible_checkpoint_db(CHECKPOINT_DB_PATH)
        saver = await stack.enter_async_context(
            sqlite_saver_cls.from_conn_string(resolved_checkpoint_db)
        )
        await _ensure_checkpointer_schema(saver)
        return saver, resolved_checkpoint_db

    memory_saver_cls = _resolve_memory_saver()
    if memory_saver_cls is None:
        raise RuntimeError(
            "No supported checkpoint saver is available. "
            "Install/update deps: pip install langgraph-checkpoint-sqlite aiosqlite"
        )

    print(
        "[service] SQLite checkpointer import failed. "
        "Using in-memory checkpoint saver for this process only. "
        "Install langgraph-checkpoint-sqlite for persistent conversations."
    )
    saver = memory_saver_cls()
    await _ensure_checkpointer_schema(saver)
    return saver, "memory"


class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)


def _is_serializer_compat_error(exc: Exception) -> bool:
    msg = str(exc)
    return "SerializerCompat" in msg and "dumps" in msg


def _is_checkpointer_signature_compat_error(exc: Exception) -> bool:
    msg = str(exc)
    return "new_versions" in msg and ("aput(" in msg or ".aput" in msg or "put(" in msg)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign_token(data: str) -> str:
    mac = hmac.new(USER_AUTH_SECRET.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(mac)


def _create_access_token(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + max(60, USER_AUTH_TOKEN_TTL_SECONDS),
    }
    payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_segment = _b64url_encode(payload_raw)
    signature_segment = _sign_token(payload_segment)
    return f"{payload_segment}.{signature_segment}"


def _verify_access_token(token: str) -> str | None:
    try:
        payload_segment, signature_segment = token.split(".", 1)
    except ValueError:
        return None

    expected = _sign_token(payload_segment)
    if not hmac.compare_digest(signature_segment, expected):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_segment).decode("utf-8"))
    except Exception:
        return None

    user_id = str(payload.get("sub") or "").strip()
    exp = int(payload.get("exp") or 0)
    if not user_id or exp <= int(time.time()):
        return None
    return user_id


def _hash_password(password: str) -> str:
    global _password_hasher
    try:
        from argon2 import PasswordHasher

        if _password_hasher is None:
            _password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
        return _password_hasher.hash(password)
    except ImportError:
        pass

    # Compatibility fallback for minimal/legacy installations.
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        max(100000, PASSWORD_HASH_ITERATIONS),
    )
    return (
        "pbkdf2_sha256$"
        f"{max(100000, PASSWORD_HASH_ITERATIONS)}$"
        f"{_b64url_encode(salt)}$"
        f"{_b64url_encode(digest)}"
    )


def _verify_password(password: str, stored_hash: str) -> bool:
    global _password_hasher
    if stored_hash.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            from argon2.exceptions import VerificationError

            if _password_hasher is None:
                _password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
            return bool(_password_hasher.verify(stored_hash, password))
        except (ImportError, VerificationError):
            return False

    # Continue accepting existing PBKDF2 hashes so upgrades do not lock users out.
    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = _b64url_decode(salt_b64)
        expected = _b64url_decode(digest_b64)
    except Exception:
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        max(100000, iterations),
    )
    return hmac.compare_digest(actual, expected)


def _validate_user_credentials(user_id: str, password: str) -> tuple[str, str]:
    clean_user_id = (user_id or "").strip()
    clean_password = password or ""
    if not _USER_ID_PATTERN.match(clean_user_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid user_id. Use 3-64 chars: letters, numbers, dot, underscore, or hyphen."
            ),
        )
    if len(clean_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    return clean_user_id, clean_password


def _is_public_route(path: str) -> bool:
    public_paths = {
        "/auth/register",
        "/auth/login",
        "/metrics",
        "/healthz",
        "/readyz",
        "/capabilities",
        "/openapi.json",
        "/docs",
        "/redoc",
    }
    if path in public_paths:
        return True
    if path.startswith("/docs/"):
        return True
    return False


def _get_request_user_id(request: Request) -> str:
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return str(user_id)


def _resolve_metric_path(request: Request) -> str:
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return request.url.path


def _rate_limit_path(path: str) -> str:
    if path.startswith("/store/"):
        return "/store/{thread_id}"
    return path


def _parse_ymd_date(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_web_preview_items(web_notes: str, recency_days: int) -> list[dict[str, Any]]:
    notes = (web_notes or "").strip()
    if not notes or notes.lower().startswith("web retrieval failed:"):
        return []

    blocks = [block.strip() for block in re.split(r"\n(?=-\s)", notes) if block.strip()]
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=max(1, recency_days))
        if recency_days > 0
        else None
    )

    items: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0].strip()
        if not first.startswith("- "):
            continue

        title = first[2:].strip()
        url = ""
        date_text = ""
        snippet_parts: list[str] = []
        for raw_line in lines[1:]:
            line = raw_line.strip()
            lower = line.lower()
            if lower.startswith("link:"):
                url = line.split(":", 1)[1].strip()
            elif lower.startswith("date:"):
                date_text = line.split(":", 1)[1].strip()
            elif lower.startswith("snippet:"):
                snippet_parts.append(line.split(":", 1)[1].strip())
            else:
                snippet_parts.append(line)

        published_dt = _parse_ymd_date(date_text)
        is_within: bool | None = None
        if cutoff and published_dt:
            is_within = published_dt >= cutoff

        items.append(
            {
                "title": title,
                "url": url,
                "snippet": " ".join(part for part in snippet_parts if part).strip(),
                "published_date": date_text,
                "is_within_recency": is_within,
            }
        )

    if items:
        return items

    # Fallback when provider output format is different.
    first_line = notes.splitlines()[0].strip() if notes else ""
    return [
        {
            "title": first_line[:120] if first_line else "Web search result",
            "url": "",
            "snippet": notes[:600],
            "published_date": "",
            "is_within_recency": None,
        }
    ]


def _b64url_encode_text(value: str) -> str:
    raw = (value or "").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _build_web_hitl_control_payload(
    action: str,
    query: str,
    recency_days: int = 0,
    route: str = "web",
    reason: str = "",
) -> str:
    clean_action = (action or "").strip().lower()
    clean_route = (route or "web").strip().lower()
    clean_days = max(0, int(recency_days or 0))
    return (
        f"WEB_HITL_DECISION?action={clean_action}"
        f"&route={clean_route}"
        f"&days={clean_days}"
        f"&query_b64={_b64url_encode_text(query or '')}"
        f"&reason_b64={_b64url_encode_text(reason or '')}"
    )


def _parse_plain_hitl_decision_text(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    lower = raw.lower()
    if not raw:
        return "", ""
    if lower.startswith("approve"):
        return "approve", ""
    if lower in {"yes", "y", "ok", "okay", "continue", "proceed"}:
        return "approve", ""
    if lower.startswith("reject"):
        return "reject", raw[len("reject"):].strip(" :-")
    if lower.startswith("no"):
        return "reject", raw[len("no"):].strip(" :-")
    return "", ""


def _extract_pending_hitl_from_state(state: Dict[str, Any]) -> Dict[str, Any] | None:
    decision = str(state.get("web_hitl_decision") or "").strip().lower()
    if decision != "awaiting":
        return None
    query = str(state.get("web_hitl_pending_query") or "").strip()
    if not query:
        return None
    route = str(state.get("web_hitl_pending_route") or state.get("route") or "web").strip().lower()
    if route not in {"web", "hybrid"}:
        route = "web"
    recency_days = int(state.get("web_hitl_pending_recency_days") or state.get("recency_days") or 0)
    return {
        "query": query,
        "route": route,
        "recency_days": recency_days,
    }


def _extract_interrupt_payload(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize LangGraph v1/v2 interrupt output to a JSON-like payload."""
    raw_interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
    if not raw_interrupts:
        raw_interrupts = getattr(state, "interrupts", None)
    if not raw_interrupts:
        return None
    items = raw_interrupts if isinstance(raw_interrupts, (list, tuple)) else [raw_interrupts]
    first = items[0] if items else None
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return dict(value)
    if value is not None:
        return {"kind": "approval", "message": str(value)}
    return None


def _cache_native_interrupt(
    app: FastAPI,
    user_id: str | None,
    thread_id: str,
    payload: Dict[str, Any],
) -> None:
    cache = getattr(app.state, "web_hitl_pending_cache", None)
    if not isinstance(cache, dict):
        return
    cache_key = f"{(user_id or '').strip()}::{thread_id}"
    cache[cache_key] = {**payload, "created_at_epoch": time.time()}


def _checkpoint_config(user_id: str | None, thread_id: str, model: str) -> RunnableConfig:
    checkpoint_ns = CHECKPOINT_NAMESPACE
    clean_user_id = (user_id or "").strip()
    if clean_user_id:
        checkpoint_ns = f"{CHECKPOINT_NAMESPACE}:{clean_user_id}"
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "model": model,
            "user_id": clean_user_id,
        }
    )


def _select_runtime_model(
    requested_model: str, query: str, request_id: str = ""
) -> tuple[str, Dict[str, Any]]:
    """Resolve the opt-in adaptive pseudo-model through the active learned policy."""
    global _adaptive_router_cache, _contextual_bandit_cache
    if requested_model != "adaptive":
        return requested_model, {"policy": "explicit", "selected_model": requested_model}
    if not ADAPTIVE_MODEL_ROUTER_ENABLED:
        raise HTTPException(status_code=400, detail="Adaptive model routing is disabled")
    if CONTEXTUAL_BANDIT_ROUTER_ENABLED:
        try:
            modified = CONTEXTUAL_BANDIT_POLICY_PATH.stat().st_mtime
            if _contextual_bandit_cache is None or _contextual_bandit_cache[0] != modified:
                artifact = BanditArtifact.load(CONTEXTUAL_BANDIT_POLICY_PATH)
                _contextual_bandit_cache = (modified, ContextualBanditPolicy(artifact))
            policy = _contextual_bandit_cache[1]
            high_risk = bool(policy.features(query)[7])
            decision = policy.decide(
                query,
                request_id=request_id or hashlib.sha256(query.encode()).hexdigest(),
                high_risk=high_risk,
                epsilon=CONTEXTUAL_BANDIT_EPSILON,
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.error(
                json.dumps({"event": "contextual_bandit_load_failed", "error": type(exc).__name__})
            )
            raise HTTPException(
                status_code=503, detail="Contextual-bandit routing policy is unavailable"
            ) from exc
        selected_estimate = next(
            item for item in decision.estimates if item.action == decision.action
        )
        return decision.model, {
            "policy": "safety_constrained_contextual_bandit",
            "selected_tier": decision.action,
            "selected_model": decision.model,
            "propensity": round(decision.propensity, 6),
            "exploration": decision.exploration,
            "expected_utility": round(selected_estimate.expected_utility, 6),
            "uncertainty": round(selected_estimate.uncertainty, 6),
            "feasible_actions": decision.feasible_actions,
            "high_risk_override": high_risk,
            "policy_fingerprint": decision.policy_fingerprint,
        }
    if not ADAPTIVE_SMALL_MODEL or not ADAPTIVE_STRONG_MODEL:
        raise HTTPException(status_code=503, detail="Adaptive model routing is not configured")
    try:
        modified = ADAPTIVE_MODEL_ROUTER_PATH.stat().st_mtime
        if _adaptive_router_cache is None or _adaptive_router_cache[0] != modified:
            _adaptive_router_cache = (modified, CostAwareRouter.load(ADAPTIVE_MODEL_ROUTER_PATH))
        router = _adaptive_router_cache[1]
    except (OSError, ValueError, KeyError) as exc:
        logger.error(json.dumps({"event": "adaptive_router_load_failed", "error": type(exc).__name__}))
        raise HTTPException(status_code=503, detail="Adaptive model router artifact is unavailable") from exc
    high_risk = bool(router.features(query)[7])
    tier = router.select(query, high_risk=high_risk)
    selected = ADAPTIVE_STRONG_MODEL if tier == "strong" else ADAPTIVE_SMALL_MODEL
    return selected, {
        "policy": "learned_cost_aware",
        "selected_tier": tier,
        "selected_model": selected,
        "strong_probability": round(router.probability_strong(query, high_risk), 6),
        "threshold": router.threshold,
        "high_risk_override": high_risk,
        "training_fingerprint": router.training_fingerprint,
    }


async def _resolve_native_resume(
    app: FastAPI,
    user_id: str | None,
    thread_id: str,
    raw_message: str,
    model: str,
) -> Dict[str, str] | None:
    action, reason = _parse_plain_hitl_decision_text(raw_message)
    if action not in {"approve", "reject"}:
        return None
    cache = getattr(app.state, "web_hitl_pending_cache", {})
    cache_key = f"{(user_id or '').strip()}::{thread_id}"
    pending = cache.get(cache_key) if isinstance(cache, dict) else None
    if isinstance(pending, dict):
        created_at = float(pending.get("created_at_epoch") or 0.0)
        if created_at > 0 and (time.time() - created_at) <= HITL_PENDING_TTL_SECONDS:
            return {"action": action, "reason": reason}
        cache.pop(cache_key, None)

    # Checkpoints, not process memory, are the source of truth. This permits a
    # native interrupt to resume after an API restart or on another replica.
    agent = getattr(app.state, "agent", None)
    if agent is None or not hasattr(agent, "aget_state"):
        return None
    try:
        snapshot = await agent.aget_state(_checkpoint_config(user_id, thread_id, model))
    except Exception:
        return None
    interrupts = list(getattr(snapshot, "interrupts", None) or [])
    for task in getattr(snapshot, "tasks", None) or []:
        interrupts.extend(list(getattr(task, "interrupts", None) or []))
    values = getattr(snapshot, "values", None) or {}
    legacy_pending = _extract_pending_hitl_from_state(values) if isinstance(values, dict) else None
    if not interrupts and legacy_pending is None:
        return None
    return {"action": action, "reason": reason}


def _maybe_rewrite_hitl_message(
    app: FastAPI,
    user_id: str | None,
    thread_id: str,
    raw_message: str,
) -> str:
    message = (raw_message or "").strip()
    if not message:
        return message
    if "WEB_HITL_DECISION?" in message or "__WEB_HITL__|" in message:
        return message

    action, reason = _parse_plain_hitl_decision_text(message)
    if action not in {"approve", "reject"}:
        return message

    cache = getattr(app.state, "web_hitl_pending_cache", {})
    cache_key = f"{(user_id or '').strip()}::{thread_id}"
    pending = cache.get(cache_key) or {}
    pending_query = str(pending.get("query") or "").strip()
    if not pending_query:
        return message
    pending_route = str(pending.get("route") or "web").strip().lower()
    if pending_route not in {"web", "hybrid"}:
        pending_route = "web"
    pending_days = int(pending.get("recency_days") or 0)
    return _build_web_hitl_control_payload(
        action=action,
        query=pending_query,
        recency_days=pending_days,
        route=pending_route,
        reason=reason,
    )


def _update_hitl_pending_cache(
    app: FastAPI,
    user_id: str | None,
    thread_id: str,
    state: Dict[str, Any],
) -> None:
    cache = getattr(app.state, "web_hitl_pending_cache", None)
    if not isinstance(cache, dict):
        return
    cache_key = f"{(user_id or '').strip()}::{thread_id}"
    decision = str(state.get("web_hitl_decision") or "").strip().lower()
    if decision in {"approved", "rejected"}:
        cache.pop(cache_key, None)
        return
    pending = _extract_pending_hitl_from_state(state)
    if pending:
        cache[cache_key] = pending


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (
        ENABLE_USER_AUTH
        and APP_ENV in {"production", "prod"}
        and USER_AUTH_SECRET == "dev-insecure-user-auth-secret-change-me"
    ):
        raise RuntimeError("USER_AUTH_SECRET must be set to a strong value in production.")
    async with AsyncExitStack() as stack:
        saver, backend = await _open_checkpointer(stack)
        store_result = open_conversation_store(
            postgres_uri=POSTGRES_STORE_URI,
            sqlite_path=STORE_DB_PATH,
            namespace=STORE_NAMESPACE,
            fallback_sqlite=STORE_FALLBACK_SQLITE,
        )
        app.state.agent = build_research_assistant(checkpointer=saver)
        app.state.checkpoint_backend = backend
        app.state.store = store_result.store
        app.state.store_backend = store_result.backend_label
        app.state.web_hitl_pending_cache = {}
        app.state.rate_limit_buckets = defaultdict(deque)
        app.state.rate_limit_lock = asyncio.Lock()
        if CODE_AGENT_ENABLED:
            await CODE_AGENT_MANAGER.start()
        logger.info(json.dumps({"event": "service_started", "checkpoint_backend": backend,
                                "store_backend": store_result.backend_label}))
        try:
            yield
        finally:
            await CODE_AGENT_MANAGER.cancel_all()
            await close_mcp_client()
            close_store = getattr(store_result.store, "close", None)
            if callable(close_store):
                await asyncio.to_thread(close_store)
    # context managers are cleaned up by AsyncExitStack on exit

app = FastAPI(
    title="AgentForge API",
    version="0.3.0",
    description=(
        "Evaluation-driven, safety-gated agent research with hybrid retrieval, "
        "durable approvals, and optional experiment reporting."
    ),
    lifespan=lifespan,
)
app.include_router(code_agent_router)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    method = request.method

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        path = _resolve_metric_path(request)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code="500").inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
        raise

    duration = time.perf_counter() - start
    path = _resolve_metric_path(request)
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
    return response


@app.middleware("http")
async def request_guard_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip()[:128] or str(uuid4())
    request.state.request_id = request_id
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            content_length = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            content_length = MAX_REQUEST_BYTES + 1
        if content_length > MAX_REQUEST_BYTES:
            return Response(
                status_code=413,
                content="Request body too large",
                headers={"X-Request-ID": request_id},
            )

    exempt = {"/healthz", "/readyz", "/metrics"}
    if request.url.path not in exempt:
        host = request.client.host if request.client else "unknown"
        key = f"{host}:{_rate_limit_path(request.url.path)}"
        now = time.monotonic()
        buckets = getattr(app.state, "rate_limit_buckets", defaultdict(deque))
        lock = getattr(app.state, "rate_limit_lock", None)
        if lock is not None:
            async with lock:
                bucket = buckets[key]
                cutoff = now - RATE_LIMIT_WINDOW_SECONDS
                while bucket and bucket[0] < cutoff:
                    bucket.popleft()
                if len(bucket) >= RATE_LIMIT_REQUESTS:
                    return Response(
                        status_code=429,
                        content="Rate limit exceeded",
                        headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS), "X-Request-ID": request_id},
                    )
                bucket.append(now)

    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": _resolve_metric_path(request),
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    agent = getattr(app.state, "agent", None)
    store = getattr(app.state, "store", None)
    if agent is None or store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        store_ok = await asyncio.wait_for(asyncio.to_thread(store.ping), timeout=2.0)
    except Exception:
        store_ok = False
    if not store_ok:
        raise HTTPException(status_code=503, detail="Conversation store is unavailable")
    return {
        "status": "ready",
        "checkpoint_backend": str(getattr(app.state, "checkpoint_backend", "unknown")),
        "store_backend": str(getattr(app.state, "store_backend", "unknown")),
    }


@app.get("/capabilities")
async def capabilities():
    """Public, non-secret runtime capabilities used by thin clients."""
    models = []
    if os.getenv("OPENAI_API_KEY"):
        openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        models.append({"id": openai_model, "label": f"OpenAI · {openai_model}"})
    if os.getenv("GROQ_API_KEY"):
        groq_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
        models.append({"id": groq_model, "label": f"Groq · {groq_model}"})
    if (
        ADAPTIVE_MODEL_ROUTER_ENABLED
        and (
            (CONTEXTUAL_BANDIT_ROUTER_ENABLED and CONTEXTUAL_BANDIT_POLICY_PATH.exists())
            or (
                ADAPTIVE_SMALL_MODEL
                and ADAPTIVE_STRONG_MODEL
                and ADAPTIVE_MODEL_ROUTER_PATH.exists()
            )
        )
    ):
        label = (
            "Adaptive - contextual bandit"
            if CONTEXTUAL_BANDIT_ROUTER_ENABLED
            else "Adaptive - quality/cost router"
        )
        models.append({"id": "adaptive", "label": label})
    return {
        "models": models,
        "features": {
            "streaming": True,
            "native_hitl": True,
            "local_rag": bool(os.getenv("USE_CHROMA_RAG", "true").lower() not in {"0", "false", "off"}),
            "graph_rag": bool(os.getenv("GRAPH_RAG_ENABLED", "true").lower() not in {"0", "false", "off"}),
            "mcp": bool(os.getenv("MCP_TOOLS_ENABLED", "false").lower() not in {"0", "false", "off"}),
            "evaluation_api": ENABLE_EVAL_API,
            "contextual_bandit_router": CONTEXTUAL_BANDIT_ROUTER_ENABLED,
            "process_reward_model": bool(
                os.getenv("PROCESS_REWARD_MODEL_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "verifier_guided_mcts": bool(
                os.getenv("VERIFIER_MCTS_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "learned_world_model_planning": bool(
                os.getenv("WORLD_MODEL_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "conservative_offline_rl_planning": bool(
                os.getenv("OFFLINE_RL_POLICY_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "search_policy_distillation": os.getenv(
                "SEARCH_DISTILLATION_ENABLED", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            "execution_feedback_calibration": os.getenv(
                "EXECUTION_REPLAY_ENABLED", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            "uncertainty_aware_verifier_ensemble": bool(
                os.getenv("VERIFIER_ENSEMBLE_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "verifier_active_learning": bool(
                os.getenv("VERIFIER_ACTIVE_LEARNING_ENABLED", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "sandboxed_code_agent": CODE_AGENT_ENABLED,
            "durable_code_jobs": True,
            "verified_pr_workflow": True,
            "graph_code_context": True,
            "hybrid_code_context": True,
            "genai_otel_tracing": bool(
                os.getenv("GENAI_OTEL_ENABLED", "false").lower()
                not in {"0", "false", "off"}
            ),
            "privacy_safe_local_traces": bool(os.getenv("GENAI_TRACE_JSONL_PATH", "").strip()),
            "reliability_fault_injection_lab": True,
            "stateful_agent_arena": True,
            "llm_inference_control_plane": True,
            "llm_gateway_runtime_enabled": MODEL_GATEWAY_ENABLED,
            "online_ai_governance": True,
            "trustworthy_long_term_memory": True,
            "long_term_memory_runtime_enabled": AGENT_MEMORY_ENABLED,
            "claim_level_grounding_verification": True,
            "grounding_runtime_enabled": os.getenv(
                "GROUNDING_VERIFICATION_ENABLED", "false"
            ).lower() in {"1", "true", "yes", "on"},
            "conformal_uncertainty_control": True,
            "uncertainty_runtime_enabled": os.getenv(
                "UNCERTAINTY_CALIBRATION_ENABLED", "false"
            ).lower() in {"1", "true", "yes", "on"},
            "adaptive_test_time_compute": True,
            "adaptive_compute_runtime_enabled": (
                os.getenv("ADAPTIVE_COMPUTE_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
                and os.getenv("GROUNDING_VERIFICATION_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
            ),
            "evidence_intelligence": True,
            "evidence_quality_runtime_enabled": os.getenv(
                "EVIDENCE_QUALITY_ENABLED", "false"
            ).lower() in {"1", "true", "yes", "on"},
        },
        "code_context": {
            "version": "2.0",
            "languages": ["python", "typescript", "javascript", "java", "go", "rust"],
            "strategies": ["lexical", "lexical_graph", "hybrid", "hybrid_rerank"],
        },
        "inference_gateway": {
            "version": os.getenv("MODEL_GATEWAY_POLICY_VERSION", "inference-policy-v1"),
            "enabled": MODEL_GATEWAY_ENABLED,
            "controls": [
                "tenant_budget",
                "prompt_admission",
                "provider_timeout",
                "circuit_breaker",
                "fallback",
                "tenant_scoped_similarity_cache",
                "stable_canary",
                "shadow_evaluation",
                "privacy_safe_receipt",
            ],
        },
        "online_evaluation": {
            "version": "online-slo-v1",
            "runtime_event_export_enabled": bool(
                os.getenv("MODEL_GATEWAY_ONLINE_EVENT_PATH", "").strip()
            ),
            "controls": [
                "exactly_once_ingestion",
                "delayed_feedback_join",
                "rolling_quality_safety_cost_latency_slos",
                "multi_window_error_budget_burn",
                "provider_mix_drift",
                "statistical_canary_promotion_rollback",
                "integrity_bound_content_free_events",
            ],
        },
        "agent_memory": {
            "version": "memory-policy-v1",
            "enabled": AGENT_MEMORY_ENABLED,
            "types": ["episodic", "semantic", "preference", "procedural"],
            "controls": [
                "tenant_isolation",
                "hybrid_temporal_retrieval",
                "token_budget",
                "provenance_and_integrity",
                "signed_audit_trail",
                "conflict_versioning",
                "poisoning_quarantine",
                "pii_redaction",
                "ttl_and_tombstones",
                "outcome_usefulness",
                "episodic_consolidation",
                "export_and_forget",
            ],
        },
        "grounding_verification": {
            "version": "grounding-policy-v1",
            "enabled": os.getenv("GROUNDING_VERIFICATION_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            "controls": [
                "claim_evidence_alignment",
                "citation_allowlisting",
                "high_risk_claim_thresholds",
                "explainable_confidence_scoring",
                "bounded_repair",
                "fail_closed_abstention",
                "integrity_bound_receipts",
            ],
        },
        "uncertainty_control": {
            "version": "mondrian-split-conformal-v1",
            "enabled": os.getenv("UNCERTAINTY_CALIBRATION_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            "controls": [
                "held_out_calibration_split",
                "route_aware_mondrian_thresholds",
                "correctness_prediction_sets",
                "selective_answering",
                "fail_closed_artifact_loading",
                "confidence_distribution_drift",
                "integrity_bound_decisions",
            ],
        },
        "adaptive_compute": {
            "version": "adaptive-compute-v1",
            "enabled": (
                os.getenv("ADAPTIVE_COMPUTE_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
                and os.getenv("GROUNDING_VERIFICATION_ENABLED", "false").lower()
                in {"1", "true", "yes", "on"}
            ),
            "controls": [
                "confidence_and_risk_early_exit",
                "bounded_candidate_generation",
                "token_latency_and_call_budgets",
                "grounded_candidate_verification",
                "conformal_candidate_filtering",
                "independent_evidence_consensus",
                "fail_closed_abstention",
                "privacy_safe_integrity_receipts",
            ],
        },
        "evidence_quality": {
            "version": "evidence-quality-v1",
            "enabled": os.getenv("EVIDENCE_QUALITY_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            "controls": [
                "retrieval_prompt_injection_quarantine",
                "cross_domain_near_duplicate_collapse",
                "independent_source_requirements",
                "temporal_freshness_validation",
                "numeric_and_negation_conflict_graph",
                "authority_signals",
                "filtered_pre_synthesis_context",
                "integrity_bound_quality_receipts",
            ],
        },
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/evals/experiments")
async def list_evaluation_experiments(limit: int = 50):
    """List locally registered experiment reports when the admin feature is enabled."""
    if not ENABLE_EVAL_API:
        raise HTTPException(status_code=404, detail="Evaluation API is disabled")
    return {"experiments": await asyncio.to_thread(ExperimentStore(EVAL_RESULTS_DIR).list, limit)}


@app.get("/evals/experiments/{experiment_id}")
async def get_evaluation_experiment(experiment_id: str):
    """Return one portable evaluation report without accepting filesystem paths."""
    if not ENABLE_EVAL_API:
        raise HTTPException(status_code=404, detail="Evaluation API is disabled")
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", experiment_id):
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    report = await asyncio.to_thread(ExperimentStore(EVAL_RESULTS_DIR).get, experiment_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return report.model_dump()


@app.middleware("http")
async def check_auth_header(request: Request, call_next):
    if _is_public_route(request.url.path):
        return await call_next(request)

    if ENABLE_USER_AUTH:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        token = auth_header[7:].strip()
        user_id = _verify_access_token(token)
        if not user_id:
            return Response(status_code=401, content="Invalid or expired token")
        request.state.user_id = user_id
    else:
        if auth_secret := os.getenv("AUTH_SECRET"):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return Response(status_code=401, content="Missing or invalid token")
            if auth_header[7:] != auth_secret:
                return Response(status_code=401, content="Invalid token")
    return await call_next(request)


@app.post("/auth/register")
async def register(auth_input: AuthRegisterInput) -> AuthToken:
    """
    Register a new user and return a bearer token.
    """
    store = getattr(app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="Conversation store is not available.")

    user_id, password = _validate_user_credentials(auth_input.user_id, auth_input.password)
    password_hash = _hash_password(password)
    created = await asyncio.to_thread(store.create_user, user_id, password_hash)
    if not created:
        raise HTTPException(status_code=409, detail="User already exists.")

    token = _create_access_token(user_id)
    return AuthToken(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        expires_in=max(60, USER_AUTH_TOKEN_TTL_SECONDS),
    )


@app.post("/auth/login")
async def login(auth_input: AuthLoginInput) -> AuthToken:
    """
    Authenticate a user and return a bearer token.
    """
    store = getattr(app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="Conversation store is not available.")

    user_id, password = _validate_user_credentials(auth_input.user_id, auth_input.password)
    stored_hash = await asyncio.to_thread(store.get_user_password_hash, user_id)
    if not stored_hash or not _verify_password(password, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid user_id or password.")

    token = _create_access_token(user_id)
    return AuthToken(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        expires_in=max(60, USER_AUTH_TOKEN_TTL_SECONDS),
    )


def _parse_input(
    user_input: UserInput,
    user_id: str | None = None,
    resume_value: Dict[str, str] | None = None,
    thread_id: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    thread_id = thread_id or user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    selected_model, model_selection = _select_runtime_model(
        user_input.model, user_input.message, thread_id
    )
    config = _checkpoint_config(user_id, thread_id, selected_model)
    config["configurable"]["requested_model"] = user_input.model
    config["configurable"]["model_selection"] = model_selection
    config["configurable"]["execution_replay_consent"] = user_input.execution_replay_consent
    config["configurable"]["execution_replay_request_id"] = str(run_id)
    replay_key = os.getenv("EXECUTION_REPLAY_KEY", "").encode()
    family = user_input.execution_replay_task_family
    # Do not put a raw family ID into graph/checkpoint/tracing configuration.
    config["configurable"]["execution_replay_task_family_fingerprint"] = (
        digest(replay_key, "task-family", [config["configurable"]["user_id"], family])
        if family and user_input.execution_replay_consent and len(replay_key) >= 32
        else None
    )
    config["run_id"] = run_id
    kwargs = dict(
        input=(Command(resume=resume_value) if resume_value is not None else {"messages": [input_message.to_langchain()]}),
        config=config,
    )
    return kwargs, run_id


async def _store_message_safely(
    app: FastAPI,
    user_id: str | None,
    thread_id: str,
    run_id: str,
    role: str,
    content: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    store = getattr(app.state, "store", None)
    if store is None:
        return
    try:
        await asyncio.to_thread(
            store.save_message,
            user_id,
            thread_id,
            str(run_id),
            role,
            content,
            metadata or {},
        )
    except Exception as e:
        logger.warning(json.dumps({"event": "store_write_failed", "kind": role, "error": str(e)}))


async def _store_hitl_event_safely(
    app: FastAPI,
    user_id: str | None,
    thread_id: str | None,
    query: str,
    decision: str,
    reason: str = "",
    metadata: Dict[str, Any] | None = None,
) -> None:
    store = getattr(app.state, "store", None)
    if store is None:
        return
    try:
        await asyncio.to_thread(
            store.save_hitl_event,
            user_id,
            thread_id,
            query,
            decision,
            reason,
            metadata or {},
        )
    except Exception as e:
        logger.warning(json.dumps({"event": "store_write_failed", "kind": f"hitl:{decision}", "error": str(e)}))


def _record_agent_result(state: Dict[str, Any], outcome: str) -> None:
    allowed_routes = {"clarify", "rewrite", "math", "web", "rag", "kg", "hybrid", "general"}
    route = str(state.get("route") or "unknown").lower()
    if route not in allowed_routes:
        route = "unknown"
    AGENT_RUNS_TOTAL.labels(route=route, outcome=outcome).inc()
    allowed_nodes = {
        "safety_agent", "memory_retrieval_agent", "intent_router_agent", "clarification_agent", "query_rewriter_agent",
        "recency_guard_agent", "web_hitl_gate_agent", "web_search_agent", "knowledge_graph_agent",
        "rag_agent", "math_agent", "response_agent", "grounding_verifier_agent",
        "evidence_adjudication_agent", "adaptive_deliberation_agent",
        "grounding_repair_agent", "evaluation_agent", "memory_write_agent",
    }
    for node, duration_ms in (state.get("agent_latency_ms") or {}).items():
        if node not in allowed_nodes:
            continue
        try:
            AGENT_NODE_DURATION_SECONDS.labels(node=node).observe(float(duration_ms) / 1000.0)
        except (TypeError, ValueError):
            continue


def _capture_genai_trace_safely(
    *,
    run_id: str,
    model: str,
    state: Dict[str, Any],
    outcome: str,
    started: float,
    stream: bool = False,
) -> None:
    try:
        capture_research_agent_trace(
            run_id=run_id,
            model=model,
            state=state,
            outcome=outcome,
            duration_ms=(time.perf_counter() - started) * 1000,
            stream=stream,
        )
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "event": "genai_trace_export_failed",
                    "run_id": run_id,
                    "error": type(exc).__name__,
                }
            )
        )


def _extract_hitl_audit_event(state: Dict[str, Any]) -> Dict[str, Any] | None:
    decision = str(state.get("web_hitl_decision") or "").strip().lower()
    if decision not in {"approved", "rejected"}:
        return None

    query = str(
        state.get("web_hitl_audit_query")
        or state.get("web_hitl_pending_query")
        or state.get("query")
        or ""
    ).strip()
    if not query:
        return None

    reason = str(state.get("web_hitl_reject_reason") or "").strip()
    if decision == "approved":
        reason = ""

    metadata = {
        "recency_days": int(state.get("web_hitl_pending_recency_days") or state.get("recency_days") or 0),
        "preview_count": int(state.get("web_hitl_preview_count") or 0),
        "all_within_recency": bool(state.get("web_hitl_all_within_recency")),
        "source": str(state.get("web_hitl_pending_source_meta") or state.get("web_source_meta") or ""),
        "cache_hit": False,
        "audit_source": "graph",
    }
    return {
        "query": query,
        "decision": decision,
        "reason": reason,
        "metadata": metadata,
    }


@app.post("/invoke")
async def invoke(user_input: UserInput, request: Request) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.
    
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledGraph = app.state.agent
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    effective_thread_id = user_input.thread_id or str(uuid4())
    resume_value = await _resolve_native_resume(
        app, user_id, effective_thread_id, user_input.message, user_input.model
    )
    kwargs, run_id = _parse_input(
        user_input,
        user_id=user_id,
        resume_value=resume_value,
        thread_id=effective_thread_id,
    )
    thread_id = kwargs["config"]["configurable"]["thread_id"]
    selected_model = kwargs["config"]["configurable"]["model"]
    await _store_message_safely(
        app,
        user_id=user_id,
        thread_id=thread_id,
        run_id=str(run_id),
        role="human",
        content=user_input.message,
        metadata={
            "requested_model": user_input.model,
            "selected_model": kwargs["config"]["configurable"]["model"],
            "model_selection": kwargs["config"]["configurable"]["model_selection"],
        },
    )
    agent_started = time.perf_counter()
    try:
        response = await agent.ainvoke(**kwargs)
    except Exception as e:
        AGENT_RUNS_TOTAL.labels(route="unknown", outcome="error").inc()
        _capture_genai_trace_safely(
            run_id=str(run_id),
            model=selected_model,
            state={},
            outcome="error",
            started=agent_started,
        )
        logger.exception(json.dumps({"event": "agent_invoke_failed", "run_id": str(run_id)}))
        raise HTTPException(status_code=503, detail="Agent execution failed. Please retry.") from e

    try:
        interrupt_payload = _extract_interrupt_payload(response)
        if interrupt_payload:
            _record_agent_result(response, "interrupted")
            _capture_genai_trace_safely(
                run_id=str(run_id),
                model=selected_model,
                state=response,
                outcome="interrupted",
                started=agent_started,
            )
            _cache_native_interrupt(app, user_id, thread_id, interrupt_payload)
            output = ChatMessage(
                type="ai",
                content=str(interrupt_payload.get("message") or "Human approval is required."),
                run_id=str(run_id),
            )
            await _store_message_safely(
                app,
                user_id=user_id,
                thread_id=thread_id,
                run_id=str(run_id),
                role="ai",
                content=output.content,
                metadata={"event": "interrupt", "kind": interrupt_payload.get("kind", "approval")},
            )
            return output
        output = ChatMessage.from_langchain(response["messages"][-1])
        _record_agent_result(response, "completed")
        _capture_genai_trace_safely(
            run_id=str(run_id),
            model=selected_model,
            state=response,
            outcome="completed",
            started=agent_started,
        )
        output.run_id = str(run_id)
        await _store_message_safely(
            app,
            user_id=user_id,
            thread_id=thread_id,
            run_id=str(run_id),
            role="ai",
            content=output.content,
            metadata={
                "route": response.get("route"),
                "evaluation_score": response.get("evaluation_score"),
                "evaluation_report": response.get("evaluation_report"),
                "grounding_action": response.get("grounding_action"),
                "grounding_confidence": response.get("grounding_confidence"),
                "grounding_report_fingerprint": (
                    response.get("grounding_report") or {}
                ).get("report_fingerprint"),
                "uncertainty_decision": (
                    response.get("uncertainty_receipt") or {}
                ).get("decision"),
                "uncertainty_receipt_fingerprint": (
                    response.get("uncertainty_receipt") or {}
                ).get("receipt_fingerprint"),
                "adaptive_compute_status": (
                    response.get("adaptive_compute_receipt") or {}
                ).get("status"),
                "adaptive_compute_receipt_fingerprint": (
                    response.get("adaptive_compute_receipt") or {}
                ).get("receipt_fingerprint"),
                "execution_replay_event_ids": response.get("execution_replay_event_ids") or [],
                "evidence_quality_action": (
                    response.get("evidence_quality_report") or {}
                ).get("action"),
                "evidence_quality_report_fingerprint": (
                    response.get("evidence_quality_report") or {}
                ).get("report_fingerprint"),
            },
        )
        hitl_event = _extract_hitl_audit_event(response)
        _update_hitl_pending_cache(app, user_id, thread_id, response)
        if hitl_event:
            await _store_hitl_event_safely(
                app,
                user_id=user_id,
                thread_id=thread_id,
                query=hitl_event["query"],
                decision=hitl_event["decision"],
                reason=hitl_event["reason"],
                metadata=hitl_event["metadata"],
            )
        return output
    except Exception as e:
        logger.exception(json.dumps({"event": "agent_response_processing_failed", "run_id": str(run_id)}))
        raise HTTPException(status_code=500, detail="Response processing failed.") from e

async def message_generator(
    user_input: StreamInput,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledGraph = app.state.agent
    effective_thread_id = user_input.thread_id or str(uuid4())
    resume_value = await _resolve_native_resume(
        app, user_id, effective_thread_id, user_input.message, user_input.model
    )
    kwargs, run_id = _parse_input(
        user_input,
        user_id=user_id,
        resume_value=resume_value,
        thread_id=effective_thread_id,
    )
    thread_id = kwargs["config"]["configurable"]["thread_id"]
    selected_model = kwargs["config"]["configurable"]["model"]
    await _store_message_safely(
        app,
        user_id=user_id,
        thread_id=thread_id,
        run_id=str(run_id),
        role="human",
        content=user_input.message,
        metadata={
            "requested_model": user_input.model,
            "selected_model": kwargs["config"]["configurable"]["model"],
            "model_selection": kwargs["config"]["configurable"]["model_selection"],
            "stream": True,
        },
    )

    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]
    
    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        try:
            async for s in agent.astream(**kwargs, stream_mode="updates"):
                await output_queue.put(s)
        except Exception:
            await output_queue.put({"__error__": "Agent execution failed. Please retry."})
        finally:
            await output_queue.put(None)
    agent_started = time.perf_counter()
    stream_task = asyncio.create_task(run_agent_stream())
    stored_message_fingerprints = set()
    hitl_event: Dict[str, Any] | None = None
    latest_agent_state: Dict[str, Any] = {}
    stream_outcome = "completed"

    # Process the queue and yield messages over the SSE stream.
    while s := await output_queue.get():
        if isinstance(s, str):
            # str is an LLM token
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue
        if isinstance(s, dict) and "__error__" in s:
            stream_outcome = "error"
            yield f"data: {json.dumps({'type': 'error', 'content': s['__error__']})}\n\n"
            continue

        interrupt_payload = _extract_interrupt_payload(s) if isinstance(s, dict) else None
        if interrupt_payload:
            stream_outcome = "interrupted"
            _cache_native_interrupt(app, user_id, thread_id, interrupt_payload)
            chat_message = ChatMessage(
                type="ai",
                content=str(interrupt_payload.get("message") or "Human approval is required."),
                run_id=str(run_id),
            )
            await _store_message_safely(
                app,
                user_id=user_id,
                thread_id=thread_id,
                run_id=str(run_id),
                role="ai",
                content=chat_message.content,
                metadata={"event": "interrupt", "kind": interrupt_payload.get("kind", "approval")},
            )
            yield f"data: {json.dumps({'type': 'message', 'content': model_dump_compat(chat_message)})}\n\n"
            continue

        # Otherwise, s should be a dict of state updates for each node in the graph.
        # s could have updates for multiple nodes, so check each for messages.
        new_messages = []
        for node_name, state in s.items():
            if node_name.startswith("__") or not isinstance(state, dict):
                continue
            latest_agent_state.update(state)
            candidate = _extract_hitl_audit_event(state)
            if candidate:
                hitl_event = candidate
            _update_hitl_pending_cache(app, user_id, thread_id, state)
            new_messages.extend(state.get("messages", []))
        for message in new_messages:
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            fingerprint = (
                chat_message.type,
                chat_message.content.strip(),
                chat_message.tool_call_id or "",
            )
            if chat_message.type == "ai" and fingerprint not in stored_message_fingerprints:
                stored_message_fingerprints.add(fingerprint)
                await _store_message_safely(
                    app,
                    user_id=user_id,
                    thread_id=thread_id,
                    run_id=str(run_id),
                    role="ai",
                    content=chat_message.content,
                    metadata={"stream": True},
                )
            yield f"data: {json.dumps({'type': 'message', 'content': model_dump_compat(chat_message)})}\n\n"
    
    await stream_task
    _record_agent_result(latest_agent_state, stream_outcome)
    _capture_genai_trace_safely(
        run_id=str(run_id),
        model=selected_model,
        state=latest_agent_state,
        outcome=stream_outcome,
        started=agent_started,
        stream=True,
    )
    if hitl_event:
        await _store_hitl_event_safely(
            app,
            user_id=user_id,
            thread_id=thread_id,
            query=hitl_event["query"],
            decision=hitl_event["decision"],
            reason=hitl_event["reason"],
            metadata=hitl_event["metadata"],
        )
    yield "data: [DONE]\n\n"

@app.post("/stream")
async def stream_agent(user_input: StreamInput, request: Request):
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.
    
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    return StreamingResponse(
        message_generator(user_input, user_id=user_id),
        media_type="text/event-stream",
    )


@app.post("/web_search/preview")
async def web_search_preview(payload: Dict[str, Any], request: Request):
    """
    Preview web-search results for human approval before final answer generation.
    """
    _ = _get_request_user_id(request) if ENABLE_USER_AUTH else None

    query = str(payload.get("query") or payload.get("message") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        recency_days = int(payload.get("recency_days") or 7)
    except Exception:
        recency_days = 7
    recency_days = max(1, min(recency_days, 30))

    try:
        max_results = int(payload.get("max_results") or 5)
    except Exception:
        max_results = 5
    max_results = max(1, min(max_results, 10))

    try:
        result = await asyncio.to_thread(
            perform_web_search,
            query,
            max_results,
            recency_days,
            query,
            True,
        )
    except Exception as e:
        logger.exception(json.dumps({"event": "web_preview_failed"}))
        raise HTTPException(status_code=500, detail="Web preview failed.") from e

    if isinstance(result, tuple):
        web_notes, meta = result
    else:
        web_notes, meta = str(result), {}

    items = _parse_web_preview_items(web_notes, recency_days)
    dated_items = [item for item in items if item.get("published_date")]
    all_within_recency = (
        bool(dated_items) and all(item.get("is_within_recency") is True for item in dated_items)
    )

    return {
        "query": query,
        "recency_days": recency_days,
        "max_results": max_results,
        "count": len(items),
        "all_within_recency": all_within_recency,
        "source": str(meta.get("source") or ""),
        "cache_hit": bool(meta.get("cache_hit")),
        "items": items,
        "web_notes": web_notes,
    }


@app.post("/hitl/web_decision")
async def record_web_hitl_decision(payload: Dict[str, Any], request: Request):
    """
    Record web-search HITL decision (approved/rejected) for audit trail.
    """
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")

    query = str(payload.get("query") or "").strip()
    decision = str(payload.get("decision") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    thread_id = str(payload.get("thread_id") or "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="decision must be approved or rejected")
    if decision == "approved":
        reason = ""

    metadata = {
        "recency_days": int(payload.get("recency_days") or 0),
        "preview_count": int(payload.get("preview_count") or 0),
        "all_within_recency": bool(payload.get("all_within_recency")),
        "source": str(payload.get("source") or ""),
        "cache_hit": bool(payload.get("cache_hit")),
    }

    await _store_hitl_event_safely(
        app,
        user_id=user_id,
        thread_id=thread_id,
        query=query,
        decision=decision,
        reason=reason,
        metadata=metadata,
    )

    return {
        "status": "recorded",
        "user_id": user_id,
        "thread_id": thread_id,
        "query": query,
        "decision": decision,
    }


@app.get("/hitl/web_decisions")
async def list_web_hitl_decisions(request: Request, limit: int = 50, thread_id: str | None = None):
    """
    List authenticated user's HITL decision audit records.
    """
    store = getattr(app.state, "store", None)
    backend = getattr(app.state, "store_backend", None)
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    if store is None or not user_id:
        return {
            "user_id": user_id or "",
            "backend": backend,
            "count": 0,
            "events": [],
        }

    bounded_limit = max(1, min(int(limit), 200))
    clean_thread_id = (thread_id or "").strip() or None
    try:
        events = await asyncio.to_thread(
            store.list_hitl_events,
            user_id,
            bounded_limit,
            clean_thread_id,
        )
    except Exception as e:
        logger.exception(json.dumps({"event": "hitl_audit_read_failed"}))
        raise HTTPException(status_code=500, detail="Could not read approval history.") from e

    return {
        "user_id": user_id,
        "backend": backend,
        "thread_id": clean_thread_id or "",
        "count": len(events),
        "events": events,
    }


@app.get("/store/threads")
async def list_user_threads(request: Request, limit: int = 30):
    """
    List recent conversation threads for the authenticated user.
    """
    store = getattr(app.state, "store", None)
    backend = getattr(app.state, "store_backend", None)
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    if store is None or not user_id:
        return {
            "user_id": user_id or "",
            "backend": backend,
            "count": 0,
            "threads": [],
        }

    try:
        threads = await asyncio.to_thread(store.list_threads, user_id, limit)
    except Exception as e:
        logger.exception(json.dumps({"event": "thread_list_failed"}))
        raise HTTPException(status_code=500, detail="Could not read threads.") from e

    return {
        "user_id": user_id,
        "backend": backend,
        "count": len(threads),
        "threads": threads,
    }


@app.get("/store/{thread_id}")
async def get_thread_store(thread_id: str, request: Request, limit: int = 50):
    """
    Retrieve persisted conversation records from the conversation Store for a thread.
    """
    store = getattr(app.state, "store", None)
    backend = getattr(app.state, "store_backend", None)
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    if store is None:
        return {
            "thread_id": thread_id,
            "user_id": user_id or "",
            "backend": None,
            "count": 0,
            "messages": [],
        }

    try:
        messages = await asyncio.to_thread(store.list_messages, thread_id, limit, user_id)
    except Exception as e:
        logger.exception(json.dumps({"event": "thread_read_failed"}))
        raise HTTPException(status_code=500, detail="Could not read messages.") from e

    return {
        "thread_id": thread_id,
        "user_id": user_id or "",
        "backend": backend,
        "count": len(messages),
        "messages": messages,
    }


def _memory_store_for_request(request: Request):
    if not AGENT_MEMORY_ENABLED:
        raise HTTPException(status_code=404, detail="Long-term memory is disabled")
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else "local-user"
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required for memory access")
    return get_memory_store(), user_id


@app.post("/memories")
async def create_memory(candidate: MemoryCandidate, request: Request):
    store, user_id = _memory_store_for_request(request)
    receipt = await asyncio.to_thread(store.remember, user_id, candidate)
    return receipt.model_dump(mode="json")


@app.get("/memories")
async def list_memories(
    request: Request,
    memory_type: str | None = None,
    include_inactive: bool = False,
    limit: int = 100,
):
    store, user_id = _memory_store_for_request(request)
    if memory_type and memory_type not in {"episodic", "semantic", "preference", "procedural"}:
        raise HTTPException(status_code=422, detail="Unsupported memory type")
    records = await asyncio.to_thread(
        store.list_memories,
        user_id,
        memory_type=memory_type,
        include_inactive=include_inactive,
        limit=limit,
    )
    return {"count": len(records), "memories": [item.model_dump(mode="json") for item in records]}


@app.get("/memories/search")
async def search_memories(request: Request, q: str, limit: int = 8, token_budget: int = 384):
    store, user_id = _memory_store_for_request(request)
    if not q.strip():
        raise HTTPException(status_code=422, detail="q is required")
    result = await asyncio.to_thread(
        store.search,
        user_id,
        q,
        limit=limit,
        token_budget=token_budget,
    )
    return result.model_dump(mode="json")


@app.get("/memories/export")
async def export_memories(request: Request):
    store, user_id = _memory_store_for_request(request)
    return await asyncio.to_thread(store.export, user_id)


@app.get("/memories/audit")
async def list_memory_audit_events(request: Request, limit: int = 100):
    store, user_id = _memory_store_for_request(request)
    events = await asyncio.to_thread(store.audit_events, user_id, limit)
    return {"count": len(events), "events": events}


@app.post("/memories/consolidate")
async def consolidate_memories(request: Request):
    store, user_id = _memory_store_for_request(request)
    return await asyncio.to_thread(store.consolidate, user_id)


@app.patch("/memories/{memory_id}")
async def correct_memory(memory_id: str, correction: MemoryCorrection, request: Request):
    store, user_id = _memory_store_for_request(request)
    try:
        receipt = await asyncio.to_thread(store.correct, user_id, memory_id, correction)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Active memory not found") from exc
    return receipt.model_dump(mode="json")


@app.post("/memories/{memory_id}/outcome")
async def record_memory_outcome(memory_id: str, outcome: MemoryOutcome, request: Request):
    store, user_id = _memory_store_for_request(request)
    try:
        record = await asyncio.to_thread(store.record_outcome, user_id, memory_id, outcome)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Active memory not found") from exc
    return record.model_dump(mode="json")


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    store, user_id = _memory_store_for_request(request)
    deleted = await asyncio.to_thread(store.delete, user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "memory_id": memory_id}


@app.delete("/memories")
async def forget_all_memories(request: Request):
    store, user_id = _memory_store_for_request(request)
    deleted = await asyncio.to_thread(store.forget_tenant, user_id)
    return {"deleted": deleted, "status": "forgotten"}

@app.post("/feedback")
async def feedback(feedback: Feedback):
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    await asyncio.to_thread(
        client.create_feedback,
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return {"status": "success"}

