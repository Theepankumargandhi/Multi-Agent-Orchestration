import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import hmac
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import AsyncGenerator, Dict, Any, Tuple
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph.graph import CompiledGraph
from langsmith import Client as LangsmithClient

from agent import research_assistant
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
USER_AUTH_TOKEN_TTL_SECONDS = int(os.getenv("USER_AUTH_TOKEN_TTL_SECONDS", "86400"))
PASSWORD_HASH_ITERATIONS = int(os.getenv("PASSWORD_HASH_ITERATIONS", "210000"))
_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")


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

    if not incompatible and "thread_ts" in columns:
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

                setattr(saver, "aput", aput_compat)
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

                setattr(saver, "put", put_compat)
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

    resolved_checkpoint_db = _rotate_incompatible_checkpoint_db(CHECKPOINT_DB_PATH)
    saver = await stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(resolved_checkpoint_db)
    )
    await _ensure_checkpointer_schema(saver)
    return saver, resolved_checkpoint_db


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
    clean_password = (password or "").strip()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        saver, backend = await _open_checkpointer(stack)
        store_result = open_conversation_store(
            postgres_uri=POSTGRES_STORE_URI,
            sqlite_path=STORE_DB_PATH,
            namespace=STORE_NAMESPACE,
            fallback_sqlite=STORE_FALLBACK_SQLITE,
        )
        research_assistant.checkpointer = saver
        app.state.agent = research_assistant
        app.state.checkpoint_backend = backend
        app.state.store = store_result.store
        app.state.store_backend = store_result.backend_label
        print(f"[service] Checkpointer backend: {backend}")
        print(f"[service] Conversation store backend: {store_result.backend_label}")
        yield
    # context managers are cleaned up by AsyncExitStack on exit

app = FastAPI(lifespan=lifespan)

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


def _parse_input(user_input: UserInput, user_id: str | None = None) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    checkpoint_ns = CHECKPOINT_NAMESPACE
    clean_user_id = (user_id or "").strip()
    if clean_user_id:
        checkpoint_ns = f"{CHECKPOINT_NAMESPACE}:{clean_user_id}"
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config=RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                # Postgres checkpoint_writes can enforce NOT NULL checkpoint_id.
                # Use run_id to guarantee a stable non-null identifier per run.
                "checkpoint_id": str(run_id),
                "model": user_input.model,
                "user_id": clean_user_id,
            },
            run_id=run_id,
        ),
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
        print(f"[service] store write failed ({role}): {e}")

@app.post("/invoke")
async def invoke(user_input: UserInput, request: Request) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.
    
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledGraph = app.state.agent
    user_id = _get_request_user_id(request) if ENABLE_USER_AUTH else None
    kwargs, run_id = _parse_input(user_input, user_id=user_id)
    thread_id = kwargs["config"]["configurable"]["thread_id"]
    await _store_message_safely(
        app,
        user_id=user_id,
        thread_id=thread_id,
        run_id=str(run_id),
        role="human",
        content=user_input.message,
        metadata={"model": user_input.model},
    )
    try:
        response = await agent.ainvoke(**kwargs)
    except Exception as e:
        if _is_serializer_compat_error(e) or _is_checkpointer_signature_compat_error(e):
            # Fallback: disable checkpoint persistence for this process and retry once.
            agent.checkpointer = None
            response = await agent.ainvoke(**kwargs)
        else:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        output = ChatMessage.from_langchain(response["messages"][-1])
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
            },
        )
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def message_generator(
    user_input: StreamInput,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input, user_id=user_id)
    thread_id = kwargs["config"]["configurable"]["thread_id"]
    await _store_message_safely(
        app,
        user_id=user_id,
        thread_id=thread_id,
        run_id=str(run_id),
        role="human",
        content=user_input.message,
        metadata={"model": user_input.model, "stream": True},
    )

    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]
    
    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        streamed_any = False
        try:
            async for s in agent.astream(**kwargs, stream_mode="updates"):
                streamed_any = True
                await output_queue.put(s)
        except Exception as e:
            if _is_serializer_compat_error(e) or _is_checkpointer_signature_compat_error(e):
                # Disable checkpoint persistence. Retry only if nothing streamed yet
                # to avoid duplicate partial outputs to the client.
                agent.checkpointer = None
                if not streamed_any:
                    try:
                        async for s in agent.astream(**kwargs, stream_mode="updates"):
                            await output_queue.put(s)
                    except Exception as retry_e:
                        await output_queue.put({"__error__": str(retry_e)})
            else:
                await output_queue.put({"__error__": str(e)})
        finally:
            await output_queue.put(None)
    stream_task = asyncio.create_task(run_agent_stream())
    stored_message_fingerprints = set()

    # Process the queue and yield messages over the SSE stream.
    while s := await output_queue.get():
        if isinstance(s, str):
            # str is an LLM token
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue
        if isinstance(s, dict) and "__error__" in s:
            yield f"data: {json.dumps({'type': 'error', 'content': s['__error__']})}\n\n"
            continue

        # Otherwise, s should be a dict of state updates for each node in the graph.
        # s could have updates for multiple nodes, so check each for messages.
        new_messages = []
        for _, state in s.items():
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
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "thread_id": thread_id,
        "user_id": user_id or "",
        "backend": backend,
        "count": len(messages),
        "messages": messages,
    }

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
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return {"status": "success"}

