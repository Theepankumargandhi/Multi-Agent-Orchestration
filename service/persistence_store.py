import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class StoreOpenResult:
    store: "BaseConversationStore"
    backend_label: str


class BaseConversationStore:
    def setup(self) -> None:
        raise NotImplementedError

    def save_message(
        self,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def list_messages(self, thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SQLiteConversationStore(BaseConversationStore):
    def __init__(self, db_path: str, namespace: str = "default") -> None:
        self.db_path = db_path
        self.namespace = namespace

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_store_thread
                ON conversation_store(namespace, thread_id, created_at DESC)
                """
            )
            conn.commit()

    def save_message(
        self,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_content = (content or "").strip()
        if not clean_content:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_store (namespace, thread_id, run_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    thread_id,
                    run_id,
                    role,
                    clean_content,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def list_messages(self, thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, run_id, role, content, metadata, created_at
                FROM conversation_store
                WHERE namespace = ? AND thread_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (self.namespace, thread_id, bounded_limit),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            try:
                parsed_meta = json.loads(row["metadata"] or "{}")
            except Exception:
                parsed_meta = {}
            output.append(
                {
                    "thread_id": row["thread_id"],
                    "run_id": row["run_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": parsed_meta,
                    "created_at": row["created_at"],
                }
            )
        return output


class PostgresConversationStore(BaseConversationStore):
    def __init__(self, conn_string: str, namespace: str = "default") -> None:
        self.conn_string = conn_string
        self.namespace = namespace

    def _connect(self):
        import psycopg

        return psycopg.connect(self.conn_string, autocommit=True)

    def setup(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_store (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_store_thread
                    ON conversation_store(namespace, thread_id, created_at DESC)
                    """
                )

    def save_message(
        self,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_content = (content or "").strip()
        if not clean_content:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_store (namespace, thread_id, run_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self.namespace,
                        thread_id,
                        run_id,
                        role,
                        clean_content,
                        json.dumps(metadata or {}),
                    ),
                )

    def list_messages(self, thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, run_id, role, content, metadata, created_at
                    FROM conversation_store
                    WHERE namespace = %s AND thread_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (self.namespace, thread_id, bounded_limit),
                )
                rows = cur.fetchall()

        output: List[Dict[str, Any]] = []
        for row in rows:
            metadata = row[4] if isinstance(row[4], dict) else {}
            output.append(
                {
                    "thread_id": row[0],
                    "run_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": metadata,
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                }
            )
        return output


def open_conversation_store(
    postgres_uri: str,
    sqlite_path: str,
    namespace: str = "default",
    fallback_sqlite: bool = True,
) -> StoreOpenResult:
    if postgres_uri:
        try:
            store = PostgresConversationStore(postgres_uri, namespace=namespace)
            store.setup()
            return StoreOpenResult(store=store, backend_label="postgres")
        except Exception as e:
            if not fallback_sqlite:
                raise RuntimeError(f"Failed to open Postgres store: {e}")
            print(f"[service] Failed to open Postgres store: {e}. Falling back to SQLite store.")

    store = SQLiteConversationStore(sqlite_path, namespace=namespace)
    store.setup()
    return StoreOpenResult(store=store, backend_label=sqlite_path)
