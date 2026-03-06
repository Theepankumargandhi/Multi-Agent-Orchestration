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
        user_id: str | None,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def list_messages(
        self,
        thread_id: str,
        limit: int = 50,
        user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_threads(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def create_user(self, user_id: str, password_hash: str) -> bool:
        raise NotImplementedError

    def get_user_password_hash(self, user_id: str) -> str | None:
        raise NotImplementedError

    def save_hitl_event(
        self,
        user_id: str | None,
        thread_id: str | None,
        query: str,
        decision: str,
        reason: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def list_hitl_events(
        self,
        user_id: str,
        limit: int = 50,
        thread_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SQLiteConversationStore(BaseConversationStore):
    def __init__(self, db_path: str, namespace: str = "default") -> None:
        self.db_path = db_path
        self.namespace = namespace

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        add_column_ddl: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = {row["name"] for row in rows}
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {add_column_ddl}")

    def setup(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Backward-compat migration for pre-user-id schema.
            self._ensure_column(conn, "conversation_store", "user_id", "TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_store_thread
                ON conversation_store(namespace, thread_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_store_thread_user
                ON conversation_store(namespace, user_id, thread_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(namespace, user_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hitl_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hitl_events_user
                ON hitl_events(namespace, user_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hitl_events_thread
                ON hitl_events(namespace, user_id, thread_id, created_at DESC)
                """
            )
            conn.commit()

    def save_message(
        self,
        user_id: str | None,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_content = (content or "").strip()
        if not clean_content:
            return
        clean_user_id = (user_id or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_store (namespace, user_id, thread_id, run_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    clean_user_id,
                    thread_id,
                    run_id,
                    role,
                    clean_content,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def list_messages(
        self,
        thread_id: str,
        limit: int = 50,
        user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        clean_user_id = (user_id or "").strip()
        with self._connect() as conn:
            if clean_user_id:
                rows = conn.execute(
                    """
                    SELECT user_id, thread_id, run_id, role, content, metadata, created_at
                    FROM conversation_store
                    WHERE namespace = ? AND user_id = ? AND thread_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (self.namespace, clean_user_id, thread_id, bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT user_id, thread_id, run_id, role, content, metadata, created_at
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
                    "user_id": row["user_id"] or "",
                    "thread_id": row["thread_id"],
                    "run_id": row["run_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": parsed_meta,
                    "created_at": row["created_at"],
                }
            )
        return output

    def list_threads(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return []
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, MAX(created_at) AS last_message_at, COUNT(*) AS message_count
                FROM conversation_store
                WHERE namespace = ? AND user_id = ?
                GROUP BY thread_id
                ORDER BY last_message_at DESC
                LIMIT ?
                """,
                (self.namespace, clean_user_id, bounded_limit),
            ).fetchall()

            output: List[Dict[str, Any]] = []
            for row in rows:
                preview_row = conn.execute(
                    """
                    SELECT content
                    FROM conversation_store
                    WHERE namespace = ? AND user_id = ? AND thread_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (self.namespace, clean_user_id, row["thread_id"]),
                ).fetchone()
                output.append(
                    {
                        "thread_id": row["thread_id"],
                        "message_count": int(row["message_count"] or 0),
                        "last_message_at": row["last_message_at"],
                        "last_message_preview": (preview_row["content"] if preview_row else "") or "",
                    }
                )
        return output

    def create_user(self, user_id: str, password_hash: str) -> bool:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            raise ValueError("user_id is required")
        if not password_hash:
            raise ValueError("password_hash is required")
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (namespace, user_id, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (self.namespace, clean_user_id, password_hash),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user_password_hash(self, user_id: str) -> str | None:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT password_hash
                FROM users
                WHERE namespace = ? AND user_id = ?
                LIMIT 1
                """,
                (self.namespace, clean_user_id),
            ).fetchone()
        if not row:
            return None
        return str(row["password_hash"] or "")

    def save_hitl_event(
        self,
        user_id: str | None,
        thread_id: str | None,
        query: str,
        decision: str,
        reason: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_query = (query or "").strip()
        clean_decision = (decision or "").strip().lower()
        if not clean_query or clean_decision not in {"approved", "rejected"}:
            return
        clean_user_id = (user_id or "").strip()
        clean_thread_id = (thread_id or "").strip()
        clean_reason = (reason or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO hitl_events (namespace, user_id, thread_id, query, decision, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    clean_user_id,
                    clean_thread_id,
                    clean_query,
                    clean_decision,
                    clean_reason,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def list_hitl_events(
        self,
        user_id: str,
        limit: int = 50,
        thread_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return []
        clean_thread_id = (thread_id or "").strip()
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            if clean_thread_id:
                rows = conn.execute(
                    """
                    SELECT user_id, thread_id, query, decision, reason, metadata, created_at
                    FROM hitl_events
                    WHERE namespace = ? AND user_id = ? AND thread_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (self.namespace, clean_user_id, clean_thread_id, bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT user_id, thread_id, query, decision, reason, metadata, created_at
                    FROM hitl_events
                    WHERE namespace = ? AND user_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (self.namespace, clean_user_id, bounded_limit),
                ).fetchall()

        output: List[Dict[str, Any]] = []
        for row in rows:
            try:
                parsed_meta = json.loads(row["metadata"] or "{}")
            except Exception:
                parsed_meta = {}
            output.append(
                {
                    "user_id": row["user_id"] or "",
                    "thread_id": row["thread_id"] or "",
                    "query": row["query"],
                    "decision": row["decision"],
                    "reason": row["reason"] or "",
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
                        user_id TEXT NOT NULL DEFAULT '',
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
                    ALTER TABLE conversation_store
                    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_store_thread
                    ON conversation_store(namespace, thread_id, created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_store_thread_user
                    ON conversation_store(namespace, user_id, thread_id, created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(namespace, user_id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hitl_events (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        user_id TEXT NOT NULL DEFAULT '',
                        thread_id TEXT NOT NULL DEFAULT '',
                        query TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        reason TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hitl_events_user
                    ON hitl_events(namespace, user_id, created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_hitl_events_thread
                    ON hitl_events(namespace, user_id, thread_id, created_at DESC)
                    """
                )

    def save_message(
        self,
        user_id: str | None,
        thread_id: str,
        run_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_content = (content or "").strip()
        if not clean_content:
            return
        clean_user_id = (user_id or "").strip()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_store (namespace, user_id, thread_id, run_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self.namespace,
                        clean_user_id,
                        thread_id,
                        run_id,
                        role,
                        clean_content,
                        json.dumps(metadata or {}),
                    ),
                )

    def list_messages(
        self,
        thread_id: str,
        limit: int = 50,
        user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        clean_user_id = (user_id or "").strip()
        with self._connect() as conn:
            with conn.cursor() as cur:
                if clean_user_id:
                    cur.execute(
                        """
                        SELECT user_id, thread_id, run_id, role, content, metadata, created_at
                        FROM conversation_store
                        WHERE namespace = %s AND user_id = %s AND thread_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (self.namespace, clean_user_id, thread_id, bounded_limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT user_id, thread_id, run_id, role, content, metadata, created_at
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
            raw_meta = row[5]
            if isinstance(raw_meta, dict):
                metadata = raw_meta
            elif isinstance(raw_meta, str):
                try:
                    metadata = json.loads(raw_meta)
                except Exception:
                    metadata = {}
            else:
                metadata = {}
            output.append(
                {
                    "user_id": row[0] or "",
                    "thread_id": row[1],
                    "run_id": row[2],
                    "role": row[3],
                    "content": row[4],
                    "metadata": metadata,
                    "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                }
            )
        return output

    def list_threads(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return []
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, MAX(created_at) AS last_message_at, COUNT(*) AS message_count
                    FROM conversation_store
                    WHERE namespace = %s AND user_id = %s
                    GROUP BY thread_id
                    ORDER BY MAX(created_at) DESC
                    LIMIT %s
                    """,
                    (self.namespace, clean_user_id, bounded_limit),
                )
                rows = cur.fetchall()

                output: List[Dict[str, Any]] = []
                for thread_id, last_message_at, message_count in rows:
                    cur.execute(
                        """
                        SELECT content
                        FROM conversation_store
                        WHERE namespace = %s AND user_id = %s AND thread_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        (self.namespace, clean_user_id, thread_id),
                    )
                    preview_row = cur.fetchone()
                    output.append(
                        {
                            "thread_id": thread_id,
                            "message_count": int(message_count or 0),
                            "last_message_at": (
                                last_message_at.isoformat()
                                if hasattr(last_message_at, "isoformat")
                                else str(last_message_at)
                            ),
                            "last_message_preview": (preview_row[0] if preview_row else "") or "",
                        }
                    )
        return output

    def create_user(self, user_id: str, password_hash: str) -> bool:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            raise ValueError("user_id is required")
        if not password_hash:
            raise ValueError("password_hash is required")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (namespace, user_id, password_hash)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (namespace, user_id) DO NOTHING
                    RETURNING user_id
                    """,
                    (self.namespace, clean_user_id, password_hash),
                )
                row = cur.fetchone()
        return row is not None

    def get_user_password_hash(self, user_id: str) -> str | None:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT password_hash
                    FROM users
                    WHERE namespace = %s AND user_id = %s
                    LIMIT 1
                    """,
                    (self.namespace, clean_user_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return str(row[0] or "")

    def save_hitl_event(
        self,
        user_id: str | None,
        thread_id: str | None,
        query: str,
        decision: str,
        reason: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        clean_query = (query or "").strip()
        clean_decision = (decision or "").strip().lower()
        if not clean_query or clean_decision not in {"approved", "rejected"}:
            return
        clean_user_id = (user_id or "").strip()
        clean_thread_id = (thread_id or "").strip()
        clean_reason = (reason or "").strip()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hitl_events (namespace, user_id, thread_id, query, decision, reason, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self.namespace,
                        clean_user_id,
                        clean_thread_id,
                        clean_query,
                        clean_decision,
                        clean_reason,
                        json.dumps(metadata or {}),
                    ),
                )

    def list_hitl_events(
        self,
        user_id: str,
        limit: int = 50,
        thread_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        clean_user_id = (user_id or "").strip()
        if not clean_user_id:
            return []
        clean_thread_id = (thread_id or "").strip()
        bounded_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            with conn.cursor() as cur:
                if clean_thread_id:
                    cur.execute(
                        """
                        SELECT user_id, thread_id, query, decision, reason, metadata, created_at
                        FROM hitl_events
                        WHERE namespace = %s AND user_id = %s AND thread_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (self.namespace, clean_user_id, clean_thread_id, bounded_limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT user_id, thread_id, query, decision, reason, metadata, created_at
                        FROM hitl_events
                        WHERE namespace = %s AND user_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (self.namespace, clean_user_id, bounded_limit),
                    )
                rows = cur.fetchall()

        output: List[Dict[str, Any]] = []
        for row in rows:
            raw_meta = row[5]
            if isinstance(raw_meta, dict):
                metadata = raw_meta
            elif isinstance(raw_meta, str):
                try:
                    metadata = json.loads(raw_meta)
                except Exception:
                    metadata = {}
            else:
                metadata = {}
            output.append(
                {
                    "user_id": row[0] or "",
                    "thread_id": row[1] or "",
                    "query": row[2],
                    "decision": row[3],
                    "reason": row[4] or "",
                    "metadata": metadata,
                    "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
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
