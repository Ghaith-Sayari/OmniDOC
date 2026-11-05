"""
MetadataStore — lightweight SQLite-backed store for document-level
metadata (source path, ingest timestamp, format, page count, status).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnidoc.utils import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("data/processed/metadata.db")


class MetadataStore:
    """
    Tracks every document that has passed through the ingestion pipeline.

    Schema
    ------
    documents (id TEXT PK, source_path TEXT, format TEXT, page_count INT,
               ingest_ts TEXT, status TEXT, extra JSON)
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_table()

    def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or replace a document metadata record."""
        import json

        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, source_path, format, page_count, ingest_ts, status, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                metadata.get("source", ""),
                metadata.get("format", ""),
                metadata.get("page_count", 0),
                datetime.now(tz=timezone.utc).isoformat(),
                metadata.get("status", "indexed"),
                json.dumps({k: v for k, v in metadata.items() if k not in {"source", "format", "page_count", "status"}}),
            ),
        )
        self._conn.commit()

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single document record by *doc_id*."""
        cur = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_all(self) -> list[dict[str, Any]]:
        """Return all indexed document records."""
        cur = self._conn.execute("SELECT * FROM documents ORDER BY ingest_ts DESC")
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_path TEXT,
                format TEXT,
                page_count INTEGER,
                ingest_ts TEXT,
                status TEXT,
                extra TEXT
            )"""
        )
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        import json

        keys = ["id", "source_path", "format", "page_count", "ingest_ts", "status", "extra"]
        d = dict(zip(keys, row))
        d["extra"] = json.loads(d.get("extra") or "{}")
        return d
