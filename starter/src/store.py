"""Store and search chunk vectors in PostgreSQL + pgvector.

One table holds every strategy's chunks, keyed by (strategy, chunk_id). Search is an
exact cosine scan (no ANN index): the corpus is a few hundred chunks per strategy, so
results are deterministic and metadata filters never drop rows the way a filtered
approximate index can.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

TABLE = "chunks"
# Only these columns may be filtered; values are always passed as SQL parameters.
FILTERABLE = {"section", "document_id", "page", "report_period", "report_type"}


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL missing")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=5)

    def create_schema(self) -> None:
        """Create the table if missing; refuse to reuse a table built for other dims."""
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy      text    NOT NULL,
                    chunk_id      text    NOT NULL,
                    document_id   text    NOT NULL,
                    source        text    NOT NULL,
                    page          integer NOT NULL,
                    section       text,
                    report_period text,
                    report_type   text,
                    text          text    NOT NULL,
                    content_hash  text    NOT NULL,
                    metadata      jsonb   NOT NULL,
                    embedding     vector({self.dimensions}) NOT NULL,
                    updated_at    timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_filter_idx "
                         f"ON {TABLE} (strategy, section, document_id)")
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,)).fetchone()
            if row and row[0] != self.dimensions:
                raise RuntimeError(f"table {TABLE} stores {row[0]}-dim vectors, "
                                   f"embedder produces {self.dimensions}")

    def existing_hashes(self, strategy: str) -> dict[str, str]:
        """chunk_id -> content_hash already stored, to skip re-embedding unchanged chunks."""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT chunk_id, content_hash FROM {TABLE} WHERE strategy = %s",
                                (strategy,)).fetchall()
        return dict(rows)

    def delete_obsolete(self, strategy: str, document_ids: list[str], keep_ids: list[str]) -> int:
        """Remove rows of these documents/strategy that the current chunking no longer produces."""
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = ANY(%s) "
                               f"AND NOT (chunk_id = ANY(%s))", (strategy, document_ids, keep_ids))
            return cur.rowcount

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        """Write chunks idempotently; stable IDs + ON CONFLICT mean no duplicates.

        Pass only new/changed chunks to avoid re-embedding; call ``delete_obsolete``
        with the full current ID list to drop chunks of edited documents.
        """
        if not chunks:
            return {"written": 0}
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"every vector must have {self.dimensions} dimensions")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in batch")

        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page,
                 c.metadata.get("section"), c.metadata.get("report_period"),
                 c.metadata.get("report_type"), c.text, c.metadata["content_hash"],
                 Jsonb(c.metadata), _vector_literal(v)) for c, v in zip(chunks, vectors)]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, section,
                                     report_period, report_type, text, content_hash, metadata,
                                     embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                    page = EXCLUDED.page, section = EXCLUDED.section,
                    report_period = EXCLUDED.report_period, report_type = EXCLUDED.report_type,
                    text = EXCLUDED.text, content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,
                    updated_at = now()
                WHERE {TABLE}.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                   OR {TABLE}.metadata IS DISTINCT FROM EXCLUDED.metadata""", rows)
        return {"written": len(rows)}

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {TABLE} WHERE strategy = %s",
                                (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Exact cosine search within one strategy, filtered in SQL before ranking."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query vector must have {self.dimensions} dimensions")
        where, params = ["strategy = %s"], [strategy]
        for column, value in (filters or {}).items():
            if column not in FILTERABLE:
                raise ValueError(f"filter on '{column}' not allowed; use one of {sorted(FILTERABLE)}")
            if isinstance(value, (list, tuple, set)):
                where.append(f"{column} = ANY(%s)")
                params.append(list(value))
            else:
                where.append(f"{column} = %s")
                params.append(value)
        vec = _vector_literal(query_vector)
        sql = (f"SELECT chunk_id, document_id, source, page, section, text, metadata, "
               f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [vec, *params, vec, top_k]).fetchall()
        return [{"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
                 "section": r[4], "text": r[5],
                 "metadata": r[6] if isinstance(r[6], dict) else json.loads(r[6]),
                 "score": round(float(r[7]), 6)} for r in rows]
