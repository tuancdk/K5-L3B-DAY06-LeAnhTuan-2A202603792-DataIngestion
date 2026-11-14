"""Real embedding provider: OpenAI text-embedding-3-small (1536 dimensions).

Documents and queries go through the same model with no prefix (OpenAI embedding
models are symmetric). Any API failure raises; there is no mock fallback.
"""
from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI

# USD per 1M input tokens, from OpenAI's pricing page; re-check before reporting cost.
PRICE_PER_MILLION_TOKENS = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
NATIVE_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenAIEmbedder:
    """OpenAI embeddings adapter that tracks token usage for cost reporting."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 batch_size: int = 128) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        if self.model_name not in NATIVE_DIMENSIONS:
            raise ValueError(f"unsupported embedding model: {self.model_name}")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing in starter/.env")
        self.dimensions = NATIVE_DIMENSIONS[self.model_name]
        self.batch_size = batch_size
        self.client = OpenAI(api_key=key, max_retries=5, timeout=60)
        self.total_tokens = 0
        self.api_calls = 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if any(not t or not t.strip() for t in texts):
            raise ValueError("cannot embed empty text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            self.api_calls += 1
            self.total_tokens += response.usage.total_tokens
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise RuntimeError(f"expected {len(batch)} vectors, got {len(data)}")
            for item in data:
                if len(item.embedding) != self.dimensions:
                    raise RuntimeError(f"expected {self.dimensions} dims, got {len(item.embedding)}")
                vectors.append(item.embedding)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

    @property
    def estimated_cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS[self.model_name]

    def usage(self) -> dict:
        return {"model": self.model_name, "dimensions": self.dimensions,
                "api_calls": self.api_calls, "total_tokens": self.total_tokens,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6)}
