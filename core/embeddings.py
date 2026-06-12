"""Configurable embedding service — local sentence-transformers or OpenAI.

Used both at build time (core.etl.loader) and at query time (core.inference.
retriever), so the same model embeds documents and queries.

Configuration (env vars, see .env.example):
    EMBEDDING_PROVIDER = local | openai     (default: openai)
    EMBEDDING_MODEL    = text-embedding-3-small   (sentence-transformers name, or OpenAI model)
    EMBEDDING_API_KEY  = ...                (openai only, falls back to OPENAI_API_KEY if not set)
    EMBEDDING_BASE_URL = https://...        (openai only, optional override)

The OpenAI path is a thin wrapper around the official ``openai`` SDK. Set
EMBEDDING_BASE_URL only to target an OpenAI-compatible endpoint (e.g. Azure or a
local gateway); leave it unset to use api.openai.com.
"""

from __future__ import annotations

import os


class EmbeddingService:
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._model = None
        self._client = None

        if provider == "local":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        elif provider == "openai":
            from openai import OpenAI

            if not api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected 'local' or 'openai')"
            )

    @classmethod
    def from_env(cls) -> "EmbeddingService":
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]

        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        # Preserve input order regardless of how the API returns it.
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
