"""Semantic retrieval — embed the query, fetch the top-k findings from Qdrant.

Wraps the embedding service and the local persistent Qdrant collection (the same
data/qdrant_db/ the ETL loader writes). Qdrant runs in local mode; only one
process may open the directory at a time, so don't run the ETL pipeline while the
app is up.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from .. import config
from ..embeddings import EmbeddingService


class Retriever:
    def __init__(self, qdrant_path: str | Path, embeddings: EmbeddingService | None = None):

        self.client = QdrantClient(path=str(qdrant_path))
        self.collection = config.COLLECTION
        self.embeddings = embeddings or EmbeddingService.from_env()

    @classmethod
    def from_data_dir(
        cls, data_dir: str | Path | None = None, embeddings: EmbeddingService | None = None
    ) -> "Retriever":
        paths = config.store_paths(data_dir)
        return cls(paths["qdrant"], embeddings=embeddings)

    def retrieve(self, query: str, top_k: int = 15) -> list[dict]:
        """Embed `query` and return the top-k matching findings as flat dicts:
        {score, text, publication_id, publication_title, topics, regions, ...}."""
        return self.search(self.embeddings.embed_one(query), top_k=top_k)

    def search(self, query_vector: list[float], top_k: int = 15) -> list[dict]:
        """Return the top-k findings for an already-embedded query vector."""
        # query_points() validates the query type: a raw dense vector must be an
        # np.ndarray/tensor, not a plain Python list (which it treats as an
        # inference input and rejects).
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [{"score": h.score, **(h.payload or {})} for h in hits.points]
