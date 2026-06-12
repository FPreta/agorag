"""ETL step 3 — persist the store and load the finding vectors into Qdrant.

Writes the publications/graph/entities files and embeds the chunks into the local
persistent Qdrant collection under data/qdrant_db/. Qdrant runs in local mode (no
server, no Docker); only one process may open the directory at a time, so don't
run the ETL pipeline while the app is up.

Embedding uses the same configurable EmbeddingService the inference side uses at
query time, so document and query vectors stay compatible.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .. import config
from ..embeddings import EmbeddingService

BATCH = 64


class Loader:
    def __init__(
        self, data_dir: str | Path | None = None, embeddings: EmbeddingService | None = None
    ):
        base = Path(data_dir) if data_dir is not None else config.data_dir()
        base.mkdir(parents=True, exist_ok=True)
        self.paths = config.store_paths(base)
        self._embeddings = embeddings  # built lazily (downloads the local model)

    @property
    def embeddings(self) -> EmbeddingService:
        if self._embeddings is None:
            self._embeddings = EmbeddingService.from_env()
        return self._embeddings

    # --- files ----------------------------------------------------------

    def save_publications(self, pubs: list[dict]) -> None:
        with self.paths["publications"].open("w", encoding="utf-8") as f:
            for pub in pubs:
                f.write(json.dumps(pub, ensure_ascii=False) + "\n")
        print(f"[loader] wrote {len(pubs)} publications -> {self.paths['publications']}")

    def save_graph(self, graph: dict, entities: dict) -> None:
        self.paths["graph"].write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.paths["entities"].write_text(
            json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        n_pub = sum(1 for n in graph["nodes"] if n["type"] == "publication")
        print(
            f"[loader] wrote graph ({len(graph['nodes'])} nodes / {len(graph['edges'])} edges, "
            f"{n_pub} publications) and entities "
            f"({len(entities['topics'])} topics, {len(entities['regions'])} regions)"
        )

    # --- Qdrant ---------------------------------------------------------

    def collection_exists(self) -> bool:
        """Whether the Qdrant collection has already been built."""
        if not self.paths["qdrant"].exists():
            return False
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(self.paths["qdrant"]))
        try:
            return client.collection_exists(config.COLLECTION)
        finally:
            client.close()  # release the directory lock before load_chunks reopens it

    def load_chunks(self, chunks: list[dict], fresh: bool = True) -> None:
        """Embed `chunks` and upsert them into the Qdrant collection.

        `fresh=True` (re)creates the collection from scratch (full rebuild);
        `fresh=False` appends into the existing collection (incremental add). When
        the collection doesn't exist yet, an incremental load is promoted to fresh."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            print("[loader] no chunks to embed — skipping Qdrant load")
            return

        client = QdrantClient(path=str(self.paths["qdrant"]))
        emb = self.embeddings
        print(
            f"[loader] embedding {len(chunks)} chunks with provider={emb.provider} "
            f"model={emb.model_name} (fresh={fresh}) ..."
        )
        try:
            # Embed the first batch to discover the vector dimension.
            first_vectors = emb.embed([c["text"] for c in chunks[:BATCH]])
            dim = len(first_vectors[0])

            if fresh or not client.collection_exists(config.COLLECTION):
                client.recreate_collection(
                    collection_name=config.COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

            def upsert(batch_chunks: list[dict], vectors: list[list[float]]) -> None:
                points = [
                    PointStruct(id=str(uuid.uuid4()), vector=vec, payload=chunk)
                    for chunk, vec in zip(batch_chunks, vectors)
                ]
                client.upsert(collection_name=config.COLLECTION, points=points)

            upsert(chunks[:BATCH], first_vectors)
            done = len(first_vectors)
            for i in range(BATCH, len(chunks), BATCH):
                batch = chunks[i : i + BATCH]
                upsert(batch, emb.embed([c["text"] for c in batch]))
                done += len(batch)
                print(f"[loader]   embedded {done}/{len(chunks)}")
        finally:
            client.close()  # release the directory lock for the next reader/run

        print(f"[loader] collection '{config.COLLECTION}' ({dim}-dim) at {self.paths['qdrant']}")
