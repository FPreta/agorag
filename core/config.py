"""Shared paths and constants for the core library.

The ETL pipeline writes the persistent store here and the inference pipeline
reads it back, so both sides agree on the layout through this one module.

The persistent store lives in $DATA_DIR (default: <repo>/data) and holds:
    publications.jsonl   one publication record per line (the source of truth)
    graph.json           the heterogeneous knowledge graph (typed nodes + edges)
    entities.json        flat name lists per type, for the linker prompt
    qdrant_db/           local persistent Qdrant collection of finding vectors
"""

from __future__ import annotations

import os
from pathlib import Path

# core/ -> repo root.
ROOT = Path(__file__).resolve().parent.parent

PUBLICATIONS_FILE = "publications.jsonl"
GRAPH_FILE = "graph.json"
ENTITIES_FILE = "entities.json"
QDRANT_DIR = "qdrant_db"

# Qdrant collection name, shared by the loader (write) and retriever (read).
COLLECTION = "findings"


def data_dir() -> Path:
    """The persistent-store directory ($DATA_DIR, or <repo>/data by default)."""
    return Path(os.getenv("DATA_DIR", ROOT / "data"))


def store_paths(base: str | Path | None = None) -> dict[str, Path]:
    """The four persistent-store paths under `base` (default: data_dir())."""
    root = Path(base) if base is not None else data_dir()
    return {
        "publications": root / PUBLICATIONS_FILE,
        "graph": root / GRAPH_FILE,
        "entities": root / ENTITIES_FILE,
        "qdrant": root / QDRANT_DIR,
    }
