"""Shared node-id helpers, used across ETL (graph building) and inference."""

from __future__ import annotations

import re
import unicodedata

TYPE_PREFIX = {"publication": "pub", "topic": "topic", "region": "region", "author": "author"}


def slugify(value: str) -> str:
    """Lowercase ASCII slug. 'Coal phase-out' -> 'coal-phase-out'."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value) or "unknown"


def node_id(node_type: str, name: str) -> str:
    return f"{TYPE_PREFIX[node_type]}::{slugify(name)}"


def canonical_pub_ids(raw_id: str | None, label: str | None) -> list[str]:
    """Candidate canonical publication node ids for an LLM-authored node.

    the answer model drifts from our id convention: it prefixes with the full type name
    ("publication::…" instead of "pub::…"), renders apostrophes/quotes differently
    from the source title (straight ' vs curly '), and sometimes appends the format
    ("… (Study)"). Its `id` therefore can't be trusted. Slugifying the *label*
    recovers the canonical id across all of these (slugify drops punctuation and
    folds case/whitespace), so callers should match against these candidates rather
    than the raw id. Ordered most- to least-trusted; the raw id comes first so an
    already-canonical graph is a no-op."""
    cands: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in cands:
            cands.append(value)

    add(raw_id)
    if label:
        add(node_id("publication", label))
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
        if stripped and stripped != label:
            add(node_id("publication", stripped))
    return cands
