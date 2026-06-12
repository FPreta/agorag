"""Entity linking — stream the linker acknowledgment, then resolve graph results.

A thin wrapper over the LINKER_MODEL call: it streams the one-sentence acknowledgment
to the user and, once the stream completes, parses the `|||ENTITIES|||` JSON tail into
the known topics/regions the query references and expands them into their
connected publications via the graph.

    stream, result = linker.link(query, history)
    for chunk in stream:        # the acknowledgment, token by token
        ...
    result.linked              # {"topics": [...], "regions": [...]}
    result.entity_pubs         # publication nodes connected to those entities
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..graph_store import GraphService
from . import core_llm
from .core_llm import LLMService


def parse_entities(tail: str) -> dict:
    """Parse the JSON after the delimiter; always return the three keys."""
    empty = {"topics": [], "regions": []}
    text = (tail or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return empty
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logging.info("Failed to parse entities JSON: %s", text)
        return empty
    return {
        "topics": list(data.get("topics", []) or []),
        "regions": list(data.get("regions", []) or []),
    }


@dataclass
class LinkResult:
    """Filled once the acknowledgment stream is fully consumed."""

    linked: dict = field(default_factory=lambda: {"topics": [], "regions": []})
    entity_pubs: list = field(default_factory=list)


class Linker:
    def __init__(self, llm: LLMService, graph: GraphService, entities: dict):
        self.llm = llm
        self.graph = graph
        self.entities = entities

    def link(self, query: str, history: list):
        """Return (stream, result): iterate `stream` for the acknowledgment text;
        afterwards `result.linked` / `result.entity_pubs` hold the graph results.

        Exceptions from the underlying stream (e.g. openai.AuthenticationError)
        propagate to the caller, which decides how to surface them."""
        result = LinkResult()
        split = core_llm.SplitStreamer(core_llm.ENTITY_DELIMITER)

        def stream():
            deltas = self.llm.stream(
                model=core_llm.LINKER_MODEL,
                system=core_llm.linker_system_prompt(self.entities),
                messages=core_llm.build_linker_messages(query, history),
                max_tokens=core_llm.LINKER_MAX_TOKENS,
            )
            for delta in deltas:
                chunk = split.feed(delta)
                if chunk:
                    yield chunk
            tail = split.flush()
            if tail:
                yield tail
            result.linked = parse_entities(split.tail)
            result.entity_pubs = self.graph.publications_for_named_entities(result.linked)

        return stream(), result
