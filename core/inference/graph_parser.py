"""Answer generation + knowledge-graph parsing (the answer-model call).

the answer model receives the retrieved findings, the entity-linked publications and the
current graph, then streams a cited answer followed by `|||GRAPH_UPDATE|||` and the
COMPLETE new graph state (full replacement, owned by the answer model). This module streams the
answer and validates the graph into the final state the backend renders.

    stream, result = parser.parse(query, history, hits, entity_pubs, pub_by_node_id, current_graph)
    for evt in stream:                 # {"type": "token", "text": ...} / {"type": "graph_loading"}
        ...
    result.graph                       # the validated {nodes, edges}

The graph is validated, then its edges are rebuilt from ground truth: the answer model draws
publication->hub edges by free association (a fossil-gas study wired to "Hydrogen"),
so the entire edge list is discarded and each publication is reconnected only to the
hubs it's actually tagged with (see GraphService.real_hub_ids).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..graph_store import GraphService
from ..ids import node_id
from . import core_llm
from .core_llm import LLMService

NODE_TYPES = {"publication", "topic", "region"}
EDGE_TYPES = {"has_topic", "has_region"}


# --- graph validation -------------------------------------------------------


def valid_hub_ids(entities: dict | None) -> set[str]:
    """Canonical node ids of every real topic/region in the taxonomy.

    Used to reject hub nodes the answer model invents: only ids in this set survive validation."""
    if not entities:
        return set()
    ids: set[str] = set()
    for etype in ("topic", "region"):
        for name in entities.get(f"{etype}s", []) or []:
            ids.add(node_id(etype, name))
    return ids


def _clean_graph(
    raw: dict, valid_hubs: set[str] | None = None, prune_orphans: bool = False
) -> dict:
    """Validate the answer model's graph: well-formed nodes and edges that reference real nodes.

    When `valid_hubs` is given, topic/region nodes whose id isn't a real
    taxonomy id are dropped (the answer model hallucinates hub slugs). Publication nodes are
    never filtered this way — they're keyed by the bare ids the answer model copies from the
    context, not by the taxonomy.

    `prune_orphans` drops publications left with no hub edge. Callers that follow up
    with enforce_real_edges (which rebuilds edges from ground truth and keeps every
    publication) pass False, so a publication isn't lost just because the answer model only wired
    it to a hub that didn't survive validation."""
    nodes_by_id: dict[str, dict] = {}
    for n in raw.get("nodes", []) or []:
        nid, ntype, label = n.get("id"), n.get("type"), n.get("label")
        if not (isinstance(nid, str) and ntype in NODE_TYPES and isinstance(label, str)):
            continue
        if valid_hubs is not None and ntype != "publication" and nid not in valid_hubs:
            continue  # hallucinated hub — drop it
        node = {"id": nid, "type": ntype, "label": label}
        if n.get("role"):
            node["role"] = n["role"]
        nodes_by_id[nid] = node

    edges: list[dict] = []
    seen: set[tuple] = set()
    connected_pubs: set[str] = set()
    for e in raw.get("edges", []) or []:
        src, tgt, etype = e.get("source"), e.get("target"), e.get("type")
        if etype not in EDGE_TYPES or src not in nodes_by_id or tgt not in nodes_by_id:
            continue
        key = (src, tgt, etype)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": src, "target": tgt, "type": etype})
        if nodes_by_id[src]["type"] == "publication":
            connected_pubs.add(src)

    if prune_orphans:
        nodes = [
            n
            for n in nodes_by_id.values()
            if n["type"] != "publication" or n["id"] in connected_pubs
        ]
    else:
        nodes = list(nodes_by_id.values())
    return {"nodes": nodes, "edges": edges}


def parse_graph(
    tail: str,
    fallback: dict | None,
    valid_hubs: set[str] | None = None,
    prune_orphans: bool = True,
) -> dict:
    """Parse + validate the graph JSON after the delimiter.

    `valid_hubs` (see valid_hub_ids) drops hallucinated topic/region nodes.
    `prune_orphans` (see _clean_graph) — pass False when enforce_real_edges follows.
    On any failure, fall back to the previous graph state (or an empty graph)."""
    fallback = fallback or {"nodes": [], "edges": []}
    text = (tail or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return fallback
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return fallback
    cleaned = _clean_graph(raw, valid_hubs, prune_orphans)
    return cleaned if cleaned["nodes"] else fallback


def enforce_real_edges(graph: dict, real_hubs_of) -> dict:
    """Replace the answer model's hub edges with ones backed by the real publication metadata.

    the answer model draws publication->hub edges by semantic association over the retrieved
    chunks, so a publication routinely ends up wired to topics it was never tagged
    with. Node validation can't catch this — the nodes are real, only the connections
    are invented.

    So we discard the answer model's entire edge list and reconnect each publication to just the
    hub nodes ALREADY in the graph that ground truth supports. `real_hubs_of(node)`
    returns the canonical hub ids a publication actually carries. No new nodes are
    added (the answer model still curates *which* nodes appear). Every publication node is kept —
    one with no real connection to a drawn hub simply floats edgeless — but hub nodes
    left with no publications are pruned (they were hallucinated associations)."""
    by_id = {n["id"]: n for n in graph.get("nodes", [])}
    etype_for = {"topic": "has_topic", "region": "has_region"}

    edges: list[dict] = []
    used_hubs: set[str] = set()
    for n in graph.get("nodes", []):
        if n.get("type") != "publication":
            continue
        for hid in sorted(real_hubs_of(n)):
            hub = by_id.get(hid)
            if not hub or hub.get("type") not in etype_for:
                continue  # hub not drawn in this graph — don't invent a node for it
            edges.append({"source": n["id"], "target": hid, "type": etype_for[hub["type"]]})
            used_hubs.add(hid)

    nodes = [
        n for n in graph.get("nodes", []) if n.get("type") == "publication" or n["id"] in used_hubs
    ]
    return {"nodes": nodes, "edges": edges}


def fallback_answer(retrieval_hits: list[dict]) -> str:
    """When the answer model is unavailable: surface the retrieved publications, unsynthesized."""
    if not retrieval_hits:
        return (
            "I couldn't reach the synthesis model, and no relevant publications were "
            "found for this query. Please try rephrasing."
        )
    seen: set[str] = set()
    lines = [
        "I couldn't reach the synthesis model, but here are the most relevant "
        "publications from Agora's archive:\n"
    ]
    for hit in retrieval_hits:
        title = hit.get("publication_title", "Unknown")
        if title in seen:
            continue
        seen.add(title)
        lines.append(f"- [{title}]")
    return "\n".join(lines)


# --- the answer-model call ----------------------------------------------------------


@dataclass
class ParseResult:
    """Filled once the answer + graph stream is fully consumed."""

    graph: dict = field(default_factory=lambda: {"nodes": [], "edges": []})


class GraphParser:
    def __init__(self, llm: LLMService, graph: GraphService, entities: dict):
        self.llm = llm
        self.graph = graph
        self.entities = entities

    def parse(
        self,
        query: str,
        history: list,
        retrieval_hits: list[dict],
        entity_pubs: list[dict],
        pub_by_node_id: dict[str, dict],
        current_graph: dict | None,
    ):
        """Return (stream, result).

        `stream` yields event dicts: {"type": "token", "text": ...} for answer text,
        then a single {"type": "graph_loading"} when the answer is done and the answer model has
        begun emitting graph JSON (no further tokens). Once `stream` is exhausted,
        `result.graph` holds the validated, edge-rebuilt graph.

        Exceptions from the underlying stream propagate to the caller (the pipeline
        catches them to serve a fallback answer + fallback graph)."""
        result = ParseResult()
        split = core_llm.SplitStreamer(core_llm.GRAPH_DELIMITER)
        context_block = core_llm.assemble_context(
            retrieval_hits, entity_pubs, pub_by_node_id, current_graph
        )

        def stream():
            deltas = self.llm.stream(
                model=core_llm.ANSWER_MODEL,
                system=core_llm.answer_system_prompt(self.entities),
                messages=core_llm.build_answer_messages(query, context_block, history),
                max_tokens=core_llm.ANSWER_MAX_TOKENS,
            )
            graph_loading_sent = False
            for delta in deltas:
                chunk = split.feed(delta)
                if chunk:
                    yield {"type": "token", "text": chunk}
                elif split.hit and not graph_loading_sent:
                    # Answer text is done; the answer model is now emitting graph JSON (no
                    # client-facing tokens), so signal the graph spinner.
                    graph_loading_sent = True
                    yield {"type": "graph_loading"}
            tail = split.flush()
            if tail:
                yield {"type": "token", "text": tail}

            parsed = parse_graph(
                split.tail,
                fallback=current_graph,
                valid_hubs=valid_hub_ids(self.entities),
                prune_orphans=False,
            )
            result.graph = enforce_real_edges(parsed, self.graph.real_hub_ids)

        return stream(), result
