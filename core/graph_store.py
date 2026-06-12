"""In-memory access to the pre-computed knowledge graph (networkx).

Loaded once from data/graph.json (written by the ETL pipeline). Used at inference
time to:
  - expand linked entities into their connected publications (extra candidates
    for the answer model beyond what Qdrant returned), and
  - produce a deterministic fallback subgraph when the answer model is
    unavailable or returns malformed JSON, and
  - repair answer-model-authored graphs by reconnecting publications to the hubs they're
    really tagged with (real_hub_ids).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .ids import canonical_pub_ids, node_id

ENTITY_TYPES = ("topic", "region")
EDGE_FOR_TYPE = {"topic": "has_topic", "region": "has_region"}


class GraphService:
    def __init__(self, graph_path: str | Path):
        data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        self.G = nx.DiGraph()
        for node in data["nodes"]:
            self.G.add_node(node["id"], **node)
        for edge in data["edges"]:
            self.G.add_edge(edge["source"], edge["target"], type=edge["type"])

        # Ground-truth indexes for repairing answer-model-authored graphs:
        #   canonical pub id -> the hub ids it's really tagged with, and
        #   exact title -> canonical pub id (the answer model's ids drift, titles don't).
        self._hub_ids_by_pub: dict[str, set[str]] = {}
        self._pub_id_by_title: dict[str, str] = {}
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == "publication":
                title = (attrs.get("title") or "").strip().lower()
                if title:
                    self._pub_id_by_title[title] = nid
        for src, tgt, _ in self.G.edges(data=True):
            if (
                self.G.nodes[src].get("type") == "publication"
                and self.G.nodes[tgt].get("type") in ENTITY_TYPES
            ):
                self._hub_ids_by_pub.setdefault(src, set()).add(tgt)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def label_for(node: dict) -> str:
        return node.get("title") or node.get("name") or node["id"]

    def real_hub_ids(self, pub_node: dict) -> set[str]:
        """Canonical hub ids a publication is actually tagged with in the real graph.

        Resolves an answer-model-authored publication node back to ground truth. the answer model's id and
        label both drift (wrong prefix, apostrophe variants, a "(Study)" suffix), so we
        try the canonical id candidates derived from the label before falling back to an
        exact-title lookup. Returns an empty set for anything unrecognised."""
        for cand in canonical_pub_ids(pub_node.get("id"), pub_node.get("label")):
            if cand in self._hub_ids_by_pub:
                return self._hub_ids_by_pub[cand]
        canon = self._pub_id_by_title.get((pub_node.get("label") or "").strip().lower())
        return self._hub_ids_by_pub.get(canon, set())

    def _entity_ids(self, linked: dict) -> list[str]:
        ids: list[str] = []
        for etype in ENTITY_TYPES:
            for name in linked.get(f"{etype}s", []) or []:
                nid = node_id(etype, name)
                if nid in self.G:
                    ids.append(nid)
        return ids

    # --- public ----------------------------------------------------------

    def publications_for_named_entities(self, linked: dict) -> list[dict]:
        """All publication nodes connected to any of the linked entities."""
        pubs: dict[str, dict] = {}
        for eid in self._entity_ids(linked):
            for source, _ in self.G.in_edges(eid):
                node = self.G.nodes[source]
                if node.get("type") == "publication":
                    pubs[source] = node
        return list(pubs.values())

    def subgraph_for_named_entities(self, linked: dict) -> dict:
        """Deterministic {nodes, edges} subgraph (label format) for fallback use."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        seen_edge: set[tuple] = set()

        def add(node: dict) -> None:
            nodes[node["id"]] = {
                "id": node["id"],
                "type": node["type"],
                "label": self.label_for(node),
            }

        for eid in self._entity_ids(linked):
            add(self.G.nodes[eid])
            for source, target, data in self.G.in_edges(eid, data=True):
                src_node = self.G.nodes[source]
                if src_node.get("type") != "publication":
                    continue
                add(src_node)
                key = (source, target, data["type"])
                if key not in seen_edge:
                    seen_edge.add(key)
                    edges.append({"source": source, "target": target, "type": data["type"]})

        return {"nodes": list(nodes.values()), "edges": edges}
