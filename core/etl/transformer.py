"""ETL step 2 — turn publications into a graph and into embeddable chunks.

The graph is heterogeneous: publications are NEVER connected directly to each
other. They connect through shared topic / region hub nodes.

    publication --[has_topic]--> topic
    publication --[has_region]--> region

Chunks are what gets embedded for semantic search: one chunk per key finding
("{headline}. {body}"), or the summary as a single chunk for a publication with
no findings. Each chunk carries the metadata the retriever surfaces back.
"""

from __future__ import annotations

from ..ids import node_id


class Transformer:
    # --- graph ----------------------------------------------------------

    def build_graph(self, pubs: list[dict]) -> tuple[dict, dict]:
        """Return (graph, entities). graph is {nodes, edges}; entities is the flat
        per-type name lists the linker prompt enumerates."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        edge_seen: set[tuple[str, str, str]] = set()

        def add_node(node: dict) -> None:
            # First writer wins, but enrich missing fields on later sightings.
            existing = nodes.get(node["id"])
            if existing is None:
                nodes[node["id"]] = node
            else:
                for k, v in node.items():
                    if v and not existing.get(k):
                        existing[k] = v

        def add_edge(source: str, target: str, etype: str) -> None:
            key = (source, target, etype)
            if key not in edge_seen:
                edge_seen.add(key)
                edges.append({"source": source, "target": target, "type": etype})

        for pub in pubs:
            pid = node_id("publication", pub["id"])
            add_node(
                {
                    "id": pid,
                    "type": "publication",
                    "title": pub.get("title"),
                    "date": pub.get("date"),
                    "format": pub.get("format"),
                }
            )

            for topic in pub.get("topics", []) or []:
                tid = node_id("topic", topic)
                add_node({"id": tid, "type": "topic", "name": topic})
                add_edge(pid, tid, "has_topic")

            for region in pub.get("regions", []) or []:
                rid = node_id("region", region)
                add_node({"id": rid, "type": "region", "name": region})
                add_edge(pid, rid, "has_region")

        graph = {"nodes": list(nodes.values()), "edges": edges}
        entities = {
            "topics": sorted({n["name"] for n in nodes.values() if n["type"] == "topic"}),
            "regions": sorted({n["name"] for n in nodes.values() if n["type"] == "region"}),
        }
        return graph, entities

    # --- chunks ---------------------------------------------------------

    def build_chunks(self, pubs: list[dict]) -> list[dict]:
        """Flatten publications into embeddable chunks with metadata payloads.

        One chunk per key finding ("{headline}. {body}"); for a publication with no
        findings, its summary as a single chunk tagged finding_number = 0."""
        chunks: list[dict] = []
        for pub in pubs:
            meta = {
                "publication_id": pub["id"],
                "publication_title": pub.get("title"),
                "topics": pub.get("topics", []) or [],
                "regions": pub.get("regions", []) or [],
            }
            findings = pub.get("key_findings") or []
            if findings:
                for finding in findings:
                    headline = (finding.get("headline") or "").strip()
                    body = (finding.get("body") or "").strip()
                    text = f"{headline}. {body}".strip(". ").strip()
                    if not text:
                        continue
                    chunks.append(
                        {"text": text, "finding_number": finding.get("number", 0), **meta}
                    )
            else:
                summary = (pub.get("summary") or "").strip()
                if summary:
                    chunks.append({"text": summary, "finding_number": 0, **meta})
        return chunks
