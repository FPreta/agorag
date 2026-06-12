"""The inference pipeline: retrieve -> link -> answer + graph, as one event stream.

Composes the Retriever, Linker and GraphParser into the sequence the backend
consumes. `run` is a generator of event dicts:

    {"type": "token", "text": ...}   answer/acknowledgment text, forwarded live
    {"type": "graph_loading"}        answer done; the graph is being assembled
    {"type": "graph", "graph": ...}  the final (or fallback) graph state

Citation linking is intentionally left to the caller (it needs the publication URL
map, a display concern), so the tokens here are the model's raw text.

Build one with `InferencePipeline.from_data_dir()`, which loads the persistent
store and also exposes `entities`, `graph`, `publications` and `pub_by_node_id`
so the backend can reuse them without re-reading the files.
"""

from __future__ import annotations

import json
from pathlib import Path

import openai

from .. import config
from ..graph_store import GraphService
from ..ids import node_id
from . import graph_parser as gp
from .core_llm import LLMService
from .graph_parser import GraphParser
from .linker import Linker
from .retriever import Retriever

EMPTY_GRAPH = {"nodes": [], "edges": []}

MISSING_KEY_MSG = "⚠️ No valid OPENAI_API_KEY. Add it to .env (or the environment) and restart."


def _load_publications(path: Path) -> dict[str, dict]:
    pubs: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pub = json.loads(line)
                pubs[pub["id"]] = pub
    return pubs


class InferencePipeline:
    def __init__(
        self,
        retriever: Retriever,
        linker: Linker,
        graph_parser: GraphParser,
        graph: GraphService,
        publications: dict[str, dict],
        pub_by_node_id: dict[str, dict],
        entities: dict,
        *,
        top_k: int = 15,
        missing_key_msg: str = MISSING_KEY_MSG,
    ):
        self.retriever = retriever
        self.linker = linker
        self.graph_parser = graph_parser
        self.graph = graph
        self.publications = publications
        self.pub_by_node_id = pub_by_node_id
        self.entities = entities
        self.top_k = top_k
        self.missing_key_msg = missing_key_msg

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path | None = None,
        *,
        llm: LLMService | None = None,
        embeddings=None,
        top_k: int = 15,
    ) -> "InferencePipeline":
        """Load the persistent store and wire up the pipeline.

        Requires the store to exist (run the ETL pipeline first). `llm` and
        `embeddings` may be injected for testing; otherwise they're built from env."""
        paths = config.store_paths(data_dir)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Persistent store not ready — missing: "
                + ", ".join(missing)
                + ". Run: python -m core.etl.etl_pipeline"
            )

        llm = llm or LLMService()
        entities = json.loads(paths["entities"].read_text(encoding="utf-8"))
        graph = GraphService(paths["graph"])
        publications = _load_publications(paths["publications"])
        pub_by_node_id = {node_id("publication", pid): pub for pid, pub in publications.items()}

        retriever = Retriever(paths["qdrant"], embeddings=embeddings)
        linker = Linker(llm, graph, entities)
        graph_parser = GraphParser(llm, graph, entities)
        return cls(
            retriever,
            linker,
            graph_parser,
            graph,
            publications,
            pub_by_node_id,
            entities,
            top_k=top_k,
        )

    def run(self, query: str, history: list, current_graph: dict | None = None):
        """Yield the pipeline's event stream for one user query."""
        current_graph = current_graph or None

        # --- retrieval (best-effort; never fatal) -----------------------
        hits: list[dict] = []
        try:
            hits = self.retriever.retrieve(query, top_k=self.top_k)
            print(f"[inference] retrieval found {len(hits)} hits")
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            print(f"[inference] retrieval failed: {exc}")

        # --- linker: acknowledgment + entity linking --------------------
        link_stream, link_result = self.linker.link(query, history)
        try:
            for chunk in link_stream:
                yield {"type": "token", "text": chunk}
        except openai.AuthenticationError:
            yield {"type": "token", "text": self.missing_key_msg}
            yield {"type": "graph", "graph": current_graph or EMPTY_GRAPH}
            return
        except openai.APIError as exc:
            print(f"[inference] linker failed: {exc}")  # continue without an acknowledgment

        linked = link_result.linked
        print(f"[inference] linked entities: {linked}")
        yield {"type": "token", "text": "\n\n"}  # visual transition

        # --- graph parser: answer + graph -------------------------------
        parse_stream, parse_result = self.graph_parser.parse(
            query, history, hits, link_result.entity_pubs, self.pub_by_node_id, current_graph
        )
        try:
            for evt in parse_stream:
                yield evt  # {"type": "token"} / {"type": "graph_loading"}
        except openai.AuthenticationError:
            yield {"type": "token", "text": self.missing_key_msg}
            yield {"type": "graph", "graph": current_graph or EMPTY_GRAPH}
            return
        except openai.APIError as exc:
            # LLM failure → unsynthesized retrieved publications + a deterministic
            # fallback graph so the user still sees connections.
            print(f"[inference] answer failed: {exc}")
            yield {"type": "token", "text": gp.fallback_answer(hits)}
            fallback = current_graph or self.graph.subgraph_for_named_entities(linked)
            yield {"type": "graph", "graph": fallback}
            return

        yield {"type": "graph", "graph": parse_result.graph}
