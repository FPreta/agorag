"""AppService — the single entry point the gradio app talks to.

It loads the persistent store via the core inference pipeline and adds the few
UI-facing extras gradio needs on top of it:

  - `run(query, history, current_graph)` — the inference event stream.
  - `citation_url(title)` — resolve an answer-model `[Exact Title]` citation to its URL.
  - `pubs_by_hub_id` / `publications` / `entities` / `graph` — for the detail cards.

Run the ETL pipeline first (`python -m core.etl.etl_pipeline`); if the store isn't
present, `AppService.load()` returns an instance with `ready == False` and a clear
message, so the app still starts.
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv

from core.ids import node_id, slugify
from core.inference.inference_pipeline import InferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

NOT_READY_MSG = (
    "⚠️ No data loaded. Build the persistent store first:\n\n"
    "    python -m core.etl.etl_pipeline            # offline sample\n"
    "    python -m core.etl.etl_pipeline --source scrape --rebuild   # live archive\n"
)


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


class AppService:
    def __init__(self, pipeline: InferencePipeline | None = None):
        self.pipeline = pipeline
        self.ready = pipeline is not None
        # Canonical hub node id ("topic::…"/"region:…") -> every
        # publication carrying that entity, across the WHOLE archive (not just the
        # rendered subgraph). Lets a hub's card list all its publications.
        self.pubs_by_hub_id: dict[str, list[dict]] = {}
        # Publication title -> url, for turning [Exact Title] citations into links.
        # Keyed two ways: a normalised exact title, and a slug (slugify drops
        # punctuation/case, so a straight ' still matches a curly ', a trailing
        # colon, etc.).
        self.url_by_title: dict[str, str] = {}
        self.url_by_slug: dict[str, str] = {}
        if pipeline is not None:
            self._build_indexes()

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> "AppService":
        """Load .env + the persistent store. Returns a not-ready instance if the
        store is missing, so the app can start and show a clear message."""
        load_dotenv(REPO_ROOT / ".env")
        try:
            print("[service] loading store + embedding model...")
            pipeline = InferencePipeline.from_data_dir(data_dir)
        except FileNotFoundError as exc:
            print(f"[service] {exc}")
            return cls(None)
        svc = cls(pipeline)
        print(f"[service] ready — {len(svc.publications)} publications")
        return svc

    # --- UI-facing data -------------------------------------------------

    @property
    def publications(self) -> dict[str, dict]:
        return self.pipeline.publications if self.pipeline else {}

    @property
    def pub_by_node_id(self) -> dict[str, dict]:
        return self.pipeline.pub_by_node_id if self.pipeline else {}

    @property
    def entities(self) -> dict:
        return self.pipeline.entities if self.pipeline else {}

    @property
    def graph(self):
        return self.pipeline.graph if self.pipeline else None

    def _build_indexes(self) -> None:
        for pub in self.publications.values():
            for etype, key in (("topic", "topics"), ("region", "regions")):
                for name in pub.get(key) or []:
                    self.pubs_by_hub_id.setdefault(node_id(etype, name), []).append(pub)
            if pub.get("title") and pub.get("url"):
                self.url_by_title[_norm_title(pub["title"])] = pub["url"]
                self.url_by_slug[slugify(pub["title"])] = pub["url"]

    def citation_url(self, title: str) -> str | None:
        """URL for a cited publication title, or None if it isn't one of ours.

        Tries a normalised-exact then a slug match (the slug absorbs apostrophe/quote
        and punctuation differences between the answer model's citation and the source title), and
        retries without a trailing parenthetical — the answer model sometimes appends the format,
        e.g. "… in Europe (Study)"."""
        for t in (title, re.sub(r"\s*\([^)]*\)\s*$", "", title or "").strip()):
            if not t:
                continue
            url = self.url_by_title.get(_norm_title(t)) or self.url_by_slug.get(slugify(t))
            if url:
                return url
        return None

    # --- pipeline -------------------------------------------------------

    def run(self, query: str, history: list, current_graph: dict | None = None):
        """The inference event stream for one query (see InferencePipeline.run)."""
        return self.pipeline.run(query, history, current_graph)
