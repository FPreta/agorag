"""The ETL pipeline: read -> transform -> load, end to end.

Reads source publications (offline sample or a live scrape), diffs them against
the existing store, rebuilds the knowledge graph, and embeds the chunks into
Qdrant — incrementally by default, so a re-run only embeds genuinely new
publications.

    # offline, from the bundled sample (default)
    python -m core.etl.etl_pipeline

    # crawl the live archive (full rebuild of the vector store)
    python -m core.etl.etl_pipeline --source scrape --rebuild

    # smoke test: first 5 scraped publications
    python -m core.etl.etl_pipeline --source scrape --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from ..embeddings import EmbeddingService
from .loader import Loader
from .reader import Reader, ReadResult
from .sample_data import sample_publications
from .transformer import Transformer


class ETLPipeline:
    def __init__(
        self, data_dir: str | Path | None = None, embeddings: EmbeddingService | None = None
    ):
        self.reader = Reader(data_dir)
        self.transformer = Transformer()
        self.loader = Loader(data_dir, embeddings)

    def run(
        self, source: str = "sample", limit: int | None = None, rebuild: bool = False
    ) -> ReadResult:
        """Run the full pipeline and return the read result (all/new publications).

        `rebuild=True` re-embeds every publication from scratch; otherwise only
        publications new since the last run are embedded (the graph and the store
        files are always rewritten in full, which is cheap)."""
        result = self.reader.read(source=source, limit=limit)
        print(
            f"[etl] read {len(result.all)} publications "
            f"({len(result.new)} new, {len(result.existing)} already in the store)"
        )

        graph, entities = self.transformer.build_graph(result.all)
        self.loader.save_publications(result.all)
        self.loader.save_graph(graph, entities)

        # Full rebuild when asked, when the store is empty/first-run, or when the
        # Qdrant collection doesn't exist yet; otherwise embed only the new pubs.
        fresh = rebuild or not result.existing or not self.loader.collection_exists()
        to_embed = result.all if fresh else result.new
        if not to_embed:
            print("[etl] nothing new to embed — store is already up to date")
            return result

        chunks = self.transformer.build_chunks(to_embed)
        self.loader.load_chunks(chunks, fresh=fresh)
        print("[etl] done")
        return result


def _confirm_sample_over_store(data_dir: str | Path | None, assume_yes: bool) -> bool:
    """Guard a sample build against an existing real store.

    The reader merges the source into whatever is already persisted, so a sample
    run on top of the committed archive mixes the synthetic records in rather than
    replacing it. If real (non-sample) publications are present, confirm first."""
    existing = Reader(data_dir).existing_publications()
    sample_ids = {p["id"] for p in sample_publications()}
    real = [p for p in existing if p["id"] not in sample_ids]
    if not real or assume_yes:
        return True
    print(
        f"[etl] the store already holds {len(real)} non-sample publication(s). "
        f"Building from the offline sample will merge {len(sample_ids)} synthetic "
        "records into it (restore with `git restore data/`)."
    )
    try:
        reply = input("[etl] proceed? [y/N] ").strip().lower()
    except EOFError:  # non-interactive (piped/CI): don't silently mutate the store
        reply = ""
    return reply in {"y", "yes"}


def main() -> None:
    # Load the repo-root .env so the ETL run sees the same EMBEDDING_* / API key
    # config the app side loads (EmbeddingService.from_env reads these).

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    parser = argparse.ArgumentParser(description="Build the agorag persistent store.")
    parser.add_argument(
        "--source",
        choices=["sample", "scrape"],
        default="sample",
        help="Read offline sample records (default) or crawl the live archive.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only read the first N publications."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed every publication instead of only the new ones.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the store location (default: $DATA_DIR or ./data).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt when a sample build would touch an existing store.",
    )
    args = parser.parse_args()

    if args.source == "sample" and not _confirm_sample_over_store(args.data_dir, args.yes):
        print("[etl] aborted — store left unchanged.")
        return

    ETLPipeline(data_dir=args.data_dir).run(
        source=args.source, limit=args.limit, rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()
