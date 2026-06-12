# Agora Research Explorer

A GraphRAG-powered research explorer for Agora Energiewende's publication archive.
Ask questions in natural language; get a streamed, cited answer alongside an
interactive knowledge graph that grows with every query, revealing connections
across Agora's research.

**Stack:** a reusable `core` library (ETL + inference) · a thin `backend` service ·
a single-file Gradio UI · Qdrant (local, persistent) · OpenAI (gpt-5.5 for answers,
gpt-5.4 for entity linking) · configurable embeddings (local
`sentence-transformers` or OpenAI).

---

## Architecture

The codebase is a `core` library imported by a thin backend:

```
agorag/
├── core/                          the reusable library
│   ├── config.py                  persistent-store paths + constants
│   ├── ids.py                     node-id helpers (slugify, node_id, …)
│   ├── embeddings.py              EmbeddingService (local ST or remote API)
│   ├── graph_store.py             GraphService — graph access over graph.json
│   ├── etl/                       build the store
│   │   ├── reader.py              Reader      — scrape/sample + diff vs. the store
│   │   ├── transformer.py         Transformer — build the graph + the chunks
│   │   ├── loader.py              Loader      — persist files + embed into Qdrant
│   │   ├── etl_pipeline.py        ETLPipeline — read → transform → load (has __main__)
│   │   └── sample_data.py         offline sample publications
│   └── inference/                 answer a query
│       ├── core_llm.py            streaming LLM client + prompts + stream helpers
│       ├── retriever.py           Retriever   — embed the query + top-k from Qdrant
│       ├── linker.py              Linker      — stream a gpt-5.4 ack + link entities
│       ├── graph_parser.py        GraphParser — stream the gpt-5.5 answer + final graph
│       └── inference_pipeline.py  InferencePipeline — the composed event stream
├── backend/
│   ├── service.py                 AppService — loads the store, wires the pipeline
├── gradio_app.py                  the chat + graph UI
└── data/                          the pre-built persistent store (committed; rebuilt by the ETL pipeline)
├── requirements.txt               Python dependencies
└── .env / .env.example            OPENAI_API_KEY + embedding config

```

### The ETL pipeline (`core.etl`)

`Reader → Transformer → Loader`, composed by `ETLPipeline`:

- **Reader** reads the source publications — the bundled offline sample, or a live
  crawl of the Agora archive — then diffs them against what's already in
  `publications.jsonl`, reporting which publications are genuinely **new**.
- **Transformer** builds the heterogeneous knowledge graph and the embeddable
  chunks (one per key finding, or the summary as a fallback) for each publication.
- **Loader** persists `publications.jsonl`, `graph.json` and `entities.json`, and
  embeds the chunks into the local persistent Qdrant collection under
  `data/qdrant_db/`.

A re-run only embeds the new publications by default (`--rebuild` re-embeds
everything); the graph and the store files are always rewritten in full.

### The inference pipeline (`core.inference`)

`Retriever`, `Linker` and `GraphParser`, composed by `InferencePipeline`:

- **Retriever** embeds the query and fetches the top-k findings from Qdrant.
- **Linker** streams a one-sentence **gpt-5.4** acknowledgment, then resolves the
  `|||ENTITIES|||` tail into the known topics/regions/authors the query references
  and expands them into their connected publications.
- **GraphParser** streams a cited **gpt-5.5** answer, then parses the
  `|||GRAPH_UPDATE|||` tail into the validated knowledge graph (edges are rebuilt
  from ground truth — see below).
- **core_llm** holds the shared streaming client, both prompts, and the
  stream-splitting / citation-linking helpers.

`InferencePipeline.run()` yields one event stream the backend consumes:

```
{"type": "token", "text": …}   → gpt-5.4 ack, then gpt-5.5 answer (live)
{"type": "graph_loading"}      → answer done; the graph is being assembled
{"type": "graph", "graph": …}  → the complete graph state (full replacement)
```

### The backend + UI

`backend.service.AppService` loads the store via `InferencePipeline.from_data_dir`
and adds the few UI-facing extras Gradio needs (publication lookups, citation
URLs). `gradio_app.py` is a two-panel layout: streaming chat with clickable
citation chips on the left, an interactive force-directed graph on the right.

---

## Quick start

The repository ships with a **pre-built store** (`data/` — the full Agora
archive, ~9 MB), so you only need an API key to run the app:

```bash
cd agorag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Configure the API key
cp .env.example .env     # add your OPENAI_API_KEY

# 2. Run the app
python gradio_app.py                     # http://127.0.0.1:7860
```

Open the app and ask, e.g. *"What has Agora published about coal phase-out in
Southeast Asia?"*

> The bundled store was embedded with the default `local` model
> (`all-MiniLM-L6-v2`), which downloads on the first query. If you change
> `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`, rebuild the store (below) so document
> and query vectors stay compatible.

---

## Rebuilding the store (ETL)

The store is regenerated by the ETL pipeline. The bundled offline sample (11
publications, no network) is the default source — handy for exercising the
pipeline without a crawl:

```bash
# Build the store from the offline sample (downloads the embed model on first run)
python -m core.etl.etl_pipeline
```

> ⚠️ A build **merges** its source into the existing store rather than replacing
> it, so running this on top of the committed archive mixes the synthetic sample
> records into the real data (and rewrites `graph.json` / `entities.json` /
> `qdrant_db/`). The command prompts for confirmation when it detects a real
> store; restore it afterwards with `git restore data/`.

To (re)build from the live Agora archive instead:

```bash
python -m core.etl.etl_pipeline --source scrape --rebuild   # full crawl (~7 min, 1.5s/page)
python -m core.etl.etl_pipeline --source scrape --limit 5   # smoke test (first 5 publications)
```

Both write `data/publications.jsonl`, `graph.json`, `entities.json` and
`qdrant_db/`, replacing the committed store. Re-running without `--rebuild` is
incremental: only publications new since the last run are embedded.

Agora runs a TYPO3 CMS; the reader's CSS selectors follow the documented page
structure but a theme change can shift them. If a field comes back empty across
the board, inspect the live HTML and adjust the selectors in
[core/etl/reader.py](core/etl/reader.py). Every field is extracted defensively, so
a publication missing findings/figures/experts is still captured.

`python -m core.etl.etl_pipeline --help` lists all options.

---

## How it works

### Data model — a heterogeneous graph

Four node types: `publication`, `topic`, `region`, `author`. Publications are
**never** linked directly to each other — they connect through shared topic,
region, and author hubs:

```
publication --[has_topic]--> topic
publication --[has_region]--> region
publication --[has_author]--> author
```

Node ids are `{prefix}::{slug}` (`pub::…`, `topic::…`, `region::…`, `author::…`).

### The search pipeline

1. **In parallel:** embed the query + Qdrant semantic search over key findings
   *(fast)*, and a **gpt-5.4** call that streams a one-sentence acknowledgment, then
   (after `|||ENTITIES|||`) a JSON list of the topics/regions/authors the query
   references. The acknowledgment streams immediately; the JSON is parsed
   server-side.
2. **Assemble context:** expand the linked entities into their connected
   publications via the pre-computed graph, and combine those with the retrieved
   findings and the current graph state.
3. **gpt-5.5** streams a cited answer, then (after `|||GRAPH_UPDATE|||`) the
   **complete** new graph state. The graph is validated and its edges rebuilt from
   ground truth, then sent as a `graph` event.

### Frontend

The graph is **owned by the answer model** — each `graph` event replaces the whole state,
preserving the positions of nodes that persist and animating in genuinely new
ones. Clicking a publication node (or a citation chip) opens a detail card with
summary, key findings, authors, tags and a link; clicking a topic/region/author
lists its connected publications, flagged in-graph or elsewhere in the archive.

---

## Configuration

`.env` (see `.env.example`):

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required for synthesis. Missing key → clear in-chat message. |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai`. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local ST model name, or OpenAI model id. |
| `EMBEDDING_BASE_URL` | — | OpenAI only — optional override for an OpenAI-compatible endpoint. |
| `EMBEDDING_API_KEY` | — | OpenAI embeddings only. |
| `DATA_DIR` | `<repo>/data` | Override where the persistent store is read/written. |

The same `EmbeddingService` builds the index (ETL) and embeds queries (inference),
so document and query vectors stay compatible.

---

## Notes on the implementation

- **Model ids:** **`gpt-5.5`** (answers) and **`gpt-5.4`**
  (entity linking), set in [core/inference/core_llm.py](core/inference/core_llm.py).
- **Streamed via the OpenAI chat completions API** with `reasoning_effort="none"`
  on both calls, so the acknowledgment and answer stream with no leading pause.

## Error handling

- **Missing API key** → a clear message is streamed into the chat.
- **No store built** → `AppService.load()` returns a not-ready instance with a
  "run the ETL pipeline" message; the app still starts.
- **Empty retrieval** → the pipeline still runs; the answer model says what's missing.
- **LLM failure** → the retrieved publications are returned unsynthesized, plus a
  deterministic fallback graph built from the linked entities.
- **Malformed graph JSON from the answer model** → falls back to the previous graph state.
