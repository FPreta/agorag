"""Gradio front-end for the Agora Research Explorer.

A single-file UI over the core library. It loads the persistent store through the
thin backend service (no FastAPI server needed) and runs the identical pipeline:

    retrieval + acknowledgment/entity-linking  ->  the answer model answer + graph

The answer streams into a chat panel; once the answer model finishes, the complete graph state
is rendered as an interactive vis-network graph (colored by entity type).

Run:
    pip install -r backend/requirements.txt
    # build the store first: python -m core.etl.etl_pipeline
    python gradio_app.py            # http://127.0.0.1:7860

Needs OPENAI_API_KEY (read from .env or the environment).
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time

import gradio as gr

from backend.service import NOT_READY_MSG, AppService
from core.ids import canonical_pub_ids, node_id
from core.inference.core_llm import CitationLinker

# --- entity-type styling (mirrors the old frontend theme) -------------------

TYPE_COLORS = {
    "publication": "#2f6f9f",
    "topic": "#7a4fa3",
    "region": "#3f8f5b",
}
TYPE_SHAPE = {"publication": "dot", "topic": "diamond", "region": "square"}
TYPE_SIZE = {"publication": 14, "topic": 22, "region": 20}

EMPTY_GRAPH = {"nodes": [], "edges": []}

# The backend service, loaded once at startup (see __main__).
svc: AppService | None = None


# --- history normalization (gradio-specific) --------------------------------


def _text_of(content) -> str:
    """Flatten a chat turn's content to plain text.

    Gradio may hand us a bare string or a list of content-part dicts
    ([{"type":"text","text":...}]) depending on version; the OpenAI API and our
    message helpers want a string. Normalize both shapes here."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts)
    return "" if content is None else str(content)


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url)


def _delinkify(text: str) -> str:
    """Reverse display-time citation linking before the text goes back to the model.

    The chat shows citations as "(ref. [Title](url))", but feeding that back into the
    conversation teaches the answer model to emit "(ref. …)" itself — which the linker then wraps
    again, producing "(ref. (ref. …))". Stripping markdown links to "[Title]" and
    unwrapping the "(ref. …)" parenthetical (honoring nesting) means the model only
    ever sees its own bare "[Title]" convention, breaking the feedback loop."""
    text = _MD_LINK_RE.sub(r"[\1]", text)  # [Title](url) -> [Title]
    while "(ref." in text:
        out: list[str] = []
        i, changed = 0, False
        while i < len(text):
            if text.startswith("(ref.", i):
                depth, j = 0, i
                while j < len(text):
                    depth += (text[j] == "(") - (text[j] == ")")
                    if depth == 0:
                        break
                    j += 1
                if j < len(text):  # found the matching ')'
                    out.append(text[i + len("(ref.") : j].lstrip())
                    i, changed = j + 1, True
                    continue
            out.append(text[i])
            i += 1
        text = "".join(out)
        if not changed:  # an unmatched "(ref." — stop rather than loop forever
            break
    return text


def _clean_history(history) -> list[dict]:
    """Replayable {"role","content"} turns with empty content dropped.

    Assistant turns are de-linkified first (see _delinkify) so the model never sees
    the display-only "(ref. …)" citation formatting and re-emits it.

    An empty content block (e.g. the assistant placeholder add_user injects, or an
    aborted prior turn) makes the OpenAI API reject the whole request with an empty
    content error, so it must never be sent."""
    cleaned: list[dict] = []
    for turn in history or []:
        role = turn["role"] if isinstance(turn, dict) else getattr(turn, "role", None)
        text = _text_of(turn["content"] if isinstance(turn, dict) else getattr(turn, "content", ""))
        if role == "assistant":
            text = _delinkify(text)
        if role and text.strip():
            cleaned.append({"role": role, "content": text})
    return cleaned


# --- graph rendering (interactive vis-network in an iframe) ------------------


def build_cards(graph) -> dict[str, dict]:
    """Per-node detail payloads ("model cards"), enriched from the loaded data.

    Publication nodes carry their full record; topic/region nodes carry the
    publications linked to them. Keyed by node id so the iframe can look a card up on
    click without any round-trip to Python.

    Hub→publication links are resolved from the graph's *own* edges (treated as
    undirected) rather than re-derived from the data: the live graph is authored by
    the answer model, which copies publication ids verbatim but invents its own hub slugs, so
    canonical node_id() keys would miss every hub. The edges are also exactly what the
    user sees drawn, so the card can't disagree with the graph."""
    publications = svc.publications
    pubs_by_hub_id = svc.pubs_by_hub_id
    pub_by_node_id = svc.pub_by_node_id

    nodes = graph.get("nodes", [])
    node_by_id = {n["id"]: n for n in nodes}

    # Title index for recovering a publication when the answer model's "pub::<slug>" id drifts
    # from our canonical node id (it sees the bare id in the retrieval context, so
    # the prefixed slug isn't always reproduced). The node label is the exact title
    # the answer model is told to cite, so it's a reliable secondary key.
    pubs_by_title = {p["title"].strip().lower(): p for p in publications.values()}

    def resolve_pub(nid: str, label: str) -> dict:
        # the answer model's pub ids drift (wrong prefix, apostrophe/quote variants, a "(Study)"
        # suffix), so match the canonical candidates from the label, then the title.
        for cand in canonical_pub_ids(nid, label):
            pub = pub_by_node_id.get(cand)
            if pub:
                return pub
        return pubs_by_title.get(label.strip().lower(), {})

    # Canonical ids of the publications currently drawn in the graph, so a hub card
    # can mark which of its publications the user already sees.
    graph_pub_ids: set[str] = set()
    for n in nodes:
        if n["type"] == "publication":
            pub = resolve_pub(n["id"], n["label"])
            if pub.get("id"):
                graph_pub_ids.add(node_id("publication", pub["id"]))

    # Edge-derived neighbours: the fallback for a hub whose id isn't a canonical
    # taxonomy id (e.g. a malformed/fallback graph), where the archive-wide index
    # below can't be keyed.
    neighbors: dict[str, set[str]] = {}
    for e in graph.get("edges", []):
        src, tgt = e.get("source"), e.get("target")
        if src in node_by_id and tgt in node_by_id:
            neighbors.setdefault(src, set()).add(tgt)
            neighbors.setdefault(tgt, set()).add(src)

    def pub_ref(pub: dict, fallback_title: str = "") -> dict:
        return {"title": pub.get("title", fallback_title), "url": pub.get("url", "")}

    def hub_related(nid: str) -> tuple[list[dict], list[dict]]:
        """All publications for this hub, split into (in this graph, elsewhere).

        Drawn from the archive-wide index so the card shows every publication that
        carries the entity — not only the few the answer model wired into the subgraph."""
        archive = pubs_by_hub_id.get(nid)
        if archive is None:
            # Non-canonical hub id: fall back to whatever edges the answer model drew.
            drawn = []
            for mid in sorted(neighbors.get(nid, ())):
                neighbor = node_by_id.get(mid)
                if neighbor and neighbor["type"] == "publication":
                    drawn.append(pub_ref(resolve_pub(mid, neighbor["label"]), neighbor["label"]))
            return drawn, []
        in_graph, other = [], []
        for pub in sorted(archive, key=lambda p: p.get("title", "").lower()):
            target = in_graph if node_id("publication", pub["id"]) in graph_pub_ids else other
            target.append(pub_ref(pub))
        return in_graph, other

    cards: dict[str, dict] = {}
    for n in nodes:
        nid, ntype, label = n["id"], n["type"], n["label"]
        if ntype == "publication":
            pub = resolve_pub(nid, label)
            cards[nid] = {
                "type": "publication",
                "title": pub.get("title", label),
                "subtitle": pub.get("subtitle", ""),
                "date": pub.get("date", ""),
                "format": pub.get("format", ""),
                "authors": pub.get("authors", []) or [],
                "topics": pub.get("topics", []) or [],
                "regions": pub.get("regions", []) or [],
                "summary": pub.get("summary", ""),
                "url": pub.get("url", ""),
                "findings": [f.get("headline", "") for f in (pub.get("key_findings") or [])],
            }
        else:
            in_graph, other = hub_related(nid)
            cards[nid] = {
                "type": ntype,
                "title": label,
                "related_in_graph": in_graph,
                "related_other": other,
            }
    return cards


# vis-network + a click-to-reveal detail panel. Kept as a template (markers
# substituted below) so the CSS/JS braces don't fight Python f-string escaping.
_GRAPH_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"/>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
html,body{margin:0;height:100%;font-family:Inter,system-ui,sans-serif}
#wrap{display:flex;height:100%}
#net{flex:1;height:100%;background:#FAFAF8;min-width:0}
#card{width:300px;height:100%;box-sizing:border-box;overflow-y:auto;padding:16px;
  border-left:1px solid #e6e7e2;background:#fff;color:#16202b;font-size:13px;line-height:1.45}
.ctype{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.ctitle{font-size:16px;font-weight:600;margin:4px 0 2px}
.csub{color:#5a6472;margin-bottom:6px}
.cmeta{color:#5a6472;font-size:12px;margin-bottom:8px}
.csec{font-size:11px;font-weight:700;text-transform:uppercase;color:#8a929c;margin:12px 0 4px}
.cbody{color:#2b3440}
.clist{margin:4px 0;padding-left:18px}
.clist li{margin:3px 0}
.clist a{color:#2f6f9f;text-decoration:none}
.chip{display:inline-block;background:#f0f1ec;border-radius:10px;padding:2px 8px;
  margin:2px 4px 2px 0;font-size:12px;color:#3a424c}
.clink{display:inline-block;margin-top:14px;color:#2f6f9f;text-decoration:none;font-weight:600}
.cplace{color:#5a6472;display:flex;height:100%;align-items:center;justify-content:center;
  text-align:center;padding:0 12px}
</style></head><body>
<div id="wrap"><div id="net"></div><div id="card"></div></div>
<script>
const nodes=new vis.DataSet(__NODES__);
const edges=new vis.DataSet(__EDGES__);
const network=new vis.Network(document.getElementById("net"),{nodes,edges},__OPTIONS__);
const cards=__CARDS__, typeColors=__TYPECOLORS__;
const card=document.getElementById("card");
const placeholder='<div class="cplace">Click any node to see its details.</div>';
card.innerHTML=placeholder;
function esc(s){return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function sec(t){return '<div class="csec">'+esc(t)+'</div>';}
function chip(t){return '<span class="chip">'+esc(t)+'</span>';}
function render(id){
  const c=cards[id];
  if(!c){card.innerHTML=placeholder;return;}
  const color=typeColors[c.type]||'#2f6f9f';
  let h='<div class="ctype" style="color:'+color+'">'+esc(c.type)+'</div>';
  h+='<div class="ctitle">'+esc(c.title)+'</div>';
  if(c.type==='publication'){
    if(c.subtitle)h+='<div class="csub">'+esc(c.subtitle)+'</div>';
    const meta=[c.format,c.date].filter(Boolean).join(' · ');
    if(meta)h+='<div class="cmeta">'+esc(meta)+'</div>';
    if(c.authors&&c.authors.length)h+=sec('Authors')+'<div class="cbody">'+esc(c.authors.join(', '))+'</div>';
    if(c.summary)h+=sec('Summary')+'<div class="cbody">'+esc(c.summary)+'</div>';
    if(c.findings&&c.findings.length)h+=sec('Key findings')+'<ul class="clist">'+c.findings.map(f=>'<li>'+esc(f)+'</li>').join('')+'</ul>';
    const tags=(c.topics||[]).concat(c.regions||[]);
    if(tags.length)h+=sec('Tags')+'<div>'+tags.map(chip).join('')+'</div>';
    if(c.url)h+='<a class="clink" href="'+esc(c.url)+'" target="_blank" rel="noopener">Open publication ↗</a>';
  }else{
    const pubList=ps=>'<ul class="clist">'+ps.map(p=>'<li><a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.title)+'</a></li>').join('')+'</ul>';
    const inG=c.related_in_graph||[], other=c.related_other||[];
    const total=inG.length+other.length;
    if(!total){h+=sec('Publications (0)')+'<div class="cbody">No linked publications.</div>';}
    else{
      h+=sec('In this graph ('+inG.length+')');
      h+=inG.length?pubList(inG):'<div class="cbody">None of this entity’s publications are in the current graph.</div>';
      if(other.length){h+=sec('Elsewhere in the archive ('+other.length+')')+pubList(other);}
    }
  }
  card.innerHTML=h;
}
network.on('click',function(p){if(p.nodes&&p.nodes.length)render(p.nodes[0]);else card.innerHTML=placeholder;});
</script></body></html>"""


def graph_loading_html(message: str = "Building the knowledge graph…") -> str:
    """An animated spinner shown in the graph panel while the answer model streams the graph JSON
    (a stretch with no client-facing tokens, so the UI would otherwise freeze). The
    CSS animation runs client-side, so it stays alive without server updates."""
    return (
        "<div style='display:flex;flex-direction:column;gap:14px;align-items:center;"
        "justify-content:center;height:600px;border:1px solid #e6e7e2;border-radius:10px;"
        "background:#FAFAF8;color:#5a6472;font-family:Inter,sans-serif'>"
        "<div style='width:34px;height:34px;border:3px solid #d7dbd2;border-top-color:#2f6f9f;"
        "border-radius:50%;animation:agspin 0.8s linear infinite'></div>"
        f"<div>{message}</div>"
        "<style>@keyframes agspin{to{transform:rotate(360deg)}}</style></div>"
    )


def legend_html() -> str:
    """A static key for the graph's entity-type encoding (color + shape).

    Generated from TYPE_COLORS/TYPE_SHAPE so it can never drift from what the graph
    actually draws. Lives outside the iframe (always visible, even before the first
    graph), so it reads as part of the panel rather than the canvas."""
    shape_css = {
        "dot": "border-radius:50%",
        "diamond": "border-radius:2px;transform:rotate(45deg)",
        "square": "border-radius:2px",
    }
    items = []
    for ntype, color in TYPE_COLORS.items():
        shape = TYPE_SHAPE.get(ntype, "dot")
        swatch = (
            f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
            f"{shape_css.get(shape, 'border-radius:50%')};margin-right:7px;flex:none'></span>"
        )
        items.append(
            "<span style='display:inline-flex;align-items:center;margin-right:18px;"
            f"font-size:12px;color:#3a424c;text-transform:capitalize'>{swatch}{ntype}</span>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;align-items:center;"
        "padding:8px 12px;border:1px solid #e6e7e2;border-radius:8px;background:#fff;"
        "font-family:Inter,sans-serif'>"
        "<span style='font-size:11px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:.06em;color:#8a929c;margin-right:16px'>Legend</span>"
        + "".join(items)
        + "</div>"
    )


def graph_iframe(graph) -> str:
    if not graph or not graph.get("nodes"):
        return (
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:600px;border:1px solid #e6e7e2;border-radius:10px;background:#FAFAF8;"
            "color:#5a6472;font-family:Inter,sans-serif'>"
            "The knowledge graph will appear here and grow with each question.</div>"
        )

    def color_for(node_type):
        bg = TYPE_COLORS.get(node_type, "#2f6f9f")
        return {
            "background": bg,
            "border": "#ffffff",
            "highlight": {"background": bg, "border": "#16202b"},
        }

    nodes = [
        {
            "id": n["id"],
            "label": n["label"],
            "shape": TYPE_SHAPE.get(n["type"], "dot"),
            "size": TYPE_SIZE.get(n["type"], 14),
            "color": color_for(n["type"]),
        }
        for n in graph["nodes"]
    ]
    edges = [{"from": e["source"], "to": e["target"]} for e in graph["edges"]]
    options = {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -9000,
                "springLength": 130,
                "springConstant": 0.03,
            },
            "stabilization": {"iterations": 180},
        },
        "nodes": {"font": {"size": 14, "face": "Inter, sans-serif"}, "borderWidth": 1.5},
        "edges": {"color": {"color": "rgba(120,120,120,0.35)"}, "smooth": False},
        "interaction": {"hover": True, "tooltipDelay": 120},
    }

    def _js(obj) -> str:
        # </ guard keeps embedded strings from prematurely closing <script>.
        return json.dumps(obj).replace("</", "<\\/")

    doc = (
        _GRAPH_TEMPLATE.replace("__NODES__", _js(nodes))
        .replace("__EDGES__", _js(edges))
        .replace("__OPTIONS__", _js(options))
        .replace("__CARDS__", _js(build_cards(graph)))
        .replace("__TYPECOLORS__", _js(TYPE_COLORS))
    )
    srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe style="width:100%;height:600px;border:1px solid #e6e7e2;'
        f'border-radius:10px" srcdoc="{srcdoc}"></iframe>'
    )


# --- Gradio wiring ----------------------------------------------------------


def add_user(message, history):
    if not message or not message.strip():
        return gr.update(), history
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    return "", history


# Typewriter smoothing: the model emits text in uneven bursts, so we buffer it and
# release it on a steady tick. Each tick emits a slice of the backlog (so we track
# the model's rate with a small, even lag), with a floor for a natural trickle when
# nearly caught up.
STREAM_TICK = 0.035  # seconds between visual updates (~28 fps)
STREAM_DRAIN_FRACTION = 9  # emit ~1/9 of the buffered backlog per tick
STREAM_MIN_CHARS = 2  # ...but at least this many, so the tail still moves


def bot(history, current_graph):
    if not svc or not svc.ready:
        history[-1]["content"] = NOT_READY_MSG
        yield history, current_graph, gr.update()
        return

    query = _text_of(history[-2]["content"])
    conv = _clean_history(history[:-2])  # prior turns, empties dropped
    if not query.strip():
        history[-1]["content"] = "⚠️ Empty question — please type something to ask."
        yield history, current_graph, gr.update()
        return

    # Run the blocking pipeline on a worker thread and drain its events here, so the
    # answer text can be paced out smoothly instead of in the model's bursts. The
    # CitationLinker rewrites the answer model's [Exact Title] citations into clickable links
    # (rendered as a parenthetical "(ref. …)") as the answer streams.
    events: queue.Queue = queue.Queue()
    DONE = object()

    def _produce():
        clink = CitationLinker(svc.citation_url, wrap=lambda body: f"(ref. {body})")

        def flush_clink():
            tail = clink.flush()
            if tail:
                events.put({"type": "token", "text": tail})

        try:
            for evt in svc.run(query, conv, current_graph):
                if evt["type"] == "token":
                    chunk = clink.feed(evt["text"])
                    if chunk:
                        events.put({"type": "token", "text": chunk})
                elif evt["type"] == "graph_loading":
                    flush_clink()  # emit any half-buffered citation before the spinner
                    events.put({"type": "graph_loading"})
                else:  # "graph"
                    flush_clink()
                    events.put(evt)
        except Exception as exc:  # noqa: BLE001 - surface, never hang the UI
            events.put({"type": "token", "text": f"\n\n⚠️ {exc}"})
        finally:
            flush_clink()
            events.put(DONE)

    threading.Thread(target=_produce, daemon=True).start()

    acc = ""  # text shown so far
    pending = ""  # text received but not yet released
    new_graph = current_graph
    spinner_pending = False
    finished = False

    while True:
        # Absorb everything the worker has produced so far without blocking.
        try:
            while True:
                evt = events.get_nowait()
                if evt is DONE:
                    finished = True
                elif evt["type"] == "token":
                    pending += evt["text"]
                elif evt["type"] == "graph_loading":
                    spinner_pending = True
                else:  # "graph"
                    new_graph = evt["graph"]
        except queue.Empty:
            pass

        if pending:
            n = max(STREAM_MIN_CHARS, len(pending) // STREAM_DRAIN_FRACTION)
            acc += pending[:n]
            pending = pending[n:]
            history[-1]["content"] = acc
            yield history, new_graph, gr.update()
        elif spinner_pending:
            # Buffered answer text is fully shown; now reveal the graph spinner.
            spinner_pending = False
            yield history, new_graph, graph_loading_html()
        elif finished:
            break

        time.sleep(STREAM_TICK)

    yield history, new_graph, graph_iframe(new_graph)


def clear():
    return [], EMPTY_GRAPH, graph_iframe(EMPTY_GRAPH)


# Lock/unlock the input controls around a run so a second question can't be sent
# (via the button or Enter) while the pipeline is streaming.
def _lock_input():
    return gr.update(interactive=False), gr.update(interactive=False)


def _unlock_input():
    return gr.update(interactive=True), gr.update(interactive=True)


def make_chatbot():
    # Gradio 6 removed the `type` arg (messages is the only format now); 4/5 need
    # type="messages" to accept the {"role","content"} dicts we pass.
    kwargs = dict(height=560, label="Conversation", elem_id="ag-chatbot")
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


# A refined editorial serif used throughout the app — suits a research tool and reads
# better than the default UI sans. Applied to the whole Gradio container (so every
# label, the question input, and the conversation share one font), with a slightly
# larger size for the conversation messages where the reading happens.
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;500;600&display=swap');
.gradio-container, .gradio-container *,
#ag-input textarea, #ag-input input {
  font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
}
#ag-chatbot .message, #ag-chatbot .message *,
#ag-chatbot .prose, #ag-chatbot .prose * {
  font-size: 15px !important;
  line-height: 1.62 !important;
}
"""


# Gradio 6 moved `css` from the Blocks constructor to launch() (and warns if it's
# passed here); 4/5 only accept it on the constructor. Route it by version.
_GR_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {} if _GR_MAJOR >= 6 else {"css": APP_CSS}

with gr.Blocks(title="Agora Research Explorer", **_blocks_kwargs) as demo:
    gr.Markdown(
        "# Agora Research Explorer\n"
        "Explore Agora's publication archive through natural language. "
        "The knowledge graph builds as you go."
    )
    graph_state = gr.State(dict(EMPTY_GRAPH))

    with gr.Row():
        with gr.Column(scale=45):
            chatbot = make_chatbot()
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about Agora's publications...",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                    elem_id="ag-input",
                )
                send = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation")
        with gr.Column(scale=55):
            gr.HTML(legend_html())
            graph_html = gr.HTML(graph_iframe(EMPTY_GRAPH))

    for trigger in (msg.submit, send.click):
        trigger(add_user, [msg, chatbot], [msg, chatbot], queue=False).then(
            _lock_input, None, [send, msg], queue=False
        ).then(bot, [chatbot, graph_state], [chatbot, graph_state, graph_html]).then(
            _unlock_input, None, [send, msg], queue=False
        )
    clear_btn.click(clear, None, [chatbot, graph_state, graph_html], queue=False)


if __name__ == "__main__":
    svc = AppService.load()
    app = demo.queue()
    # Gradio 6 takes `theme` (and `css`) in launch(); 4/5 don't accept them there
    # (css went on the Blocks constructor above instead).
    launch_kwargs = {"theme": gr.themes.Soft()}
    if _GR_MAJOR >= 6:
        launch_kwargs["css"] = APP_CSS
    try:
        app.launch(**launch_kwargs)
    except TypeError:
        app.launch()
