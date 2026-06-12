"""The shared streaming LLM client, the prompts, and the stream-splitting helpers.

Both inference calls go through here:

    Linker (LINKER_MODEL)   <acknowledgment> |||ENTITIES|||    {entities json}
    GraphParser (ANSWER_MODEL) <answer>      |||GRAPH_UPDATE||| {graph json}

Each streams a human-facing section, then a delimiter, then a machine-only JSON
section. `LLMService.stream` yields the model's text deltas; `SplitStreamer`
forwards the part before the delimiter and buffers the JSON tail for parsing;
`CitationLinker` rewrites `[Exact Title]` citations into clickable links on the fly.

Models (current): gpt-5.5 for answers, gpt-5.4 for entity linking, both via
the OpenAI chat completions API with streaming. Reasoning effort is set to
"none" so text streams with no leading pause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import openai

# --- models / protocol constants -------------------------------------------

LINKER_MODEL = "gpt-5.4"
ANSWER_MODEL = "gpt-5.5"

ENTITY_DELIMITER = "|||ENTITIES|||"
GRAPH_DELIMITER = "|||GRAPH_UPDATE|||"

LINKER_MAX_TOKENS = 2560
ANSWER_MAX_TOKENS = 10000

# Context-assembly caps for the answer prompt.
MAX_FINDINGS = 15
MAX_ENTITY_PUBS = 20
SNIPPET_CHARS = 320


# --- the streaming client ---------------------------------------------------


class LLMService:
    """Thin wrapper over the OpenAI client that streams text deltas.

    The client reads OPENAI_API_KEY from the environment; construction never
    fails, so a missing/invalid key surfaces as an AuthenticationError on the first
    call. Callers handle openai.AuthenticationError / openai.APIError."""

    def __init__(self, client: "openai.OpenAI | None" = None):
        if client is None:
            import openai

            client = openai.OpenAI()
        self.client = client

    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
        reasoning_effort: str = "none",
    ):
        """Yield text deltas for one streamed completion.

        `reasoning_effort="none"` disables gpt-5 reasoning so the acknowledgment and
        answer start streaming with no leading pause."""
        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_completion_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text


# --- shared message helper --------------------------------------------------


def history_messages(history: list) -> list[dict]:
    """Replay prior conversation turns as OpenAI {"role","content"} messages."""
    messages: list[dict] = []
    for turn in history or []:
        role = turn["role"] if isinstance(turn, dict) else turn.role
        content = turn["content"] if isinstance(turn, dict) else turn.content
        messages.append({"role": role, "content": content})
    return messages


# --- linker (LINKER_MODEL) prompt -------------------------------------------


def linker_system_prompt(entities: dict) -> str:
    topics = ", ".join(entities.get("topics", []))
    regions = ", ".join(entities.get("regions", []))
    return f"""You are a research assistant for Agora Energiewende. The user is searching \
Agora's publication archive.

First, write a brief one-sentence acknowledgment of what you're about to search \
for. Keep it natural and specific to the query. Examples:
- "Looking into Agora's work on coal phase-out in Southeast Asia..."
- "Searching for publications on hydrogen and industrial decarbonization..."
- "Let me find what Agora has published about electricity market design..."

Then, on a new line, output {ENTITY_DELIMITER} followed by a JSON object \
identifying which of the following entities are referenced or implied by the query.

TOPICS: {topics}
REGIONS: {regions}

The JSON format:
{{"topics": [...], "regions": [...]}}

Only include entities that are clearly relevant. Use the exact names from the \
lists above. If none match for a category, use an empty array.

IMPORTANT: Do NOT answer the user's question. Your only job is to \
produce the acknowledgment and entity linking. Even if the question \
is unrelated to Agora's publications, output your best-guess \
acknowledgment and entity linking. Leave all answering to the \
next model.
"""


def build_linker_messages(query: str, history: list) -> list[dict]:
    messages = history_messages(history)
    messages.append({"role": "user", "content": query})
    return messages


# --- answer (ANSWER_MODEL) prompt -------------------------------------------


def _taxonomy_block(entities: dict | None) -> str:
    """The closed topic/region vocabulary the answer model must draw hub labels from.

    Topics and regions are small, fixed taxonomies, so we list them in full and
    forbid invention."""
    if not entities:
        return ""
    topics = entities.get("topics") or []
    regions = entities.get("regions") or []
    if not topics and not regions:
        return ""
    return f"""

ALLOWED HUB LABELS (do NOT invent topics or regions):
- topic nodes: use ONLY these exact labels — {", ".join(topics)}
- region nodes: use ONLY these exact labels — {", ".join(regions)}
Never create a topic or region outside the lists above. If a publication doesn't \
fit any allowed topic, connect it via a region instead."""


def answer_system_prompt(entities: dict | None = None) -> str:
    return f"""You are a research assistant for Agora Energiewende, a climate and energy \
policy think tank. Answer the user's question based on the provided publication \
excerpts. Be precise, factual, and concise (2-3 paragraphs max). Don't mention \
the retrieval process or the graph. If the context doesn't contain enough \
information to answer the question, say so. Finally, don't mention what is NOT in the context. \

CITATION RULES:
- Cite publications by exact title in square brackets, e.g. [Reforming power \
purchase agreements for flexible coal power]
- Only cite publications from the provided context
- If the context doesn't fully answer the question, say what's missing

GRAPH UPDATE:
After your answer, on a new line, output {GRAPH_DELIMITER} followed by a JSON \
object representing the COMPLETE current knowledge graph state.

The graph should contain all publications, topics, and regions that are \
relevant to the conversation SO FAR (not just this turn). You may add new nodes \
from the retrieved context or remove nodes that are no longer relevant to the \
overall conversation thread.

You will receive the current graph state from the previous turn. Update it by \
adding newly relevant nodes/edges and optionally pruning nodes that no \
longer serve the conversation. Every publication node must be connected to at \
least one topic or region node.

Node format: {{"id": "type::slug", "type": "publication|topic|region", "label": "Display Name"}}
Edge format: {{"source": "node-id", "target": "node-id", "type": "has_topic|has_region"}}

Keep the graph focused and useful — not every retrieved publication needs to be \
in the graph, only those genuinely relevant to the conversation.{_taxonomy_block(entities)}"""


def assemble_context(
    retrieval_hits: list[dict],
    entity_pubs: list[dict],
    pub_by_node_id: dict[str, dict],
    current_graph: dict | None,
) -> str:
    parts: list[str] = []

    # A — retrieved key findings (semantic search).
    parts.append("## Retrieved key findings (semantic search)")
    if retrieval_hits:
        for hit in retrieval_hits[:MAX_FINDINGS]:
            title = hit.get("publication_title", "Unknown")
            pid = hit.get("publication_id", "")
            text = (hit.get("text") or "").strip()
            parts.append(f"- [{title}] (id: {pid})\n  {text}")
    else:
        parts.append("- (no semantic matches found)")

    # B — entity-linked publications (graph expansion).
    parts.append("\n## Publications linked via topics/regions")
    if entity_pubs:
        for node in entity_pubs[:MAX_ENTITY_PUBS]:
            title = node.get("title", node["id"])
            pub = pub_by_node_id.get(node["id"], {})
            summary = (pub.get("summary") or "").strip().replace("\n", " ")
            if len(summary) > SNIPPET_CHARS:
                summary = summary[:SNIPPET_CHARS].rstrip() + "…"
            parts.append(f"- [{title}] (id: {node['id']})\n  {summary}")
    else:
        parts.append("- (none)")

    # C — current graph state (what the user already sees).
    parts.append("\n## Current graph state (from the previous turn)")
    if current_graph and current_graph.get("nodes"):
        by_type: dict[str, list[str]] = {}
        for n in current_graph["nodes"]:
            by_type.setdefault(n.get("type", "?"), []).append(n.get("label", n.get("id", "")))
        for t in ("publication", "topic", "region"):
            if by_type.get(t):
                parts.append(f"- {t}: {', '.join(by_type[t])}")
    else:
        parts.append("- (empty — this is the first turn)")

    return "\n".join(parts)


def build_answer_messages(query: str, context_block: str, history: list) -> list[dict]:
    messages = history_messages(history)
    messages.append({"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"})
    return messages


# --- stream-splitting helpers -----------------------------------------------


class SplitStreamer:
    """Split a model's token stream on a delimiter.

    Forward the part before the delimiter to the client token-by-token, and buffer
    everything so the JSON tail can be parsed once the stream completes. The
    delimiter can arrive split across deltas, so hold back up to len(delimiter)-1
    trailing characters until we're sure they aren't the start of the delimiter."""

    def __init__(self, delimiter: str):
        self.delimiter = delimiter
        self.buffer = ""
        self._emitted = 0
        self._hit_index = -1  # index of the delimiter in buffer, once seen

    @property
    def hit(self) -> bool:
        return self._hit_index != -1

    def feed(self, text: str) -> str:
        """Append a delta; return the chunk that's safe to emit to the client now."""
        self.buffer += text
        if self.hit:
            return ""

        idx = self.buffer.find(self.delimiter)
        if idx != -1:
            self._hit_index = idx
            chunk = self.buffer[self._emitted : idx]
            self._emitted = idx
            return chunk

        # Delimiter not seen yet — emit everything except a possible partial tail.
        safe_end = max(self._emitted, len(self.buffer) - (len(self.delimiter) - 1))
        chunk = self.buffer[self._emitted : safe_end]
        self._emitted = safe_end
        return chunk

    def flush(self) -> str:
        """If the delimiter never appeared, emit the held-back tail."""
        if self.hit:
            return ""
        chunk = self.buffer[self._emitted :]
        self._emitted = len(self.buffer)
        return chunk

    @property
    def before(self) -> str:
        return self.buffer[: self._hit_index] if self.hit else self.buffer

    @property
    def tail(self) -> str:
        """Text after the delimiter — the JSON payload, empty if never hit."""
        if not self.hit:
            return ""
        return self.buffer[self._hit_index + len(self.delimiter) :]


class CitationLinker:
    """Rewrite `[Exact Title]` citations into clickable markdown links in a stream.

    The answer model cites publications by exact title in square brackets and the
    chat panel renders markdown, so linking a recognised citation makes it clickable.
    A bracket span can arrive split across deltas, so text after a `[` is held until
    its `]`.

    Three subtleties keep the output well-formed:
      * Idempotency — a `]` immediately followed by `(` is already a markdown link
        (the model echoing a previously-linked citation from the conversation history,
        or writing one itself), so it's passed through untouched rather than re-wrapped
        into `(ref. (ref. …))`.
      * Multi-citation brackets — the model packs several titles into one bracket
        separated by `;`; each is resolved independently.
      * `url_for(title)` returns the link target or None; a bracket with no resolved
        title is emitted unchanged.

    `link_fmt(title, url)` formats one resolved citation (default: a markdown link);
    `wrap(body)` wraps the whole bracket's rendered body (default: unchanged) — the
    caller uses it to present citations as a parenthetical reference."""

    # An open `[` with no `]` within this many chars isn't a citation (titles are
    # short); release it literally so a stray bracket can't stall the stream.
    MAX_SPAN = 200

    def __init__(self, url_for, wrap=None, link_fmt=None):
        self.url_for = url_for
        self.wrap = wrap or (lambda body: body)
        self.link_fmt = link_fmt or (lambda title, url: f"[{title}]({url})")
        self._held = ""  # bracket content incl. '[' while collecting
        self._span = ""  # a closed '[...]' span awaiting one lookahead char
        self._state = "normal"  # normal | bracket | await_paren

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            self._step(ch, out)
        return "".join(out)

    def _step(self, ch: str, out: list[str]) -> None:
        if self._state == "await_paren":
            if ch == "(":
                # Already a markdown link `[text](…)` — leave it untouched.
                out.append(self._span + ch)
                self._span, self._state = "", "normal"
                return
            out.append(self._render(self._span))
            self._span, self._state = "", "normal"
            # fall through: this char is ordinary, handle it below

        if self._state == "normal":
            if ch == "[":
                self._held, self._state = "[", "bracket"
            else:
                out.append(ch)
        elif self._state == "bracket":
            if ch == "]":
                self._span, self._held, self._state = self._held + "]", "", "await_paren"
            elif ch == "[":
                out.append(self._held)  # the prior '[' wasn't a citation
                self._held = "["
            else:
                self._held += ch
                if len(self._held) > self.MAX_SPAN:
                    out.append(self._held)
                    self._held, self._state = "", "normal"

    def flush(self) -> str:
        """Emit any text still held at end of stream."""
        out = ""
        if self._state == "await_paren":
            out = self._render(self._span)
        elif self._state == "bracket":
            out = self._held
        self._held = self._span = ""
        self._state = "normal"
        return out

    def _render(self, span: str) -> str:
        rendered: list[str] = []
        resolved_any = False
        for part in span[1:-1].split(";"):
            title = part.strip()
            url = self.url_for(title) if title else None
            if url:
                rendered.append(self.link_fmt(title, url))
                resolved_any = True
            elif title:
                rendered.append(title)
        if not resolved_any:
            return span  # nothing matched — leave the citation as written
        return self.wrap("; ".join(rendered))
