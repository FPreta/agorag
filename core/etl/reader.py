"""ETL step 1 — read source publications and diff them against the store.

The Reader is the scraper, plus an incremental check: it loads whatever is
already persisted in publications.jsonl and reports which incoming publications
are genuinely new, so the rest of the pipeline only has to embed the additions.

Sources:
    source="sample"  -> the offline SAMPLE_PUBLICATIONS (default; no network)
    source="scrape"  -> crawl the live Agora archive (TYPO3 CMS)

Agora runs TYPO3. The CSS selectors below follow the documented page structure,
but a theme change can shift them — if a field comes back empty across the board,
inspect the live HTML and adjust the selector. Every extraction is defensive: a
publication missing key findings/figures/experts is still captured.

Region/country is not on the detail page; it lives only in the listing's
"Region / Country" facet. After the detail scrape, a second pass walks each facet
(/publications/filter/tab-1/2-<id>) and back-fills every publication's `regions`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .. import config
from .sample_data import sample_publications

BASE = "https://www.agora-energiewende.org"
LISTING = BASE + "/publications"
TOTAL_LISTING_PAGES = 29
DELAY_SECONDS = 1.5
HEADERS = {
    "User-Agent": "AgoraResearchExplorer/1.0 (+research prototype; polite crawl, 1.5s delay)"
}

# Region/country is not exposed on a publication's detail page. It lives only in
# the listing's "Region / Country" facet: /publications/filter/tab-1/2-<id> shows
# the publications tagged with that region. The facet ids and their labels (the
# site has no machine-readable mapping) are pinned here. The enrichment pass walks
# each facet and back-fills every publication's `regions` list.
REGION_FILTERS = {
    26: "Brazil",
    21: "China",
    18: "European Union",
    20: "France",
    19: "Germany",
    36: "Indonesia",
    14: "Japan",
    37: "Kazakhstan",
    38: "Pakistan",
    23: "Poland",
    39: "South Africa",
    16: "South Korea",
    17: "Southeast Asia",
    13: "Southeast Europe",
    40: "Thailand",
    22: "Turkey",
    41: "Vietnam",
}
# Safety cap on facet pages: the listing paginates ~9 per page and the largest
# facet has well under this many publications. The walk normally stops earlier
# when a page yields no new slugs (see _collect_region_map).
MAX_FILTER_PAGES = 60


@dataclass
class ReadResult:
    """The merged view after reading a source against the existing store.

    `all` is every publication that should be persisted (existing ∪ incoming, with
    incoming winning on id collisions); `new` is the subset whose ids weren't in the
    store before; `existing` is what was already persisted."""

    all: list[dict]
    new: list[dict] = field(default_factory=list)
    existing: list[dict] = field(default_factory=list)

    @property
    def has_new(self) -> bool:
        return bool(self.new)


class Reader:
    def __init__(self, data_dir: str | Path | None = None):
        base = Path(data_dir) if data_dir is not None else config.data_dir()
        self.publications_path = base / config.PUBLICATIONS_FILE

    # --- store ----------------------------------------------------------

    def existing_publications(self) -> list[dict]:
        """Publications already persisted in the store (empty if none yet)."""
        if not self.publications_path.exists():
            return []
        out: list[dict] = []
        with self.publications_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # --- public ---------------------------------------------------------

    def read(self, source: str = "sample", limit: int | None = None) -> ReadResult:
        """Read the source, then diff it against the existing store.

        `source` is "sample" (offline records) or "scrape" (live crawl). `limit`
        caps how many publications are read (a smoke-test aid for the scraper)."""
        if source == "sample":
            incoming = sample_publications()
        elif source == "scrape":
            incoming = self.scrape(limit=limit)
        else:
            raise ValueError(f"Unknown source: {source!r} (expected 'sample' or 'scrape')")
        if limit is not None:
            incoming = incoming[:limit]
        return self._diff(incoming)

    def _diff(self, incoming: list[dict]) -> ReadResult:
        existing = self.existing_publications()
        existing_ids = {p["id"] for p in existing}
        merged: dict[str, dict] = {p["id"]: p for p in existing}
        new: list[dict] = []
        for pub in incoming:
            if pub["id"] not in existing_ids:
                new.append(pub)
            merged[pub["id"]] = pub  # incoming wins on collision (refreshes fields)
        return ReadResult(all=list(merged.values()), new=new, existing=existing)

    # --- scraping -------------------------------------------------------

    def scrape(self, limit: int | None = None) -> list[dict]:
        """Crawl the live archive, returning publication records with regions filled."""
        session = requests.Session()
        urls = self._collect_publication_urls(session)
        if limit is not None:
            urls = urls[:limit]

        pubs: list[dict] = []
        for i, url in enumerate(urls, start=1):
            print(f"[reader] publication {i}/{len(urls)} -> {url}")
            pub = self._scrape_publication(session, url)
            if pub and pub.get("title"):
                pubs.append(pub)
            time.sleep(DELAY_SECONDS)

        # Region/country only exists on the listing facets, so enrich in a second
        # pass once every detail page has been read.
        self._enrich_regions(session, pubs)
        return pubs

    @staticmethod
    def _get(session, url: str):

        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            print(f"  ! failed to fetch {url}: {exc}")
            return None

    @staticmethod
    def _listing_page_url(page: int) -> str:
        return LISTING if page == 1 else f"{LISTING}/page/{page}"

    @staticmethod
    def _publication_slugs(soup) -> list[str]:
        """Publication slugs from a listing/facet page, in order.

        A real publication link is /publications/<slug> — exactly one path segment
        after /publications/. Anything with a further "/" (e.g. /publications/filter/…
        or /publications/page/…) is navigation, not a publication."""
        slugs: list[str] = []
        seen: set[str] = set()
        for a in soup.select("a[href*='/publications/']"):
            href = urljoin(BASE, a.get("href", "").split("?")[0].rstrip("/"))
            tail = href.split("/publications/", 1)[-1]
            if href.startswith(BASE + "/publications/") and tail and "/" not in tail:
                if tail not in seen:
                    seen.add(tail)
                    slugs.append(tail)
        return slugs

    def _collect_publication_urls(self, session) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for page in range(1, TOTAL_LISTING_PAGES + 1):
            url = self._listing_page_url(page)
            print(f"[reader] listing page {page}/{TOTAL_LISTING_PAGES} -> {url}")
            soup = self._get(session, url)
            if soup is None:
                continue
            for slug in self._publication_slugs(soup):
                href = f"{BASE}/publications/{slug}"
                if href not in seen:
                    seen.add(href)
                    urls.append(href)
            time.sleep(DELAY_SECONDS)
        print(f"[reader] collected {len(urls)} publication URLs")
        return urls

    @staticmethod
    def _filter_page_url(filter_id: int, page: int) -> str:
        base = f"{LISTING}/filter/tab-1/2-{filter_id}"
        return base if page == 1 else f"{base}/page/{page}"

    def _collect_region_map(self, session) -> dict[str, list[str]]:
        """Map each publication slug to the region/country facets it appears under.

        The facet's pagination has no last-page marker and clamps out-of-range pages
        to the final page, so we stop once a page (after the first) yields no new
        slugs. A publication can sit under several facets, so regions accumulate."""
        slug_regions: dict[str, list[str]] = {}
        for filter_id, region in REGION_FILTERS.items():
            collected: set[str] = set()
            for page in range(1, MAX_FILTER_PAGES + 1):
                url = self._filter_page_url(filter_id, page)
                soup = self._get(session, url)
                time.sleep(DELAY_SECONDS)
                if soup is None:
                    break
                page_slugs = self._publication_slugs(soup)
                new = [s for s in page_slugs if s not in collected]
                if page > 1 and not new:
                    break
                collected.update(page_slugs)
                if not page_slugs:
                    break
            print(f"[reader] region {region}: {len(collected)} publications")
            for slug in collected:
                regions = slug_regions.setdefault(slug, [])
                if region not in regions:
                    regions.append(region)
        return slug_regions

    def _enrich_regions(self, session, pubs: list[dict]) -> None:
        """Back-fill the `regions` field of each publication in place."""
        slug_regions = self._collect_region_map(session)
        tagged = 0
        for pub in pubs:
            regions = sorted(slug_regions.get(pub.get("id", ""), []))
            pub["regions"] = regions
            if regions:
                tagged += 1
        print(f"[reader] tagged {tagged}/{len(pubs)} publications with a region/country")

    @staticmethod
    def _text(node) -> str:
        return node.get_text(" ", strip=True) if node else ""

    @staticmethod
    def _heading(soup, label: str):
        """A heading (h1-h4) whose text matches `label` (case-insensitive)."""
        pattern = re.compile(rf"^\s*{re.escape(label)}\s*$", re.I)
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            if pattern.match(tag.get_text(strip=True)):
                return tag
        return None

    @staticmethod
    def _slug_from_url(url: str) -> str:
        return url.rstrip("/").split("/")[-1]

    def _scrape_publication(self, session, url: str) -> dict | None:
        soup = self._get(session, url)
        if soup is None:
            return None

        _text = self._text
        title = _text(soup.select_one("h1.intro__title")) or _text(soup.select_one("h1"))
        subtitle = _text(soup.select_one(".intro__description"))

        # Summary: the "Summary" heading sits inside a <header>, so the body text is
        # not a sibling of the heading — it lives in a .rich-text__body div within the
        # same module section. Pull that container's text.
        summary = ""
        summary_h = self._heading(soup, "Summary")
        if summary_h:
            module = summary_h.find_parent("section") or summary_h.find_parent()
            body = module.select_one(".rich-text__body, .ce-bodytext") if module else None
            summary = _text(body)

        # Key findings: numbered list items after a "Key findings" heading.
        key_findings = []
        kf_h = self._heading(soup, "Key findings")
        if kf_h:
            container = kf_h.find_next(["ol", "ul", "div"])
            if container:
                for i, li in enumerate(container.find_all("li"), start=1):
                    bold = li.find(["strong", "b"])
                    headline = _text(bold)
                    body = _text(li)
                    if headline and body.startswith(headline):
                        body = body[len(headline) :].strip(" .:-")
                    key_findings.append(
                        {"number": i, "headline": headline or body[:80], "body": body}
                    )

        # Topics: the intro tag list links to /topics/<slug>.
        topics = []
        for a in soup.select(".intro__tags a.intro__tag-link, .intro__tags a"):
            name = a.get_text(strip=True).lstrip("#").strip()
            if name and name not in topics:
                topics.append(name)

        # Figures: cards with title, page reference and a PNG url.
        figures = []
        for i, img in enumerate(soup.select("img[src$='.png']"), start=1):
            src = urljoin(BASE, img.get("src", ""))
            if "abb" in src.lower() or "/AutomaticFiles/" in src:
                figures.append(
                    {"number": i, "title": img.get("alt", "").strip(), "page": None, "png_url": src}
                )

        # Experts: "Our experts" section is a ul.team of li.member cards, each with a
        # name link (/about-us/team/<slug>) and a position line.
        experts = []
        for li in soup.select("ul.team li.member"):
            a = li.select_one("h2.member__name a.member__link") or li.select_one(".member__name a")
            name = _text(a)
            if name:
                experts.append(
                    {
                        "name": name,
                        "role": _text(li.select_one(".member__position")),
                        "profile_url": urljoin(BASE, a.get("href", "")),
                    }
                )

        # Bibliographical data: a definition list keyed by term. The reliable source
        # for authors, the publication date and the suggested citation.
        biblio = {}
        for section in soup.select("dl.meta-list .meta-list__section"):
            term = _text(section.select_one(".meta-list__term"))
            value = _text(section.select_one(".meta-list__definition"))
            if term:
                biblio[term.lower()] = value

        # Authors come as one string with "(affiliation)" annotations, separated by
        # commas and/or "and" ("A, B and C" or "A and B").
        authors_raw = re.sub(r"\s*\([^)]*\)", "", biblio.get("authors", ""))
        authors = [a.strip() for a in re.split(r",|\band\b", authors_raw) if a.strip()]

        citation = biblio.get("suggested citation", "")

        # Date: the intro meta carries the display date; fall back to biblio data.
        date = _text(soup.select_one(".intro__meta--date .intro__format")) or biblio.get(
            "publication date", ""
        )
        format_ = _text(soup.select_one(".intro__meta--format .intro__format"))

        # Related: teaser cards in the "Related" module.
        related = []
        related_h = self._heading(soup, "Related")
        if related_h:
            module = related_h.find_parent("section") or related_h.find_parent()
            for li in module.select("li.teaser") if module else []:
                a = li.select_one(".teaser__title a")
                if a:
                    related.append({"title": _text(a), "url": urljoin(BASE, a.get("href", ""))})

        # PDF: a link in the "Downloads" area.
        pdf_url = ""
        for a in soup.select("a[href$='.pdf']"):
            href = urljoin(BASE, a.get("href", ""))
            if "/fileadmin/" in href:
                pdf_url = href
                break

        return {
            "id": self._slug_from_url(url),
            "url": url,
            "title": title,
            "subtitle": subtitle,
            "date": date,
            "format": format_,
            "topics": topics,
            "regions": [],  # filled by the region enrichment pass
            "summary": summary,
            "key_findings": key_findings,
            "authors": authors,
            "citation": citation,
            "pdf_url": pdf_url,
            "figures": figures,
            "experts": experts,
            "related": related,
        }
