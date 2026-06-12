"""Offline sample publications — a small, realistic stand-in for the live scrape.

Scraping the live Agora site means 29 listing pages + ~260 publication pages at a
polite 1.5s delay (~7 minutes) and depends on the live HTML staying stable. These
records mirror the real schema so the whole pipeline (graph -> chunks -> Qdrant ->
gradio) can run immediately, offline. The Reader serves them when `source="sample"`.
"""

from __future__ import annotations

SAMPLE_PUBLICATIONS = [
    {
        "id": "reforming-power-purchase-agreements-for-flexible-coal-power",
        "url": (
            "https://www.agora-energiewende.org/publications/"
            "reforming-power-purchase-agreements-for-flexible-coal-power"
        ),
        "title": "Reforming power purchase agreements for flexible coal power",
        "subtitle": "A focus on South and Southeast Asian power systems",
        "date": "2026-06-02",
        "format": "Study",
        "topics": [
            "Coal phase-out",
            "Financing and investing in the transition",
            "Scaling up renewable energy",
        ],
        "regions": ["Southeast Asia"],
        "summary": (
            "Using existing thermal power plants for system flexibility can ease "
            "the integration of renewables across Asia's young coal fleets, but "
            "rigid power purchase agreements stand in the way. This study sets out "
            "contract reforms that let coal plants ramp rather than run baseload."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Asia's young coal fleets can technically support flexible operation",
                "body": (
                    "As flexibility requirements across Asia's power systems "
                    "increase with rising renewable shares, the region's relatively "
                    "young coal fleets can technically support cycling and ramping "
                    "with limited retrofits."
                ),
            },
            {
                "number": 2,
                "headline": "Take-or-pay clauses lock in must-run coal",
                "body": (
                    "Most power purchase agreements in the region contain capacity "
                    "payments and take-or-pay clauses that reward baseload "
                    "generation, removing any incentive for plants to operate flexibly."
                ),
            },
            {
                "number": 3,
                "headline": "Renegotiated contracts can pay for flexibility",
                "body": (
                    "Restructured agreements that compensate availability and "
                    "ramping rather than energy volume can align plant economics "
                    "with the needs of a renewable-heavy grid."
                ),
            },
        ],
        "authors": ["Ernst Kuneman", "Dr. Anatole Boute"],
        "citation": (
            "Agora Energiewende (2026): Reforming power purchase agreements for "
            "flexible coal power."
        ),
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2026/sample/reforming-ppa.pdf"
        ),
        "figures": [
            {
                "number": 1,
                "title": "Repurpose, reserve and retire – a coal exit strategy",
                "page": 12,
                "png_url": (
                    "https://www.agora-energiewende.org/fileadmin/AutomaticFiles/1863/abb-01.png"
                ),
            }
        ],
        "experts": [
            {
                "name": "Ernst Kuneman",
                "role": "Senior Associate Power System Transformations",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/ernst-kuneman",
            },
            {
                "name": "Dr. Anatole Boute",
                "role": "Senior Associate",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/anatole-boute",
            },
        ],
        "related": [],
    },
    {
        "id": "breaking-free-from-fossil-gas-in-industry",
        "url": (
            "https://www.agora-energiewende.org/publications/"
            "breaking-free-from-fossil-gas-in-industry"
        ),
        "title": "Breaking free from fossil gas in industry",
        "subtitle": "Electrification and hydrogen pathways for European manufacturing",
        "date": "2026-03-18",
        "format": "Study",
        "topics": ["Hydrogen", "Industry decarbonisation", "European Green Deal"],
        "regions": ["European Union", "Germany"],
        "summary": (
            "Europe's industrial gas demand can be cut sharply this decade through "
            "direct electrification of low- and medium-temperature heat, reserving "
            "hydrogen for genuinely hard-to-abate processes. The report quantifies "
            "the split and the infrastructure each pathway needs."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Most industrial heat below 500°C can be electrified today",
                "body": (
                    "Industrial heat pumps and electric boilers can already cover "
                    "the majority of low- and medium-temperature process heat, "
                    "displacing fossil gas without waiting for hydrogen."
                ),
            },
            {
                "number": 2,
                "headline": "Hydrogen should be reserved for hard-to-abate processes",
                "body": (
                    "Scarce and expensive renewable hydrogen delivers the most "
                    "value in steel, ammonia and high-temperature chemistry rather "
                    "than in heat applications that electricity can serve."
                ),
            },
        ],
        "authors": ["Dimitri Pescia", "Fabian Barrera"],
        "citation": "Agora Industry (2026): Breaking free from fossil gas in industry.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2026/sample/"
            "fossil-gas-industry.pdf"
        ),
        "figures": [
            {
                "number": 1,
                "title": "Electrification vs. hydrogen by temperature band",
                "page": 8,
                "png_url": (
                    "https://www.agora-energiewende.org/fileadmin/AutomaticFiles/1901/abb-02.png"
                ),
            }
        ],
        "experts": [
            {
                "name": "Dimitri Pescia",
                "role": "Programme Lead Industry",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/dimitri-pescia",
            }
        ],
        "related": [],
    },
    {
        "id": "power-market-design-for-the-renewable-age",
        "url": (
            "https://www.agora-energiewende.org/publications/"
            "power-market-design-for-the-renewable-age"
        ),
        "title": "Power market design for the renewable age",
        "subtitle": "Aligning short-term markets with a weather-driven grid",
        "date": "2025-11-27",
        "format": "Report",
        "topics": ["Electricity market design", "Scaling up renewable energy"],
        "regions": ["European Union"],
        "summary": (
            "As wind and solar come to dominate supply, electricity markets must "
            "reward flexibility and locational signals. This report reviews reform "
            "options for intraday and balancing markets and their interaction with "
            "long-term contracts."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Short-term markets need sharper scarcity and locational signals",
                "body": (
                    "Intraday and balancing markets should reflect real-time "
                    "scarcity and grid constraints so that flexible assets are "
                    "dispatched where and when the system needs them."
                ),
            },
            {
                "number": 2,
                "headline": "Long-term contracts can hedge investors without dulling signals",
                "body": (
                    "Two-sided contracts for difference can stabilise revenues for "
                    "renewables while preserving the short-term price signals that "
                    "drive efficient operation."
                ),
            },
        ],
        "authors": ["Dimitri Pescia", "Murielle Gagnebin"],
        "citation": "Agora Energiewende (2025): Power market design for the renewable age.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2025/sample/market-design.pdf"
        ),
        "figures": [],
        "experts": [
            {
                "name": "Murielle Gagnebin",
                "role": "Senior Associate Electricity Markets",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/murielle-gagnebin",
            }
        ],
        "related": [],
    },
    {
        "id": "the-future-of-district-heating",
        "url": "https://www.agora-energiewende.org/publications/the-future-of-district-heating",
        "title": "The future of district heating",
        "subtitle": "Decarbonising heat networks in dense cities",
        "date": "2025-09-09",
        "format": "Study",
        "topics": ["Buildings and heat transition", "Scaling up renewable energy"],
        "regions": ["Germany", "European Union"],
        "summary": (
            "District heating can decarbonise quickly by tapping large-scale heat "
            "pumps, waste heat and seasonal storage. The study maps the transition "
            "for European cities and the regulatory barriers that slow it."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Large heat pumps and waste heat can anchor clean networks",
                "body": (
                    "River, sewage and data-centre waste heat lifted by "
                    "megawatt-scale heat pumps can supply the bulk of urban "
                    "district heat at competitive cost."
                ),
            },
            {
                "number": 2,
                "headline": "Seasonal storage smooths winter demand",
                "body": (
                    "Pit and aquifer thermal storage lets networks bank summer "
                    "heat for winter, cutting peak capacity needs and fossil back-up."
                ),
            },
        ],
        "authors": ["Frank Peter", "Murielle Gagnebin"],
        "citation": "Agora Energiewende (2025): The future of district heating.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2025/sample/district-heating.pdf"
        ),
        "figures": [
            {
                "number": 1,
                "title": "Heat sources for a decarbonised network",
                "page": 15,
                "png_url": (
                    "https://www.agora-energiewende.org/fileadmin/AutomaticFiles/1777/abb-03.png"
                ),
            }
        ],
        "experts": [
            {
                "name": "Frank Peter",
                "role": "Director",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/frank-peter",
            }
        ],
        "related": [],
    },
    {
        "id": "coal-to-clean-in-india",
        "url": "https://www.agora-energiewende.org/publications/coal-to-clean-in-india",
        "title": "Coal to clean in India",
        "subtitle": "Managing a just and financeable transition",
        "date": "2025-07-21",
        "format": "Study",
        "topics": ["Coal phase-out", "Financing and investing in the transition"],
        "regions": ["India", "Southeast Asia"],
        "summary": (
            "India can begin retiring its oldest, least efficient coal units while "
            "protecting workers and grid reliability, provided transition finance "
            "and reskilling arrive in time. The study quantifies the fleet at risk "
            "and the investment needed."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Old subcritical units are the first candidates for retirement",
                "body": (
                    "A large share of India's emissions and local pollution comes "
                    "from ageing subcritical plants that are increasingly "
                    "uneconomic against new solar."
                ),
            },
            {
                "number": 2,
                "headline": "Transition finance must reach affected regions",
                "body": (
                    "Concessional capital and reskilling programmes targeted at "
                    "coal regions are essential to make early retirement just and "
                    "politically durable."
                ),
            },
        ],
        "authors": ["Ernst Kuneman", "Fabian Barrera"],
        "citation": "Agora Energiewende (2025): Coal to clean in India.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2025/sample/"
            "coal-to-clean-india.pdf"
        ),
        "figures": [],
        "experts": [
            {
                "name": "Fabian Barrera",
                "role": "Associate International",
                "profile_url": "https://www.agora-energiewende.org/about-us/team/fabian-barrera",
            }
        ],
        "related": [],
    },
    {
        "id": "scaling-green-hydrogen-imports",
        "url": "https://www.agora-energiewende.org/publications/scaling-green-hydrogen-imports",
        "title": "Scaling green hydrogen imports",
        "subtitle": "Building resilient supply chains for Europe",
        "date": "2025-05-14",
        "format": "Report",
        "topics": ["Hydrogen", "Financing and investing in the transition", "European Green Deal"],
        "regions": ["European Union", "Brazil"],
        "summary": (
            "Europe will import a meaningful share of its future hydrogen. This "
            "report assesses cost-competitive export regions, the role of "
            "derivatives like ammonia, and the contracts needed to de-risk "
            "first-mover projects."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Ammonia is the near-term carrier of choice",
                "body": (
                    "For long-distance trade before dedicated pipelines exist, "
                    "ammonia offers the most mature shipping and handling chain "
                    "despite reconversion losses."
                ),
            },
            {
                "number": 2,
                "headline": "Offtake contracts unlock first-mover investment",
                "body": (
                    "Long-term offtake agreements and public guarantees are needed "
                    "to bring early export projects in regions such as Brazil to "
                    "final investment decision."
                ),
            },
        ],
        "authors": ["Dimitri Pescia", "Dr. Anatole Boute"],
        "citation": "Agora Industry (2025): Scaling green hydrogen imports.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2025/sample/hydrogen-imports.pdf"
        ),
        "figures": [
            {
                "number": 1,
                "title": "Delivered cost of hydrogen by export region",
                "page": 11,
                "png_url": (
                    "https://www.agora-energiewende.org/fileadmin/AutomaticFiles/1742/abb-04.png"
                ),
            }
        ],
        "experts": [],
        "related": [],
    },
    {
        "id": "renewables-grids-and-permitting",
        "url": "https://www.agora-energiewende.org/publications/renewables-grids-and-permitting",
        "title": "Renewables, grids and permitting",
        "subtitle": "Removing the bottlenecks to faster deployment",
        "date": "2025-02-03",
        "format": "Impulse",
        "topics": ["Scaling up renewable energy", "Electricity market design"],
        "regions": ["European Union", "France", "Germany"],
        "summary": (
            "Permitting delays and grid connection queues, not technology cost, "
            "are now the binding constraint on renewable deployment in Europe. "
            "This impulse sets out administrative and planning reforms to clear "
            "the backlog."
        ),
        "key_findings": [
            {
                "number": 1,
                "headline": "Permitting is now the binding constraint",
                "body": (
                    "Project timelines are dominated by multi-year permitting and "
                    "grid-connection waits rather than by the falling cost of wind "
                    "and solar hardware."
                ),
            },
            {
                "number": 2,
                "headline": "Go-to areas and digital queues speed deployment",
                "body": (
                    "Pre-designated renewable acceleration areas and transparent, "
                    "digital grid-connection queues can compress timelines without "
                    "weakening environmental safeguards."
                ),
            },
        ],
        "authors": ["Murielle Gagnebin", "Frank Peter"],
        "citation": "Agora Energiewende (2025): Renewables, grids and permitting.",
        "pdf_url": (
            "https://www.agora-energiewende.org/fileadmin/Projekte/2025/sample/permitting.pdf"
        ),
        "figures": [],
        "experts": [],
        "related": [],
    },
]


def sample_publications() -> list[dict]:
    """A copy of the sample publication records (safe for the caller to mutate)."""
    import copy

    return copy.deepcopy(SAMPLE_PUBLICATIONS)
