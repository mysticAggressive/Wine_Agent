# Madrid Wine Agent — Development Plan

This is the build plan for the Madrid Wine Agent. It is written so that a coding agent
(or a person) can pick up any single task, do it without reading the rest of the code
base, and know when it is finished.

**How to use this file**

1. Read sections 1–4 once. They hold the rules, the decisions already made, and the
   shared data contracts. Every task depends on them.
2. Pick the lowest-numbered task whose dependencies are done (section 5 has the order).
3. Do exactly what the task lists, run its acceptance checks, then tick it off in
   section 14.
4. Do not change a contract in section 4 without updating this file in the same commit.

---

## 1. What we are building

An AI agent that takes wine preferences (style, price, shop vs. restaurant, food to
pair) and searches real Madrid wine shops and restaurants, picks the best matches,
pairs wine to food, and shows ranked results with a walking route on a live map.

The recommendation engine is a **Constraint Satisfaction Problem** solved with
backtracking, forward checking, arc consistency (AC-3) and MRV + LCV ordering. The
ranking formula is an **evaluation function** `Eval(s) = Σ wᵢ·fᵢ`. The route is found
with **A\***. An optional "sommelier vs. you" mode uses **minimax with alpha-beta
pruning**.

The three things the grading rubric looks for, and where each lives:

| Rubric item | Where it is delivered |
|---|---|
| Dynamic | Preference panel: any input change re-runs the solver and redraws results (task 7.2) |
| Presentation layer | Folium map with pins, route, and a "why this pick" popup (task 7.3) |
| Result monitoring | Ranked table, per-pick score breakdown, per-constraint prune counts, search statistics (task 7.4) |

---

## 2. Rules for every agent

1. **Python only.** Target Python 3.11 or newer. No JavaScript.
2. **No network at run time.** The Streamlit app and the tests read only files under
   `data/`. Only the scripts named `fetch_*.py` may touch the network. Map tiles are the
   one exception (the browser loads them).
3. **No invented venues or wines.** Venues come from OpenStreetMap, wines from the Kaggle
   dataset. The two derived layers (inventory and pairing rules, section 3) are the only
   non-source data, and the UI and README must label them as derived.
4. **The AI techniques are written by hand.** `csp_solver.py`, the A\* in
   `search_route.py`, and `minimax.py` must be our own code, modelled on aima-python
   (`csp.py`, `search.py`, `games.py`). NetworkX and OSMnx may be used to *hold* the
   graph and to *check* our answers in tests, not to do the search for us.
5. **Pure logic stays out of the UI.** Nothing under `agent/` may import `streamlit` or
   `folium`. `app.py` and `ui/` only call `agent/` functions.
6. **Determinism.** Same inputs give the same output. Any randomness is seeded from a
   stable id (see task 1.4). Sort ties by id.
7. **Every task ships with tests.** `pytest` must pass before a task is ticked off.
8. **Type hints and docstrings** on every public function. Run `ruff check .` and
   `ruff format .` before committing.
9. **One task = one commit** (or a few small ones), message format
   `task 4.2: implement AC-3`.
10. **Attribution.** The app footer and README must show
    "© OpenStreetMap contributors" (ODbL requirement) and credit the Kaggle dataset.

---

## 3. Decisions already made

These fill gaps the brief leaves open. Treat them as settled unless the project owner
says otherwise. Items marked **OWNER CHECK** are assumptions the owner should confirm.

### 3.1 Scope of the map
- Centre point: Puerta del Sol, `(40.4168, -3.7038)`.
- Radius: 4 000 m. Both the venue fetch and the street graph use this same circle.
- Travel mode: walking (`network_type="walk"`).
- The user's start point defaults to Puerta del Sol and can be changed (task 7.2).

### 3.2 Which wines a venue carries — **OWNER CHECK**
No free data source says which Madrid venue stocks which wine. We therefore build a
**derived inventory**: a deterministic assignment of dataset wines to venues, seeded by
the venue's OSM id (task 1.4). It is reproducible, committed to the repo, and labelled
"simulated inventory" in the UI. Without it the `wine`–`venue` constraint has nothing
to work on. If the owner has real wine lists for some venues, they replace rows in
`data/inventory.csv` and nothing else changes.

### 3.3 Prices at restaurants and bars — **OWNER CHECK**
The dataset price is a retail bottle price in euros. The price at a venue is
`retail × markup`, with markup `1.0` for shops, `2.0` for bars and pubs, `2.5` for
restaurants. The markups live in `agent/config.py`. The budget constraint applies to
the price at the venue.

### 3.4 Tasting notes
The brief mentions tasting notes, but the named dataset has none. Its columns are
`winery, wine, year, rating, num_reviews, country, region, price, type, body, acidity`.
Taste matching therefore uses `body`, `acidity`, `type`, `region` and `rating`.

### 3.5 The food side
- A fixed list of **dishes** lives in `data/pairing_rules.json` (task 3.1). Each dish
  has rules describing the wine that suits it.
- Which dishes a venue can serve comes from its real OSM `cuisine` tag, through a
  cuisine → dishes map in the same file. Shops can "serve" every dish (the user is
  cooking at home). A restaurant with no `cuisine` tag gets the `spanish` dish set, and
  the venue is flagged `cuisine_assumed = true`. **OWNER CHECK**

### 3.6 Opening hours
- Parsed from the OSM `opening_hours` tag with the `opening-hours-py` package.
- "Open now" is evaluated at a time the user can set (default: now, `Europe/Madrid`).
- A venue with a missing or unparseable tag has state `unknown`. By default `unknown`
  venues are kept and badged "hours unknown". A checkbox lets the user exclude them.

### 3.7 CSP formulation

| Variable | Domain |
|---|---|
| `wine` | all wine ids |
| `venue` | all venue ids |
| `dish` | all dish ids (a single value if the user chose a dish) |

Hard constraints:

| Id | Scope | Meaning |
|---|---|---|
| `style` | `wine` | wine style is one the user accepts |
| `venue_type` | `venue` | venue is a shop / a restaurant-or-bar, as the user asked |
| `open_now` | `venue` | venue is open at the chosen time (see 3.6) |
| `max_distance` | `venue` | straight-line distance from the user ≤ the user's limit |
| `stocks` | `wine, venue` | the venue carries the wine (inventory) |
| `budget` | `wine, venue` | price of that wine at that venue ≤ budget |
| `serves` | `venue, dish` | the venue can serve the dish |
| `pairs` | `wine, dish` | pairing score ≥ `MIN_PAIRING_SCORE` (0.5) |

Soft preferences (the evaluation function), each feature in `[0, 1]`:

| Feature | Formula |
|---|---|
| `taste` | `1 − (0.6·|body − want_body|/3 + 0.4·|acidity − want_acidity|/2)` |
| `quality` | `(rating − min_rating) / (max_rating − min_rating)` over the cleaned dataset |
| `price_fit` | `1 − price_at_venue / budget`, clipped to `[0, 1]` |
| `pairing` | pairing score from the pairing engine |
| `distance` | `1 − distance_m / max_distance_m`, clipped to `[0, 1]` |

`Eval = Σ wᵢ·fᵢ / Σ wᵢ`, weights from the user's sliders (0–10 each, default 5). If all
weights are 0, use equal weights.

A **solution** is a `(wine, venue, dish)` triple that passes every hard constraint. The
solver returns the top `k` by `Eval` (default `k = 10`), with at most
`MAX_PER_VENUE = 2` results per venue so the list is not one shop repeated.

### 3.8 Dependencies

`requirements.txt` (minimum versions; these were current when the plan was written):

```
streamlit>=1.40
streamlit-folium>=0.25
folium>=0.19
osmnx>=2.0
networkx>=3.3
pandas>=2.2
opening-hours-py>=2.0
requests>=2.32
kagglehub>=0.3
```

`requirements-dev.txt`: `pytest>=8`, `ruff>=0.6`.

---

## 4. Shared contracts

Agents working in parallel rely on these. Build them first (phase 2) and do not drift.

### 4.1 File layout

```
Wine_Agent/                      (repo root)
  app.py                         Streamlit entry point
  ui/
    __init__.py
    sidebar.py                   preference panel
    map_view.py                  Folium map
    results_view.py              table, breakdown, diagnostics
  agent/
    __init__.py
    config.py                    constants (centre, radius, markups, thresholds)
    models.py                    dataclasses (4.2)
    data_loader.py               load data/ files into models
    preferences.py               Preferences + validation
    opening.py                   opening-hours evaluation
    pairing.py                   pairing engine
    csp_solver.py                generic CSP + AC-3 + backtracking
    wine_csp.py                  builds the wine CSP from data + preferences
    scoring.py                   evaluation function
    recommend.py                 one entry point: recommend(prefs) -> RecommendationResult
    search_route.py              A* routing
    minimax.py                   optional sommelier game
  data/
    fetch_wines.py               downloads the Kaggle CSV
    clean_wines.py               raw CSV -> wines.csv
    fetch_osm.py                 Overpass -> venues_madrid.json
    fetch_graph.py               OSMnx -> madrid_walk.graphml
    build_inventory.py           venues + wines -> inventory.csv
    raw/wines_SPA.csv            raw download (see task 1.1 about committing it)
    wines.csv                    cleaned wines (committed)
    venues_madrid.json           cached venues (committed)
    inventory.csv                derived inventory (committed)
    pairing_rules.json           dishes, rules, cuisine map (committed)
    madrid_walk.graphml          street graph (committed if < 50 MB, see task 1.3)
  tests/
    conftest.py                  small fixtures (5 wines, 4 venues, 3 dishes)
    test_*.py                    one file per agent module
  requirements.txt
  requirements-dev.txt
  pyproject.toml                 ruff + pytest settings
  README.md
  DEVELOPMENT_PLAN.md
```

### 4.2 Data models — `agent/models.py`

```python
from dataclasses import dataclass, field
from enum import Enum

class Style(str, Enum):
    RED = "red"; WHITE = "white"; ROSE = "rose"
    SPARKLING = "sparkling"; FORTIFIED = "fortified"; UNKNOWN = "unknown"

class VenueKind(str, Enum):
    SHOP = "shop"              # OSM shop=wine, shop=alcohol
    RESTAURANT = "restaurant"  # OSM amenity=restaurant
    BAR = "bar"                # OSM amenity=bar, amenity=pub

class OpenState(str, Enum):
    OPEN = "open"; CLOSED = "closed"; UNKNOWN = "unknown"

@dataclass(frozen=True)
class Wine:
    wine_id: str          # "w0001" …, stable across runs
    winery: str
    name: str
    year: int | None      # None for non-vintage
    rating: float
    num_reviews: int
    region: str
    retail_price: float   # euros
    wine_type: str        # original `type` column, "" if missing
    style: Style
    body: int             # 1–5
    acidity: int          # 1–3
    imputed: bool         # True if body or acidity was filled in

@dataclass(frozen=True)
class Venue:
    venue_id: str         # "node/123456" or "way/123456"
    name: str
    kind: VenueKind
    lat: float
    lon: float
    address: str          # "" if OSM has none
    opening_hours: str    # raw OSM tag, "" if missing
    cuisines: tuple[str, ...]   # lower-case, split on ";"
    cuisine_assumed: bool
    website: str

@dataclass(frozen=True)
class Dish:
    dish_id: str          # "grilled_red_meat"
    label: str            # "Grilled red meat"

@dataclass(frozen=True)
class Candidate:          # one CSP solution
    wine_id: str
    venue_id: str
    dish_id: str
    price_at_venue: float
    distance_m: float
    features: dict[str, float]   # taste, quality, price_fit, pairing, distance
    score: float                 # Eval
    reasons: tuple[str, ...]     # human-readable "why" lines

@dataclass
class Diagnostics:
    initial_domain_sizes: dict[str, int]
    pruned_by_constraint: dict[str, int]     # constraint id -> values removed
    domain_sizes_after_node_consistency: dict[str, int]
    domain_sizes_after_ac3: dict[str, int]
    ac3_revisions: int
    assignments: int
    backtracks: int
    forward_check_prunes: int
    solutions_found: int
    variable_order: list[str]                # order MRV picked
    elapsed_ms: float
    relaxation_hints: list[str]              # filled when there are 0 solutions

@dataclass
class RecommendationResult:
    candidates: list[Candidate]              # ranked, best first
    diagnostics: Diagnostics
```

### 4.3 Preferences — `agent/preferences.py`

```python
@dataclass(frozen=True)
class Preferences:
    styles: frozenset[Style]            # empty = any style
    budget_eur: float                   # > 0
    venue_kinds: frozenset[VenueKind]   # empty = any kind
    dish_id: str | None                 # None = let the agent choose the dish
    want_body: int                      # 1–5
    want_acidity: int                   # 1–3
    user_lat: float
    user_lon: float
    max_distance_m: float               # 200–4000
    at_time: datetime                   # timezone-aware, Europe/Madrid
    open_only: bool                     # apply the open_now constraint
    exclude_unknown_hours: bool
    weights: dict[str, float]           # keys = the five feature names
    top_k: int = 10
```

`validate(prefs) -> list[str]` returns a list of problems (empty when valid).

### 4.4 Function signatures other modules call

```python
# agent/data_loader.py
def load_wines(path=DATA/"wines.csv") -> dict[str, Wine]
def load_venues(path=DATA/"venues_madrid.json") -> dict[str, Venue]
def load_inventory(path=DATA/"inventory.csv") -> Inventory
def load_pairing_rules(path=DATA/"pairing_rules.json") -> PairingRules

class Inventory:
    def wines_at(self, venue_id) -> frozenset[str]
    def venues_with(self, wine_id) -> frozenset[str]
    def price(self, venue_id, wine_id) -> float      # KeyError if not stocked

# agent/opening.py
def open_state(opening_hours: str, at: datetime) -> OpenState

# agent/pairing.py
def pairing_score(wine: Wine, dish_id: str, rules: PairingRules) -> float   # 0–1
def pairing_reason(wine: Wine, dish_id: str, rules: PairingRules) -> str
def dishes_for_venue(venue: Venue, rules: PairingRules) -> frozenset[str]

# agent/scoring.py
def features(wine, venue, dish_id, prefs, ctx) -> dict[str, float]
def evaluate(features: dict[str, float], weights: dict[str, float]) -> float

# agent/recommend.py
def recommend(prefs: Preferences, data: AgentData,
              solver_options: SolverOptions | None = None) -> RecommendationResult

# agent/search_route.py
def load_graph(path=DATA/"madrid_walk.graphml") -> nx.MultiDiGraph
def route(graph, start: tuple[float, float], goal: tuple[float, float]) -> RouteResult
# RouteResult: path_latlon: list[tuple[float, float]], length_m: float,
#              nodes_expanded: int, walk_minutes: float
```

`AgentData` is a dataclass bundling wines, venues, inventory and pairing rules.
`SolverOptions` holds booleans `use_mrv`, `use_lcv`, `use_forward_checking`, `use_ac3`
(all default `True`) so the UI can switch techniques off and show the difference.

---

## 5. Order of work

```
Phase 0  scaffolding
   │
Phase 2  models + preferences + loaders ──────────────┐
   │                                                  │
   ├── Phase 1  data pipeline (1.1→1.2, 1.3, 1.4)     │   (1.1–1.3 can start with phase 0)
   ├── Phase 3  pairing                               │
   ├── Phase 4  generic CSP solver  (needs nothing but phase 0)
   └── Phase 6  routing             (needs 1.3)
            │
Phase 5  wine CSP + scoring + recommend   (needs 1, 2, 3, 4)
            │
Phase 7  Streamlit app                    (needs 5, 6)
            │
Phase 8  optional minimax                 (needs 5)
Phase 9  README, demo script, final QA    (needs everything)
```

Work that can run in parallel with separate agents once phase 0 and phase 2 are in:
phase 1, phase 3, phase 4, and phase 6 (after 1.3). They touch disjoint files.

---

## 6. Phase 0 — Scaffolding

### Task 0.1 — Repository skeleton
1. Create every directory and empty `__init__.py` from section 4.1.
2. Write `requirements.txt` and `requirements-dev.txt` from section 3.8.
3. Write `pyproject.toml` with:
   - `[tool.ruff]` `line-length = 100`, `target-version = "py311"`,
     lint rules `E, F, I, B, UP`.
   - `[tool.pytest.ini_options]` `testpaths = ["tests"]`, `pythonpath = ["."]`.
4. Write `.gitignore`: `__pycache__/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`,
   `cache/` (OSMnx's HTTP cache), `.streamlit/secrets.toml`.
5. Write `agent/config.py`:
   ```python
   from pathlib import Path
   ROOT = Path(__file__).resolve().parent.parent
   DATA = ROOT / "data"
   CENTER = (40.4168, -3.7038)        # Puerta del Sol
   RADIUS_M = 4000
   TIMEZONE = "Europe/Madrid"
   MARKUP = {"shop": 1.0, "bar": 2.0, "restaurant": 2.5}
   MIN_PAIRING_SCORE = 0.5
   MAX_PER_VENUE = 2
   WALK_SPEED_M_PER_MIN = 80
   FEATURES = ("taste", "quality", "price_fit", "pairing", "distance")
   ```
6. Add a placeholder `app.py` that shows a title, so `streamlit run app.py` starts.

**Done when:** `pip install -r requirements.txt -r requirements-dev.txt` succeeds in a
fresh virtual environment, `pytest` runs (0 tests is fine), `ruff check .` is clean, and
`streamlit run app.py` serves a page.

---

## 7. Phase 1 — Data pipeline

All scripts in this phase are run by hand, once. Their outputs are committed. Each
script prints a short summary (row counts, what was dropped) when it finishes.

### Task 1.1 — Fetch and clean the wine dataset
Files: `data/fetch_wines.py`, `data/clean_wines.py`, `data/wines.csv`.

1. `fetch_wines.py`: call
   `kagglehub.dataset_download("fedesoriano/spanish-wine-quality-dataset")` and copy
   `wines_SPA.csv` to `data/raw/wines_SPA.csv`. If the download asks for credentials,
   stop and print instructions: download the CSV from the Kaggle page by hand and place
   it at that path. Do not put credentials in the repo.
2. Before committing the raw file, read the licence on the Kaggle dataset page. If it
   does not allow redistribution, add `data/raw/` to `.gitignore` and commit only the
   cleaned `wines.csv` if that is allowed; if neither is, document the manual download
   step in the README and stop for owner input.
3. Open the raw CSV and print, for every column: dtype, null count, and the unique
   values for `type`, `body`, `acidity`, `country`. Paste this summary into the top
   docstring of `clean_wines.py`. The cleaning rules below assume the columns listed in
   3.4; if the file differs, adapt and note the change in this plan.
4. `clean_wines.py` does, in order:
   1. Strip whitespace from all text columns.
   2. `year`: convert to integer; `"N.V."` or blank becomes empty (None).
   3. Drop rows with a missing or non-positive `price`.
   4. Drop exact duplicates on `(winery, wine, year)`, keeping the row with the most
      `num_reviews`.
   5. `type`: fill missing with `""`.
   6. `style`: derive from `type` with an explicit dictionary in the script. Build the
      dictionary from the actual unique values found in step 3. Guide: names containing
      "Red", and red grapes (Tempranillo, Grenache, Mencia, Monastrell, Syrah, Cabernet
      Sauvignon …) → `red`; names containing "White", and white grapes (Albarino,
      Verdejo, Chardonnay, Sauvignon Blanc …) → `white`; Cava, Sparkling → `sparkling`;
      Sherry, Pedro Ximenez → `fortified`; rosé names → `rose`; `""` → `unknown`.
      The script must raise an error on any `type` value that is not in the dictionary,
      so nothing is silently mis-styled.
   7. `body`, `acidity`: where missing, fill with the median for that `type`, or the
      global median if the type has none; round to an integer; set `imputed = True`.
   8. Sort by `(winery, wine, year)` and assign `wine_id` = `w0001`, `w0002`, ….
   9. Write `data/wines.csv` with columns exactly matching the `Wine` dataclass.
5. Tests (`tests/test_clean_wines.py`), run against the committed `wines.csv`:
   no null in any column except `year`; `wine_id` unique; every `style` is a valid
   `Style`; `body` in 1–5; `acidity` in 1–3; `retail_price > 0`.

**Done when:** `wines.csv` is committed, tests pass, and the script's printed summary
(rows in, rows out, rows per style, rows imputed) is pasted into the commit message.

### Task 1.2 — Fetch Madrid venues from OpenStreetMap
Files: `data/fetch_osm.py`, `data/venues_madrid.json`.

1. Send this query to `https://overpass-api.de/api/interpreter` (POST, field `data`).
   Set a descriptive generic `User-Agent`. Timeout 180 s. On HTTP 429 or 504, wait and
   retry up to 4 times (10 s, 20 s, 40 s, 80 s), then try the mirror
   `https://overpass.kumi.systems/api/interpreter`.
   ```
   [out:json][timeout:120];
   (
     nwr["shop"~"^(wine|alcohol)$"](around:4000,40.4168,-3.7038);
     nwr["amenity"~"^(restaurant|bar|pub)$"](around:4000,40.4168,-3.7038);
   );
   out center tags;
   ```
2. For each element build a `Venue`:
   - `venue_id` = `"{type}/{id}"`.
   - Coordinates: `lat`/`lon` for nodes, `center.lat`/`center.lon` for ways and
     relations.
   - `kind`: `shop=*` → `shop`; `amenity=restaurant` → `restaurant`;
     `amenity=bar|pub` → `bar`. If both a shop and an amenity tag exist, `shop` wins.
   - `address`: join `addr:street`, `addr:housenumber`, `addr:postcode` when present.
   - `cuisines`: split the `cuisine` tag on `;`, lower-case, strip.
   - `website`: `website` or `contact:website`.
3. Drop elements with no `name`. Drop elements whose coordinates are missing.
4. Remove duplicates: same lower-cased name within 30 m — keep the one with more tags.
5. Sort by `venue_id` and write `data/venues_madrid.json` as
   `{"fetched_at": "...", "center": [...], "radius_m": 4000, "source":
   "OpenStreetMap contributors (ODbL)", "venues": [ ... ]}`, indented by 1 space.
   `cuisine_assumed` is `true` for restaurants and bars whose `cuisines` is empty.
6. Print counts per kind, how many have `opening_hours`, how many have `cuisine`.
7. If there are more than 2 500 restaurants, keep all shops and bars, and keep only
   restaurants that have an `opening_hours` tag or a `cuisine` tag. Record the rule and
   the resulting counts in the JSON under `"filter_note"`.
8. Tests (`tests/test_venues.py`) against the committed file: every venue has a name,
   a valid kind, coordinates within 4 100 m of the centre; ids unique; at least 20
   shops (if not, widen the shop tag list and note it here).

**Done when:** the JSON is committed and tests pass.

### Task 1.3 — Fetch the walking street graph
Files: `data/fetch_graph.py`, `data/madrid_walk.graphml`.

1. ```python
   import osmnx as ox
   G = ox.graph_from_point(CENTER, dist=RADIUS_M + 300, network_type="walk", simplify=True)
   G = ox.truncate.largest_component(G, strongly=True)
   ox.save_graphml(G, DATA / "madrid_walk.graphml")
   ```
   The extra 300 m keeps venues at the edge of the circle reachable. Check the OSMnx
   2.x documentation for exact function locations if a call fails; the API moved
   between 1.x and 2.x.
2. Print node count, edge count and file size.
3. If the file is larger than 50 MB, reduce `dist` to `RADIUS_M` and, if still too
   large, store it gzip-compressed (`madrid_walk.graphml.gz`) and make `load_graph`
   handle both. Never commit a file over 95 MB (GitHub rejects 100 MB).
4. Test (`tests/test_graph_file.py`): the graph loads, has more than 5 000 nodes, every
   edge has a numeric `length`, every node has `x` and `y`.

**Done when:** the graph file is committed and the test passes.

### Task 1.4 — Build the derived inventory
Files: `data/build_inventory.py`, `data/inventory.csv`. Needs 1.1 and 1.2.

1. For each venue, seed a `random.Random` with
   `int(hashlib.sha256(venue_id.encode()).hexdigest()[:16], 16)`.
2. Number of wines carried, drawn uniformly from: shop (`shop=wine`) 60–120; shop
   (`shop=alcohol`) 30–60; restaurant 15–40; bar 10–25. Keep the original `shop` tag
   value in the venue JSON under `"osm_shop"` so this script can tell the two apart.
3. Sample that many wines **without replacement**, weighted by
   `log10(num_reviews + 10)`, so well-known wines appear in more venues. This is what
   gives the LCV heuristic something real to work with.
4. Restaurants with a non-Spanish cuisine tag still get Spanish wines (the dataset has
   no others); no special handling.
5. `price_at_venue = round(retail_price × MARKUP[kind], 2)`.
6. Write `data/inventory.csv` with columns `venue_id, wine_id, price_eur`, sorted.
7. Tests (`tests/test_inventory.py`): running the script twice produces identical
   bytes; every `venue_id` and `wine_id` exists in the source files; every venue has at
   least 10 wines; no duplicate `(venue_id, wine_id)`.

**Done when:** `inventory.csv` is committed and tests pass.

---

## 8. Phase 2 — Models, preferences, loaders

### Task 2.1 — `agent/models.py`
Copy section 4.2 exactly. Add `Candidate.as_row()` returning a flat dict for the
results table. Test: dataclasses construct, enums round-trip from their string values.

### Task 2.2 — `agent/preferences.py`
1. Implement `Preferences` from 4.3.
2. `default_preferences(now: datetime) -> Preferences`: any style, budget 25, any venue
   kind, no dish, body 3, acidity 2, user at `CENTER`, max distance 2 000 m,
   `open_only=True`, `exclude_unknown_hours=False`, all weights 5, `top_k=10`.
3. `validate`: budget > 0; body 1–5; acidity 1–3; distance 200–4 000; `at_time` is
   timezone-aware; weights has exactly the five feature keys, each 0–10; `top_k` 1–50;
   user position within 4 500 m of `CENTER`.
4. Tests: defaults validate clean; each rule has one failing case.

### Task 2.3 — `agent/data_loader.py`
1. Implement the four loaders and the `Inventory` class from 4.4. `Inventory` builds
   two dicts on load (`venue → {wine: price}` and `wine → set(venues)`) so every lookup
   is O(1). The solver depends on this for speed.
2. `load_all() -> AgentData`.
3. The loaders must work on the small fixtures in `tests/conftest.py` (pass a path).
4. Write `tests/conftest.py` with a hand-made miniature world used by all later tests:
   5 wines (2 red, 1 white, 1 sparkling, 1 fortified, prices 8–60), 4 venues (2 shops,
   1 restaurant with `cuisine=spanish`, 1 bar; one venue with hours that are closed on
   Mondays, one with no hours), 3 dishes, and an inventory linking them so that at
   least one, but not every, combination is valid.
5. Tests: loaders return the right counts on fixtures and on the real files;
   `Inventory.price` raises `KeyError` for an unstocked pair.

### Task 2.4 — `agent/opening.py`
1. `open_state(tag, at)`: empty tag → `UNKNOWN`. Otherwise
   `OpeningHours(tag).is_open(at_naive_local)` from `opening-hours-py`; any exception →
   `UNKNOWN`. Convert `at` to `Europe/Madrid` and drop the tzinfo before passing it, if
   the library wants naive datetimes (check its docs).
2. Cache results with `functools.lru_cache` keyed on `(tag, at rounded to the minute)`.
3. Tests: `"Mo-Fr 10:00-20:00"` open Wednesday 12:00, closed Sunday 12:00; `"24/7"`
   open; `""` unknown; garbage string unknown; a range crossing midnight
   (`"Fr 20:00-02:00"`) open Saturday 01:00.

---

## 9. Phase 3 — Pairing engine

### Task 3.1 — `data/pairing_rules.json`
Structure:

```json
{
  "dishes": [
    {
      "dish_id": "grilled_red_meat",
      "label": "Grilled red meat",
      "styles": {"red": 1.0, "rose": 0.4},
      "body": [4, 5],
      "acidity": [2, 3],
      "type_bonus": ["Ribera Del Duero Red", "Toro Red", "Rioja Red"],
      "why": "Tannin and body stand up to charred, fatty meat."
    }
  ],
  "cuisine_to_dishes": {
    "spanish": ["jamon_iberico", "tapas", "grilled_red_meat", "..."],
    "seafood": ["shellfish", "white_fish", "paella"],
    "_default": ["tapas", "cured_cheese", "..."]
  }
}
```

1. Write about 18 dishes covering Spanish eating: jamón ibérico, mixed tapas, grilled
   red meat, roast lamb, stew (cocido), paella, shellfish, white fish, oily fish,
   poultry, pork, mushrooms/vegetables, salad, cured cheese (manchego), blue cheese,
   spicy food, chocolate dessert, fruit/cream dessert.
2. Base every rule on standard sommelier guidance (for example: albariño with
   shellfish, fino sherry with jamón, Pedro Ximénez with chocolate or blue cheese, cava
   with fried tapas, Ribera/Toro with lamb). Each dish's `why` states the principle in
   one sentence. `type_bonus` values must be exact `wine_type` strings that occur in
   `wines.csv`.
3. `cuisine_to_dishes`: list the 25 most frequent `cuisine` values in
   `venues_madrid.json` (count them with a short script) and map each to dishes.
   Anything unlisted uses `_default`.
4. Tests: JSON loads; every dish id is unique; every `type_bonus` string exists in
   `wines.csv`; every dish referenced in the cuisine map exists; for every dish at
   least 20 wines in `wines.csv` reach a pairing score ≥ 0.5 (otherwise the `pairs`
   constraint would wipe the domain).

### Task 3.2 — `agent/pairing.py`
1. `pairing_score(wine, dish_id, rules)`:
   ```
   style_part   = rules.styles.get(wine.style, 0.0)          # 0–1
   if style_part == 0: return 0.0
   body_part    = 1.0 if lo ≤ body ≤ hi else max(0, 1 − 0.5·gap)     # gap = steps outside the range
   acidity_part = same idea with the acidity range
   bonus        = 0.15 if wine.wine_type in type_bonus else 0.0
   score        = min(1.0, style_part · (0.5 + 0.3·body_part + 0.2·acidity_part) + bonus)
   ```
2. `pairing_reason` returns a sentence such as
   `"Full-bodied Ribera del Duero red — tannin and body stand up to charred, fatty meat."`
3. `dishes_for_venue`: shops → all dishes; others → union of the mapped dishes of each
   cuisine tag, or the `spanish` set when the venue has no cuisine tag.
4. Cache `pairing_score` per `(wine_id, dish_id)`.
5. Tests on fixtures: a full red scores higher than a light white for red meat; a wine
   whose style is absent from the rule scores exactly 0; score always within `[0, 1]`;
   a shop serves every dish.

---

## 10. Phase 4 — Generic CSP solver

File: `agent/csp_solver.py`. This file knows nothing about wine. Model it on
aima-python `csp.py` and cite the AIMA chapter and function it mirrors in each
docstring.

### Task 4.1 — CSP representation
```python
class Constraint:
    id: str
    scope: tuple[str, ...]            # 1 or 2 variable names
    def satisfied(self, assignment: dict[str, Any]) -> bool   # all scope vars assigned
    def has_support(self, var: str, value: Any, other_domain: set) -> bool
        # binary only: is there a value in other_domain compatible with var=value?
        # Default implementation loops over other_domain. Subclasses override it with
        # an index lookup.

class CSP:
    def __init__(self, domains: dict[str, set], constraints: list[Constraint]): ...
    variables, domains (current), neighbors, constraints_on(var), constraints_between(a, b)
```
The `has_support` hook matters: the wine and venue domains hold thousands of values, so
a naive pairwise AC-3 would do tens of millions of Python calls. Index-backed overrides
(task 5.1) keep AC-3 near-instant.

### Task 4.2 — Node consistency and AC-3
1. `node_consistency(csp, stats)`: apply every unary constraint; record, per constraint
   id, how many values it removed (`stats.pruned_by_constraint`). Apply unary
   constraints in the order given, and count a value against the first constraint that
   removes it.
2. `ac3(csp, stats, queue=None) -> bool`: standard AC-3 with a queue of arcs
   `(Xi, Xj)`. `revise` removes a value of `Xi` when **any** constraint between `Xi`
   and `Xj` has no support for it; count removals per constraint id and count calls to
   `revise` in `stats.ac3_revisions`. Return `False` on an empty domain.
3. Tests: the textbook cases — a 3-variable map-colouring CSP; a CSP where AC-3 detects
   inconsistency; a CSP where AC-3 alone solves it; the prune counts match hand-worked
   numbers.

### Task 4.3 — Backtracking search with MRV, LCV, forward checking
1. `select_unassigned_variable(csp, assignment, use_mrv)`: MRV = smallest current
   domain; tie-break by most constraints on unassigned variables (degree heuristic),
   then by name. Without MRV: declaration order.
2. `order_domain_values(csp, var, assignment, use_lcv)`: LCV = sort by the number of
   values the choice would remove from neighbours' domains, fewest first. Constraints
   may provide `count_conflicts(var, value, other_domain)`; fall back to counting with
   `satisfied`. Because exact LCV on large domains is costly, accept an optional
   `value_key` callable on the CSP that gives a cheap, equivalent ordering (the wine
   CSP uses "number of in-domain venues carrying this wine", descending).
3. `forward_check(csp, var, value, assignment, removals, stats)`: prune inconsistent
   values from unassigned neighbours; return `False` on a wipe-out. Keep a `removals`
   list so the caller can restore domains on backtrack.
4. `backtracking_search(csp, options, stats, on_solution, should_stop=None)`:
   recursive; calls `on_solution(assignment_copy)` for **every** complete consistent
   assignment (we want all solutions, not the first), then continues. Counts
   `assignments` and `backtracks` (a backtrack = a value tried that led to no solution
   in its subtree or failed immediately). Records the order in which variables were
   first chosen.
5. `solve_all(csp, options) -> (list[dict], SearchStats)` ties it together:
   node consistency → AC-3 (if enabled) → backtracking.
6. Tests:
   - 4-queens returns exactly 2 solutions, 6-queens exactly 4, under every combination
     of the four option flags (16 combinations).
   - Australia map colouring with 3 colours returns 18 solutions.
   - With forward checking on, `assignments` is ≤ the count with it off, on 6-queens.
   - Domains are fully restored after the search finishes.
   - An unsatisfiable CSP returns `[]` without error.

---

## 11. Phase 5 — The wine CSP, scoring and the recommender

### Task 5.1 — `agent/wine_csp.py`
`build_wine_csp(prefs, data) -> CSP` creates the three variables and the eight
constraints from 3.7.

1. Unary constraints as small classes with a `satisfied` method: `StyleConstraint`,
   `VenueTypeConstraint`, `OpenNowConstraint` (skipped when `open_only` is false;
   treats `UNKNOWN` as satisfied unless `exclude_unknown_hours`), `MaxDistanceConstraint`
   (haversine from the user; cache each venue's distance in a dict that scoring reuses).
2. Binary constraints with index-backed `has_support`:
   - `StocksAndBudget` (`wine, venue`) — implement `stocks` and `budget` as two
     constraint objects so the diagnostics can count them apart, but have each use the
     `Inventory` index: for `wine=w`, support exists if any venue in
     `inventory.venues_with(w) ∩ venue_domain` satisfies the test; for `venue=v`,
     iterate `inventory.wines_at(v) ∩ wine_domain`.
   - `Serves` (`venue, dish`) — precompute `dishes_for_venue` once per venue.
   - `Pairs` (`wine, dish`) — precompute, per dish, the set of wines with
     score ≥ `MIN_PAIRING_SCORE`.
3. If `prefs.dish_id` is set, the `dish` domain is that single value.
4. Supply the `value_key` for LCV described in task 4.3.
5. Tests on fixtures: with budget below the cheapest wine, AC-3 empties a domain; a
   restaurant none of whose wines fit budget + style is removed by AC-3 (this is the
   brief's headline example — assert it by venue id); every solution returned by
   `solve_all` passes an independent brute-force check of all eight constraints, and
   the brute-force solution set equals the solver's set.

### Task 5.2 — `agent/scoring.py`
1. `features(...)` implements the table in 3.7. `min_rating`/`max_rating` are computed
   once from the wine data and passed in `ctx`.
2. `evaluate(features, weights)` implements the weighted mean, with the all-zero rule.
3. `reasons(wine, venue, dish, features, prefs) -> tuple[str, ...]`: 3–5 short lines,
   e.g. `"€18.50 — 26% under your €25 budget"`, `"650 m away (about 8 min walk)"`,
   the pairing reason, `"Rated 4.5 by 1,200 reviewers"`, and
   `"Hours unknown — check before you go"` when relevant.
4. Tests: each feature hits 0 and 1 at its extremes; raising one weight never lowers
   the rank of the candidate that is best on that feature; all-zero weights equal
   all-equal weights.

### Task 5.3 — `agent/recommend.py`
`recommend(prefs, data, solver_options)`:

1. `validate(prefs)`; raise `ValueError` listing the problems if any.
2. Build the CSP, run `solve_all`, timing it with `time.perf_counter`.
3. Turn each solution into a `Candidate` (features, score, reasons).
4. Sort by score descending, then `wine_id`, `venue_id`, `dish_id`.
5. Walk the sorted list keeping at most `MAX_PER_VENUE` per venue and at most one row
   per `(wine_id, venue_id)` (keep its best dish) until `top_k` are kept.
6. Fill `Diagnostics` from the solver stats.
7. **Zero solutions:** fill `relaxation_hints`. For each relaxable constraint in turn
   (`budget`, `max_distance`, `open_now`, `style`, `venue_type`), rebuild the CSP with
   only that constraint removed and count solutions; report the ones that help, e.g.
   `"Removing the budget limit gives 34 options (cheapest: €31.00)"`.
8. If the solution count is very large (over 200 000), do not hold them all: keep a
   bounded heap of the best `top_k × 20` in `on_solution`.
9. Tests on fixtures: the ranked list is sorted; the per-venue cap holds; the
   zero-solution case returns hints; results are identical across two runs; results
   are identical with every solver option on or off (the options change the statistics,
   never the answer).
10. Performance test on the real data: default preferences return in under 2 seconds
    on a laptop. If not, profile and fix the indexes before moving on.

---

## 12. Phase 6 — Routing

File: `agent/search_route.py`. Needs task 1.3.

### Task 6.1 — A\* on the street graph
1. `load_graph(path)`: `ox.load_graphml`, handle the `.gz` variant if task 1.3 made one.
2. `nearest_node(graph, lat, lon)`: use `ox.distance.nearest_nodes(G, X=lon, Y=lat)`.
   Note the argument order: X is longitude.
3. `astar(graph, start_node, goal_node) -> (path_nodes, length_m, nodes_expanded)`:
   our own implementation following AIMA's `astar_search` (best-first graph search on
   `f = g + h`), using `heapq` with a counter tie-breaker and a `best_g` dict.
   - Step cost: the edge `length` in metres; for parallel edges take the shortest.
   - Heuristic: haversine distance to the goal in metres. It never overestimates, so
     the result is optimal.
4. `route(graph, start, goal) -> RouteResult`: nearest nodes, A\*, convert the node
   path to `(lat, lon)` points. Where an edge has a `geometry` attribute, use its
   coordinates so the line follows the street's curve. `walk_minutes =
   length_m / WALK_SPEED_M_PER_MIN`.
5. If start and goal snap to the same node, return a two-point path with length equal
   to the haversine distance.
6. Tests (use a small hand-made graph for most; the real graph for one):
   - On a 5-node graph with a known shortest path, A\* finds it.
   - On the real graph, for 20 seeded random node pairs, our length equals
     `nx.shortest_path_length(G, a, b, weight="length")` within 1e-6.
   - `nodes_expanded` for A\* is ≤ the count for the same code run with `h = 0`.
   - Unreachable goal raises a clear `NoRouteError`.

### Task 6.2 — Multi-stop route (optional, do after phase 7)
`route_through(graph, start, stops)`: visit several chosen venues in nearest-neighbour
order, concatenating A\* legs. Skip unless time allows.

---

## 13. Phase 7 — Streamlit app

Run with `streamlit run app.py`. Layout: wide mode; sidebar = preference panel; main
area = map on top, then tabs "Results", "Why these picks", "Solver diagnostics".

### Task 7.1 — App shell and caching
1. `st.set_page_config(page_title="Madrid Wine Agent", layout="wide")`.
2. `@st.cache_resource` for `load_all()` and `load_graph()` (loaded once per server).
3. `@st.cache_data` for `recommend`, keyed on the preferences and solver options.
   `Preferences` is a frozen dataclass; make it hashable (convert the weights dict to a
   sorted tuple for the cache key).
4. Round `at_time` to the minute before it enters the key, or the cache never hits.
5. Footer: data attribution (rule 10) and the note "Venue wine lists are simulated".

### Task 7.2 — Preference panel (`ui/sidebar.py`)
`render_sidebar(data) -> tuple[Preferences, SolverOptions]`. Widgets, top to bottom:

| Widget | Type | Maps to |
|---|---|---|
| Wine style | `st.multiselect` (Red, White, Rosé, Sparkling, Fortified) | `styles` |
| Budget per bottle (€) | `st.slider` 5–150, default 25 | `budget_eur` |
| Where | `st.radio`: Anywhere / Shop / Restaurant or bar | `venue_kinds` |
| Food | `st.selectbox`: "Surprise me" + dish labels | `dish_id` |
| Body | `st.select_slider` Light … Full (1–5) | `want_body` |
| Acidity | `st.select_slider` Soft / Medium / Crisp (1–3) | `want_acidity` |
| Starting point | `st.selectbox` of landmarks (Puerta del Sol, Plaza Mayor, Atocha, Retiro, Gran Vía–Callao, Malasaña–Plaza Dos de Mayo, Chamberí–Plaza de Olavide, Lavapiés) with fixed coordinates in `config.py`, plus "Clicked on map" | `user_lat/lon` |
| Max walking distance | `st.slider` 200–4000 m step 100 | `max_distance_m` |
| Open only | `st.checkbox`, default on | `open_only` |
| Time | `st.radio` Now / Pick a time → `st.date_input` + `st.time_input` | `at_time` |
| Exclude unknown hours | `st.checkbox` | `exclude_unknown_hours` |
| "What matters to you" | five `st.slider` 0–10 | `weights` |
| Advanced: solver techniques | `st.expander` with four checkboxes | `SolverOptions` |

No "Submit" button: Streamlit re-runs on every change, which is the "Dynamic" rubric
item. When a map click sets the start point, store it in `st.session_state`.

### Task 7.3 — Map (`ui/map_view.py`)
`render_map(result, data, prefs, route) -> dict` (returns the `st_folium` output).

1. `folium.Map(location=user position, zoom_start=15, tiles="OpenStreetMap")`.
2. A distinct marker for the user's start point.
3. A translucent `folium.Circle` of radius `max_distance_m` around the user — this
   shows the distance constraint visually.
4. One marker per recommended venue, numbered by rank (`folium.DivIcon`), colour by
   kind (shop / restaurant / bar). Popup HTML: venue name, kind, address, open state,
   then for each recommended wine there: wine name, year, price, score, and the
   `reasons` lines. Escape all text with `html.escape` — venue names come from OSM.
5. A checkbox "Show venues the agent ruled out" adds small grey `CircleMarker`s for
   venues removed by a constraint, with the tooltip naming the constraint. Cap at 300
   markers.
6. The route as a `folium.PolyLine` from the user to the selected result (default:
   rank 1). Fit the map bounds to the route.
7. Render with `st_folium(m, height=520, use_container_width=True,
   returned_objects=["last_clicked", "last_object_clicked"])`. Limiting
   `returned_objects` stops the app re-running on every pan and zoom.
8. Under the map, one line: `Route to <venue>: 850 m, about 11 min on foot — A*
   expanded 412 nodes`.

### Task 7.4 — Results and diagnostics (`ui/results_view.py`)
**Tab "Results"**
- `st.dataframe` of the ranked candidates: rank, wine, winery, year, style, venue,
  kind, dish, price, distance, score (as a `ProgressColumn`). Single-row selection
  (`on_select="rerun"`, `selection_mode="single-row"`) sets which venue the route goes
  to.
- When there are no results: `st.warning` plus the `relaxation_hints` as bullets.

**Tab "Why these picks"**
- For the selected candidate: the five feature values, each multiplied by its weight,
  shown as a horizontal bar chart (`st.bar_chart`), plus the formula written out with
  the actual numbers: `Eval = (5×0.82 + 5×0.60 + …) / 25 = 0.71`.
- The `reasons` list.

**Tab "Solver diagnostics"**
- Table: for each variable, domain size at start → after node consistency → after
  AC-3.
- Bar chart: values pruned per constraint id.
- Metrics row (`st.metric`): solutions found, assignments, backtracks, forward-check
  prunes, AC-3 revisions, solve time in ms.
- The variable order MRV chose, with a one-line explanation.
- A "Compare techniques" button that runs the solver with all techniques on, and with
  each one switched off in turn, and shows the statistics side by side in one table.
  This is the clearest demonstration that the techniques do something.

### Task 7.5 — Front-end checks
1. `streamlit.testing.v1.AppTest` smoke test: the app runs with defaults without an
   exception and renders a dataframe; setting the budget to its minimum shows the
   warning instead of crashing.
2. Manual check list (run through it and record the result in the pull request):
   change each widget once and confirm the table and map update; click a table row and
   confirm the route moves; open a popup; switch every solver technique off and confirm
   the results are unchanged; disconnect the network and confirm everything except the
   map tiles still works.

---

## 14. Phase 8 — Optional: "Sommelier vs. you" (minimax + alpha-beta)

Do this only after phases 0–7 are finished and working.

File: `agent/minimax.py`, plus a fourth tab in the UI.

**The game.** A three-ply, zero-sum game played over the solutions the CSP already
found:

1. **Sommelier (MAX)** chooses a venue from the top 8 venues.
2. **You (MIN)** choose a dish that venue serves.
3. **Sommelier (MAX)** chooses a wine there that passes all hard constraints for that
   dish.

Terminal utility (what the sommelier wants high and you want low) is the upsell value:
`U = 0.5·quality + 0.5·(price_at_venue / budget)`. The minimax value of a venue is the
upsell you can hold it to in the worst case; the UI presents the venue with the
**lowest** minimax value as the "sommelier-proof" pick, alongside the sommelier's
preferred line of play.

Tasks:
1. `minimax(node)` and `alphabeta(node, alpha, beta)` written generically over a
   `Game` interface (`actions`, `result`, `is_terminal`, `utility`, `to_move`),
   following aima-python `games.py`. Both count the nodes they visit.
2. `WineGame` implements that interface from a `RecommendationResult`'s solution set.
3. Tests: on a hand-built tree with known value (use the textbook 3×3 example,
   value 3) both functions return it; alpha-beta always returns the same value as
   minimax on 50 seeded random trees; alpha-beta visits ≤ the nodes minimax visits.
4. UI tab: the chosen line of play, the game value, and
   `Minimax visited N nodes, alpha-beta visited M (X% fewer)`.

---

## 15. Phase 9 — Documentation and final QA

### Task 9.1 — README.md
Sections: what it is (one paragraph + screenshot); quick start (`python -m venv`,
`pip install -r requirements.txt`, `streamlit run app.py`); how it works (the CSP table
from 3.7, the evaluation function, A\*, each mapped to its AIMA chapter); data sources
and licences; **what is real and what is derived** (inventory, markups, pairing rules,
assumed cuisines — stated plainly); how to refresh the data (the four scripts, in
order); project layout; how to run tests.

### Task 9.2 — Demo script
`docs/DEMO.md`: a 5-minute walkthrough with exact clicks, each tied to a rubric item
and a technique:
1. Defaults → ranked results and a route (A\*).
2. Drop the budget to €10 → domains shrink; show the prune chart (forward checking /
   node consistency).
3. Choose "Shellfish" + "Restaurant" → show a restaurant removed by AC-3 because none
   of its wines fit.
4. Diagnostics → "Compare techniques" table (MRV, LCV, forward checking).
5. Move the weight sliders → ranking reorders; show the score breakdown (evaluation
   function).
6. Impossible request → relaxation hints.
7. (If built) the sommelier tab (minimax, alpha-beta).

### Task 9.3 — Final checks
1. Fresh clone, fresh virtual environment, install, `pytest`, `ruff check .`, run the
   app — all clean.
2. Run the app with the network off; only map tiles may fail.
3. Confirm no file in the repo exceeds 50 MB and no credentials are committed.
4. Confirm the attribution footer and the "simulated inventory" note are visible.
5. Read every **OWNER CHECK** item in section 3 and confirm the owner has answered it,
   or that the README states the assumption.

---

## 16. Progress checklist

Tick a box in the same commit that finishes the task.

- [ ] 0.1 Repository skeleton
- [ ] 1.1 Wine dataset fetched and cleaned
- [ ] 1.2 Venues fetched from OpenStreetMap
- [ ] 1.3 Walking graph fetched
- [ ] 1.4 Derived inventory built
- [ ] 2.1 Models
- [ ] 2.2 Preferences
- [ ] 2.3 Data loaders and test fixtures
- [ ] 2.4 Opening hours
- [ ] 3.1 Pairing rules file
- [ ] 3.2 Pairing engine
- [ ] 4.1 CSP representation
- [ ] 4.2 Node consistency and AC-3
- [ ] 4.3 Backtracking with MRV, LCV, forward checking
- [ ] 5.1 Wine CSP
- [ ] 5.2 Scoring
- [ ] 5.3 Recommender
- [ ] 6.1 A\* routing
- [ ] 6.2 Multi-stop route (optional)
- [ ] 7.1 App shell and caching
- [ ] 7.2 Preference panel
- [ ] 7.3 Map
- [ ] 7.4 Results and diagnostics
- [ ] 7.5 Front-end checks
- [ ] 8 Sommelier vs. you (optional)
- [ ] 9.1 README
- [ ] 9.2 Demo script
- [ ] 9.3 Final checks

---

## 17. Known risks and what to do about them

| Risk | What to do |
|---|---|
| Overpass is rate-limited or down | Retry with back-off, then the mirror (task 1.2). The data is cached in the repo, so this only matters on a refresh. |
| Kaggle download needs a login | Manual download to `data/raw/` (task 1.1). |
| The dataset's columns or `type` values differ from what this plan assumes | Task 1.1 step 3 inspects the file first; the cleaner fails loudly on unknown values. Update section 3.4 if needed. |
| Few venues have `opening_hours` | The `unknown` state (3.6) keeps them visible with a badge. |
| Graph file too large for git | Task 1.3 step 3. |
| AC-3 or search is slow on real data | Index-backed `has_support` (4.1, 5.1); performance test in 5.3. |
| The map makes the app re-run constantly | `returned_objects` limit (7.3 step 7). |
| OSMnx or `opening-hours-py` API differs from the calls shown | Read the installed version's documentation; the calls here target OSMnx 2.x and opening-hours-py 2.x. |
| A dish has too few matching wines, so `pairs` empties the domain | Test in task 3.1 step 4 catches it; widen that dish's rule. |
