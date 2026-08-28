# Reports from the future 

The idea-sky of AI research & discourse: three spherical star-map projections, drawn
as constellations turning inside an armillary sphere. Every star is a real document.

One self-contained page, no dependencies, no build step: **`index.html`** (~3.8 MB).

## The three plates

| Plate | Sky | Data | Colour | Slider |
|---|---|---|---|---|
| I | **The Research Sky** — six months of arXiv (49,135 papers, Feb–Aug 2026), clustered by embedding UMAP → HDBSCAN (`min_cluster_size` 100 → 105 clusters), constellations named by TF-IDF keywords; **every cluster** is a constellation, drawn from the export's deterministic 20,000-paper sample (unclustered papers as dust). Only the labels that fit are painted; the key lists every one | mindspace `arxiv` export | that month's publishing **rate** — papers per weekday-equivalent day, one absolute scale across all months: white quiet → yellow → red busy | month by month |
| II | **The Attention Sky** — the same 11 weeks and the same sphere as Plate III, but **every** cluster the run found (209), named by its raw TF-IDF keywords: what the discourse *calls* things | mindspace `quarter_without_fable` run | cluster size **normalised inside each week** and log-scaled: that week's smallest constellation white → its largest red | week by week |
| III | **The Emergent Sky** — metaconcepts whose meaning runs ahead of shared vocabulary. Each of the 11 weeks is clustered **independently**: positions come from the shared 3-month window projection, highlight colours only from that week's own clustering run. Only the clusters a 4-model panel named via Claude Fable are drawn (142 of the same 209), each carrying its full reading — no TF-IDF anywhere. Flip between Plates II and III to see the name gap itself | mindspace `quarter_with_fable` run | continuous name gap: indigo buzzword → red established → pale-gold emerging | week by week |

Plate II's colour scale is rescaled per week on purpose: cluster sizes are
long-tailed, so a single absolute ramp painted one red constellation across the
whole quarter and left everything else white. The cost is that colour ranks a
week against itself, not against the other weeks — the legend says so.

Plate I keeps one absolute scale across all months, but measures a **rate**
rather than a monthly total. Both end months are clipped by the export window —
February starts on the 4th, August has papers on only 3 days — and short spans
carry an unrepresentative mix of weekdays (arXiv's weekend runs at ≈0.58 of a
weekday, and 2 of August's 3 days are a weekend). So each month is divided by
its *weekday-equivalent days*: every observed day counts as its weekday's share
of an average day. On totals August sat at 6% of the May peak and February at
78% of March; on rate they are 75% and 97%. The build derives the denominator
from the shipped sample — the same rows the numerator counts — and prints a
note when the export's own `days` field disagrees.

August's rate rests on 3 days of papers, so its *level* is now honest but its
per-cluster ranking is thin; the caption says so.

Clicking a Plate III constellation opens its reading in two parts: the panel's
plain description of what the documents are (grok-4.5's compression), then the
Fable synthesis naming the metaconcept.

Interactions: drag to turn a sphere · hover a star for its document · click a
constellation (or its key row) to focus it — Plate III opens the full Fable reading.
Clicking a star focuses its constellation *and* puts that document at the top of
the detail panel; every document listed there links out to its source (arXiv,
X, LessWrong, Hacker News, the blogs) in a new tab.

Nothing plays on its own: the reveal only advances via the play button, the
slider, or the arrow keys.

On Plate I the slider does not reveal the sky, it moves a light across it: all
six months are present as grey dust at every stop and only the selected month's
papers are lit, so the shape of the corpus never changes and each month is read
against the same backdrop. Focusing a constellation shows its *whole* six-month
extent, with the selected month brightest.

On Plate III the slider steps between **visual frames**, not a cumulative reveal:
each week highlights only its own clusters and un-highlights the previous week's,
its unclustered documents render neutral-pale, and the trailing window's earlier
documents fall back to grey dust. Cluster identities and names are per-week and
independent, so the same theme reappearing under a different name week to week is
expected — that is the signal, not a bug.
Each plate carries an honesty caption: **real** (what you could check against the
source data) vs **simplified** (windows, sampling, and labeling choices made here).

## Repository layout

- `index.html` — the built page (what Vercel serves)
- `noosphere_template.html` — the page source: canvas engine, armillary chrome, keys,
  sliders, deterministic `#record` hooks; data is injected at the `/*__DATA_JSON__*/` token
- `noosphere_data.json` — the packed data (2 document-spaces + 3 view definitions).
  Source URLs are stored host-prefix + path (`{pre, i, s}`) and rebuilt in the
  page; stored whole, the 25,005 links would cost ~470 KB more
- `build_noosphere.py` — injects the JSON into the template:
  `uv run build_noosphere.py --standalone --out index.html`
  (`--standalone` wraps the page in a full `<!doctype html>` document; without it
  you get the bare fragment used for claude.ai artifact publishing. The default
  title is "Reports from the future"; override with `--title`.)
- `build/build_noosphere_data.py` — the data-prep script, **runnable from this repo**
  (takes seconds; all projections are precomputed by the pipelines):
  `uv run build/build_noosphere_data.py`
  It reads the pipeline exports directly:
  - `…/data/output/quarter_without_fable` — Plate II **and the shared corpus
    space**: this is the export that ships `projector/` and `projection3d.npz`.
    Its clusters keep raw TF-IDF names.
  - `…/data/output/quarter_with_fable` — Plate III: the *same* run exported with
    the panel labels (`*_by_week.json` Fable names, per-model compressions &
    attractor readings). This directory has been renamed more than once
    (`quarter_with_fable` → `quarter` → back), so the build tries the known
    aliases, and it has not always shipped `projector/` — hence the space is
    read from whichever export has one. The build then proves the two are the
    same run — identical row order, coordinates and `frames.json` — before
    letting the plates share a space, so a wrong guess fails loudly rather than
    quietly mixing two clusterings.
  - `…/data/output/arxiv` — Plate I arXiv space
    (the small files snapshotted into `build/arxiv_source/`)
- `pyproject.toml` — uv project (numpy; Python 3.12)

## Deploying on Vercel

Pure static site — no framework, no build.

- Framework preset: **Other**
- Build command: *(empty)* · Output directory: **`.`** (repo root)

## Editing

Change the template, then rebuild the page:

```
uv run build_noosphere.py --standalone --out index.html
```

Refreshing the data (new pipeline runs):

```
uv run build/build_noosphere_data.py
```
