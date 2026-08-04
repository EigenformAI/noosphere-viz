# Armillary Noosphere

The idea-sky of AI research & discourse: three spherical star-map projections, drawn
as constellations turning inside an armillary sphere. Every star is a real document.

One self-contained page, no dependencies, no build step: **`index.html`** (~3 MB).

## The three plates

| Plate | Sky | Data | Slider |
|---|---|---|---|
| I | **The Research Sky** — six months of arXiv (50,295 papers, Feb–Aug 2026), clustered by TF-IDF → SVD → UMAP → HDBSCAN, constellations named by TF-IDF top terms | trend-detection pipeline | month by month |
| II | **The Attention Sky** — one week of discourse (X, Hacker News, LessWrong, blogs; 299 docs), 15 clusters on the week's own embedding map | mindspace `week_projector` export | day by day |
| III | **The Emergent Sky** — metaconcepts whose meaning runs ahead of shared vocabulary: the final week's highest *name-gap* clusters, named by a 4-model panel synthesized by Claude Fable | mindspace pipeline run | day by day |

Interactions: drag to turn a sphere · hover a star for its document · click a
constellation (or its key row) to focus it — Plate III opens the full Fable reading.
Each plate carries an honesty caption: **real** (what you could check against the
source data) vs **simplified** (windows, sampling, and labeling choices made here).

## Repository layout

- `index.html` — the built page (what Vercel serves)
- `noosphere_template.html` — the page source: canvas engine, armillary chrome, keys,
  sliders, deterministic `#record` hooks; data is injected at the `/*__DATA_JSON__*/` token
- `noosphere_data.json` — the packed data (3 document-spaces + 3 view definitions)
- `build_noosphere.py` — injects the JSON into the template:
  `python3 build_noosphere.py --out index.html`
- `build/build_noosphere_data.py` — provenance copy of the data-prep script; it reads
  the private scraping/clustering pipelines (trend-detection + mindspace) and is **not
  runnable from this repo** — it documents exactly how `noosphere_data.json` was made

## Deploying on Vercel

Pure static site — no framework, no build.

- Framework preset: **Other**
- Build command: *(empty)* · Output directory: **`.`** (repo root)

## Editing

Change the template, then rebuild the page:

```
python3 build_noosphere.py --out index.html
```

Refreshing the *data* (new pipeline runs) happens in the source repos; regenerate
`noosphere_data.json` there with `build/build_noosphere_data.py` and copy it over.
