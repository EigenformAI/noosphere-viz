#!/usr/bin/env python3
"""Build noosphere_data.json — the data pack for the Armillary Noosphere page.

Reads the per-plate pipeline outputs (plate III corpus from backend/output_number_3
with titles/sources from the live backend/mindspace.db; plate II from
backend/output_projector_week_number_2; plate I from the trend-detection repo)
and emits one compact JSON with:

  docs    — all 7,073 corpus docs: title, source class, publish-day offset,
            and BOTH spherical coordinate sets (PCA-sphereize of the raw
            1536-d embeddings, and radialized UMAP-3D)
  views   — the three projections:
              arxiv   HF-Papers subset (13 dense weeks), HDBSCAN re-cluster
                      + TF-IDF keyword labels, weekly stops (space 0)
              social  the week-only export backend/output/projector_week —
                      its own 299-doc embedding/UMAP space (space 1), all 15
                      clusters, TF-IDF keyword labels, daily stops
              meta    the panel+Fable "name gap" clusters with attractor
                      text, daily stops (space 0)

Run from the backend dir so its uv env + modules resolve:
  cd backend && uv run python ../noosphere/build_noosphere_data.py
"""
import csv
import json
import re
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent / "backend"
OUT_DIR = BACKEND / "output_number_3"           # plate III + its corpus space
WEEK_DIR = BACKEND / "output_projector_week_number_2"  # plate II
DB = BACKEND / "mindspace.db"                   # titles/sources for OUT_DIR's corpus
DEST = Path(__file__).resolve().parent / "noosphere_data.json"

sys.path.insert(0, str(BACKEND))

X_SOURCES = {"Desearch X", "Twitter/Grok"}
ARXIV_SOURCE = "HuggingFace Papers"

# ---------------------------------------------------------------- load corpus
proj = np.load(OUT_DIR / "projection.npz")
ids = proj["article_ids"]
published = proj["published"]
n = len(ids)
coords2d = proj["coords"]

vec = np.fromfile(OUT_DIR / "projector" / "vectors.bytes", dtype="<f4")
assert vec.size == n * 1536, f"vectors.bytes size {vec.size} != {n}x1536"
vec = vec.reshape(n, 1536)

p3 = np.load(OUT_DIR / "projection3d.npz")
assert list(p3["article_ids"]) == list(ids), "projection3d row order mismatch"
coords3d = p3["coords"]

con = sqlite3.connect(DB)
rows = dict(
    (r[0], r[1:])
    for r in con.execute("SELECT id, source, title, content FROM articles")
)
missing = [i for i in ids if i not in rows]
assert not missing, f"{len(missing)} article_ids missing from db.bak"
sources = [rows[i][0] for i in ids]
titles = [rows[i][1] or "" for i in ids]
contents = [rows[i][2] or "" for i in ids]
con.close()

# ------------------------------------------------------------- sphere coords
def unit_rows(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

# (a) TF-Projector-style sphereize: center, renormalize, PCA -> 3, renormalize
xc = unit_rows(vec - vec.mean(axis=0))
cov = xc.T @ xc
evals, evecs = np.linalg.eigh(cov)
pca3 = unit_rows(xc @ evecs[:, -3:][:, ::-1])

# (b) UMAP-3D radialized: center, project outward onto the unit sphere
umap3 = unit_rows(coords3d - coords3d.mean(axis=0))

# ------------------------------------------------------------------ doc pack
BASE = date(2026, 3, 11)
day_off = [(date.fromisoformat(p) - BASE).days for p in published]

def src_class(s):
    if s == ARXIV_SOURCE:
        return 0  # arxiv
    if s in X_SOURCES:
        return 1  # x / social
    return 2      # web (forums, HN, blogs)

def flat(m):
    return [round(float(v), 4) for v in m.ravel()]

docs = {
    "title": [t[:90] for t in titles],
    "src": [src_class(s) for s in sources],
    "pub": day_off,
    "pca": flat(pca3),
    "umap": flat(umap3),
}

KW_JUNK = {"https", "http", "www", "com", "amp"}

def clean_kw(s):
    toks = [t for t in s.split(" · ") if t.lower() not in KW_JUNK]
    return " · ".join(toks) if len(toks) >= 2 else s

def stop_counts(rows_idx, stops, unit_days):
    """Docs published within each stop's window (week or day ending at stop)."""
    out = []
    for s in stops:
        end = date.fromisoformat(s)
        start = end - timedelta(days=unit_days - 1)
        out.append(sum(1 for i in rows_idx
                       if start <= BASE + timedelta(days=day_off[i]) <= end))
    return out

# --------- view 1: trend-detection 6-month arXiv / TF-IDF (own space)
# 50k papers is too many to ship/render whole: keep the top NAMED clusters
# complete, sample the rest — all counts disclosed in the plate caption.
TD = Path("/root/michael-folder/trend-detection/backend/data")
TD_ALGO = "tfidf-2026-02_2026-08--d50-nn15-mcs15-ms5-eom"
TD_WIN = "2026-02_2026-08"
TD_NAMED = 24       # constellations shown in full
TD_CTX_RATE = 1 / 3  # sample rate for other clusters' papers (pale members)
TD_DUST_RATE = 1 / 5  # sample rate for noise papers (dust)

red50 = np.load(TD / "reduced" / "tfidf-2026-02_2026-08--d50-nn15" / f"{TD_WIN}.npy")
csv.field_size_limit(10 ** 9)
with open(TD / "projector" / TD_WIN / "metadata.tsv") as f:
    rd2 = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    thdr = next(rd2)
    trows = list(rd2)
assert thdr[0] == "cluster_name" and thdr[9] == "cluster_id" \
    and thdr[11] == "month" and thdr[12] == "document", thdr
assert len(trows) == red50.shape[0], (len(trows), red50.shape)

from collections import Counter  # noqa: E402

doc_chunks = {}
for j, r in enumerate(trows):
    doc_chunks.setdefault(r[12], []).append(j)

NOISE_IDS = {"-1", "noise", ""}
td_docs = []  # (cid, month, title, mean 50-d vector)
for doc, js in doc_chunks.items():
    cid = Counter(trows[j][9] for j in js).most_common(1)[0][0]
    j0 = js[0]
    td_docs.append((cid, trows[j0][11], trows[j0][8], red50[js].mean(axis=0)))

cl_count = Counter(d[0] for d in td_docs if d[0] not in NOISE_IDS)
top_ids = [cid for cid, _ in cl_count.most_common(TD_NAMED)]
top_set = set(top_ids)
cl_names = {r[9]: r[0] for r in trows}

rng = np.random.default_rng(42)
keep = []
n_ctx_total = n_dust_total = 0
for d in td_docs:
    if d[0] in top_set:
        keep.append(d)
    elif d[0] not in NOISE_IDS:
        n_ctx_total += 1
        if rng.random() < TD_CTX_RATE:
            keep.append(d)
    else:
        n_dust_total += 1
        if rng.random() < TD_DUST_RATE:
            keep.append(d)

import umap  # noqa: E402
um3 = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.05,
                metric="euclidean", random_state=42,
                verbose=False).fit_transform(np.stack([d[3] for d in keep]))
umap3_td = unit_rows(um3 - um3.mean(axis=0))

td_months = sorted({d[1] for d in td_docs})
def month_end(m):
    y, mo = int(m[:4]), int(m[5:7])
    nxt = date(y + (mo == 12), mo % 12 + 1, 1)
    return (nxt - timedelta(days=1)).isoformat()
arx_stops = [month_end(m) for m in td_months]

docs_arx6 = {
    "title": [d[2][:90] for d in keep],
    "src": [0] * len(keep),
    "pub": [(date.fromisoformat(d[1] + "-01") - BASE).days for d in keep],
    "umap": flat(umap3_td),
}

arx_clusters = []
for cid in top_ids:
    rows_k = [k for k, d in enumerate(keep) if d[0] == cid]
    arx_clusters.append({
        "name": clean_kw(cl_names[cid]),
        "size": len(rows_k),
        "rows": rows_k,
        "wk": [sum(1 for k in rows_k if keep[k][1] == m) for m in td_months],
    })
arx_members = sorted(k for k, d in enumerate(keep) if d[0] not in NOISE_IDS)
n_arx_total = len(cl_count)

# ---------------- view 2: the week-only projector export (own space)
# output/ is volatile (pipeline reruns clear it) — snapshot the export here,
# and fall back to the snapshot when the live dir is gone.
SNAP_DIR = Path(__file__).resolve().parent / "week_source"
if (WEEK_DIR / "week_meta.json").exists():
    SNAP_DIR.mkdir(exist_ok=True)
    for f in WEEK_DIR.iterdir():
        (SNAP_DIR / f.name).write_bytes(f.read_bytes())
else:
    print(f"note: {WEEK_DIR} missing, using snapshot {SNAP_DIR}")
    WEEK_DIR = SNAP_DIR

with open(WEEK_DIR / "metadata.tsv") as f:
    rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    whdr = next(rd)
    wrows = list(rd)
assert whdr == ["label", "cluster", "source", "title", "url", "published"], whdr
nw = len(wrows)
wmeta = json.loads((WEEK_DIR / "week_meta.json").read_text())
assert wmeta["documents"] == nw

wvec = np.fromfile(WEEK_DIR / "vectors.bytes", dtype="<f4")
assert wvec.size == nw * 1536, f"week vectors size {wvec.size} != {nw}x1536"
wvec = wvec.reshape(nw, 1536)
wum = np.fromfile(WEEK_DIR / "umap3d.bytes", dtype="<f4").reshape(nw, 3)

wxc = unit_rows(wvec - wvec.mean(axis=0))
wcov = wxc.T @ wxc
wev, wevec = np.linalg.eigh(wcov)
wpca3 = unit_rows(wxc @ wevec[:, -3:][:, ::-1])
wumap3 = unit_rows(wum - wum.mean(axis=0))

w_pub = [(date.fromisoformat(r[5]) - BASE).days for r in wrows]
docs_week = {
    "title": [r[3][:90] for r in wrows],
    "src": [src_class(r[2]) for r in wrows],
    "pub": w_pub,
    "pca": flat(wpca3),
    "umap": flat(wumap3),
}

wk_start = date.fromisoformat(wmeta["window"].split("..")[0])
n_wdays = max(w_pub) - (wk_start - BASE).days + 1
week_day_stops = [(wk_start + timedelta(days=k)).isoformat() for k in range(n_wdays)]

soc_clusters = []
for c in wmeta["clusters"]:
    cid = str(c["cluster_id"])
    rws = [i for i, r in enumerate(wrows) if r[1] == cid]
    assert len(rws) == c["size"], (cid, len(rws), c["size"])
    share = sum(1 for i in rws if wrows[i][2] in X_SOURCES) / len(rws)
    soc_clusters.append({
        "name": clean_kw(c["keywords"]),
        "size": c["size"],
        "share": round(share, 2),
        "rows": rws,
        "wk": [sum(1 for i in rws
                   if BASE + timedelta(days=w_pub[i]) == date.fromisoformat(s))
               for s in week_day_stops],
    })
soc_clusters.sort(key=lambda c: -c["size"])
w_srcmix = {}
for r in wrows:
    w_srcmix[r[2]] = w_srcmix.get(r[2], 0) + 1

# ------------------------------------------- view 3: final frame, daily
frames = json.loads((OUT_DIR / "frames.json").read_text())
frame = frames[-1]
week_idx = frame["week_idx"]
fw_end = date.fromisoformat(frame["week_end"])
day_stops = [(fw_end - timedelta(days=6 - k)).isoformat() for k in range(7)]

def cluster_rows(c):
    return [week_idx[p] for p in c["points"]]

ai_labels = json.loads((OUT_DIR / "name_gap_ai_labels.json").read_text())
compressions = {str(c["cluster_id"]): c
                for c in json.loads((OUT_DIR / "name_gap_compressions.json").read_text())}

def excerpt(text, max_len=220):
    plain = re.sub(r"[#*_`]+", "", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", plain)
    out = ""
    for p in parts:
        if out and len(out) + len(p) > max_len:
            break
        out = (out + " " + p).strip()
    return out[:max_len + 60]

BAND_ORDER = {"emerging": 0, "established": 1, "buzzword": 2}
meta_clusters = []
for cid, name in ai_labels.items():
    c = next(cl for cl in frame["clusters"] if str(cl["cluster_id"]) == cid)
    comp = compressions[cid]
    rws = cluster_rows(c)
    meta_clusters.append({
        "name": name,
        "band": c["band"],
        "gap": round(c["name_gap"], 3),
        "kw": clean_kw(c["keywords"]),
        "size": c["size"],
        "rows": rws,
        "wk": stop_counts(rws, day_stops, 1),
        "excerpt": excerpt(comp["attractor"]),
        "attractor": comp["attractor"],
        "panel": list(comp["compressions"].keys()),
    })
meta_clusters.sort(key=lambda c: (BAND_ORDER[c["band"]], -c["gap"]))

# --------------------------------------------------------------------- emit
data = {
    "meta": {
        "built_from": "output.bak.2 + output/projector_week",
        "week_end": frame["week_end"],
        "base_date": BASE.isoformat(),
        "n_docs": n,
        "sources": ["arxiv", "x", "web"],
    },
    "spaces": [docs, docs_week, docs_arx6],
    "views": [
        {
            "id": "arxiv", "space": 2,
            "stops": arx_stops, "stopUnit": "month",
            "partialLast": td_months[-1] == date.today().isoformat()[:7],
            "members": arx_members,
            "clusters": arx_clusters,
            "docs_total": len(td_docs),
            "named_docs": sum(c["size"] for c in arx_clusters),
            "ctx_total": n_ctx_total,
            "dust_total": n_dust_total,
            "dust_kept": len(keep) - len(arx_members),
            "total_clusters": n_arx_total,
        },
        {
            "id": "social", "space": 1,
            "stops": week_day_stops, "stopUnit": "day",
            "members": sorted({i for c in soc_clusters for i in c["rows"]}),
            "clusters": soc_clusters,
            "n_docs": nw,
            "noise": wmeta["noise"],
            "window": wmeta["window"],
            "srcmix": w_srcmix,
        },
        {
            "id": "meta", "space": 0,
            "stops": day_stops, "stopUnit": "day",
            "members": sorted({i for c in meta_clusters for i in c["rows"]}),
            "clusters": meta_clusters,
        },
    ],
}
DEST.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
kb = DEST.stat().st_size / 1024
print(f"wrote {DEST} ({kb:.0f} KB)")
print(f"arxiv (trend-detection space): {len(td_docs)} papers {td_months[0]}..{td_months[-1]}, "
      f"kept {len(keep)} ({len(arx_members)} members + {len(keep) - len(arx_members)} dust); "
      f"top {len(arx_clusters)} of {n_arx_total} clusters named "
      f"(sizes {[c['size'] for c in arx_clusters][:8]}...)")
print(f"social (week space): {nw} docs {wmeta['window']}, {len(soc_clusters)} clusters "
      f"+ {wmeta['noise']} noise, stops {week_day_stops[0]}..{week_day_stops[-1]}, "
      f"mix {sorted(w_srcmix.items(), key=lambda t: -t[1])}")
print(f"meta: {len(meta_clusters)} clusters, bands "
      f"{[(c['band'], c['name']) for c in meta_clusters][:5]} ...")
