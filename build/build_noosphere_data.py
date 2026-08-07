#!/usr/bin/env python3
"""Build noosphere_data.json — the data pack for the "Reports from the future" page.

Reads the published pipeline exports directly (no private DBs needed):

  plate III + corpus space  /root/michael-folder/mindspaceai/data/output/quarter
  plate II week space       /root/michael-folder/mindspaceai/data/output/week
  plate I arXiv space       /root/michael-folder/mindspaceai/data/output/arxiv

and emits one compact JSON with:

  spaces  — three document sets, each with title, source class, publish-day
            offset, and unit-sphere coordinates (PCA-sphereize of the raw
            1536-d embeddings where available, and radialized UMAP-3D)
  views   — the three projections:
              arxiv   the 6-month arXiv export (own space): the pipeline ships
                      a deterministic 20k-paper sample; top clusters named,
                      other clusters as pale stars, sampled noise as dust,
                      monthly stops
              social  the week-only projector export — its own embedding/UMAP
                      space, all clusters, TF-IDF keyword labels, daily stops
              meta    every scored name-gap cluster of the final week; the
                      top-gap ones carry panel+Fable names and attractor
                      readings, daily stops (corpus space)

Run from the repo root:
  uv run build/build_noosphere_data.py
"""
import collections
import csv
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "noosphere_data.json"

MS_OUT = Path("/root/michael-folder/mindspaceai/data/output")
# the same quarter run exported twice: one carrying the panel+Fable names,
# one with none. Plate II reads the TF-IDF variant, plate III the Fable one —
# same weeks, same clusters, same positions, two naming regimes.
def pick(*names):
    """First of these export directories that exists.

    The pipeline has renamed the Fable export several times (quarter_with_fable
    -> quarter -> quarter_with_fable), so try the known aliases rather than
    breaking. Whichever one is picked is then *proved* to be the same run as the
    other export below, so a wrong guess fails loudly instead of quietly mixing
    two clusterings.
    """
    for nm in names:
        if (MS_OUT / nm).is_dir():
            return MS_OUT / nm
    raise SystemExit(f"none of {names} found under {MS_OUT}")


QUARTER_TFIDF = pick("quarter_without_fable")
QUARTER_FABLE = pick("quarter_with_fable", "quarter")
# the shared corpus space comes from whichever export actually ships projector/
# and projection3d.npz — the Fable one has not always written them
QUARTER = QUARTER_TFIDF if (QUARTER_TFIDF / "projector").is_dir() else QUARTER_FABLE
ARXIV = MS_OUT / "arxiv"

X_SOURCES = {"Desearch X", "Twitter/Grok"}
ARXIV_SOURCE = "HuggingFace Papers"
BASE = date(2026, 3, 11)   # day offsets are relative to this; negatives are fine

csv.field_size_limit(10 ** 9)


def unit_rows(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def pca_sphereize(vec):
    """TF-Projector-style: center, renormalize, PCA -> 3, renormalize."""
    xc = unit_rows(vec - vec.mean(axis=0))
    cov = xc.T @ xc
    evals, evecs = np.linalg.eigh(cov)
    return unit_rows(xc @ evecs[:, -3:][:, ::-1])


def flat(m):
    return [round(float(v), 4) for v in m.ravel()]


def src_class(s):
    if s == ARXIV_SOURCE:
        return 0  # arxiv
    if s in X_SOURCES:
        return 1  # x / social
    return 2      # web (forums, HN, blogs)


KW_JUNK = {"https", "http", "www", "com", "amp"}


def clean_kw(s):
    toks = [t for t in s.split(" · ") if t.lower() not in KW_JUNK]
    return " · ".join(toks) if len(toks) >= 2 else s


def pack_urls(urls, tag):
    """Split each url after its host and store the hosts once.

    The 20,000 arXiv links share a single prefix and the corpus reuses a few
    dozen hosts, so this costs a small table and saves ~470 KB on the page.
    """
    for u in urls[:200]:
        assert u.startswith("http"), f"{tag}: url column does not look like a url: {u!r}"
    pres, idx, sufs = {}, [], []
    for u in urls:
        m = re.match(r"^(https?://[^/]+/)(.*)$", u)
        pre, suf = m.groups() if m else ("", u)
        idx.append(pres.setdefault(pre, len(pres)))
        sufs.append(suf)
    print(f"  {tag}: {len(urls)} urls, {len(pres)} hosts, "
          f"{sum(len(s) for s in sufs) // 1024} KB of paths")
    return {"pre": list(pres), "i": idx, "s": sufs}


def excerpt(text, max_len=220):
    plain = re.sub(r"[#*_`]+", "", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", plain)
    out = ""
    for p in parts:
        if out and len(out) + len(p) > max_len:
            break
        out = (out + " " + p).strip()
    return out[:max_len + 60]


# ------------------------------------------- space 0: quarter corpus (plate III)
proj = np.load(QUARTER / "projection.npz")
ids = proj["article_ids"]
published = proj["published"]
n = len(ids)

p3 = np.load(QUARTER / "projection3d.npz")
assert (p3["article_ids"] == ids).all(), "projection3d row order mismatch"
# Plates II and III share this space and this slider, so the two exports must be
# the same clustering run written twice — prove it rather than trusting the
# directory names: same rows, same coordinates, same weekly frames.
pf = np.load(QUARTER_FABLE / "projection.npz")
assert np.array_equal(pf["article_ids"], ids) and np.array_equal(pf["coords"], proj["coords"]), \
    f"{QUARTER_FABLE.name} and {QUARTER_TFIDF.name} disagree on the corpus projection"
assert (json.loads((QUARTER_FABLE / "frames.json").read_text())
        == json.loads((QUARTER_TFIDF / "frames.json").read_text())), \
    f"{QUARTER_FABLE.name} and {QUARTER_TFIDF.name} disagree on the weekly frames"

vec = np.fromfile(QUARTER / "projector" / "vectors.bytes", dtype="<f4")
assert vec.size == n * 1536, f"vectors.bytes size {vec.size} != {n}x1536"
vec = vec.reshape(n, 1536)

with open(QUARTER / "projector" / "metadata.tsv") as f:
    rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    qhdr = next(rd)
    qrows = list(rd)
qcol = {name: i for i, name in enumerate(qhdr)}
for req in ("title", "source", "published", "url"):
    assert req in qcol, f"projector metadata.tsv missing column {req!r}: {qhdr[:12]}"
assert len(qrows) == n, (len(qrows), n)
# no id column in the tsv — prove row alignment via the published-date sequence
mism = sum(1 for r, p in zip(qrows, published) if r[qcol["published"]] != str(p))
assert mism == 0, f"{mism}/{n} published-date mismatches tsv vs projection.npz"

titles = [r[qcol["title"]] or "" for r in qrows]
sources = [r[qcol["source"]] for r in qrows]
day_off = [(date.fromisoformat(str(p)) - BASE).days for p in published]

docs = {
    "title": [t[:90] for t in titles],
    "src": [src_class(s) for s in sources],
    "pub": day_off,
    "url": pack_urls([r[qcol["url"]] for r in qrows], "corpus"),
    "pca": flat(pca_sphereize(vec)),
    "umap": flat(unit_rows(p3["coords"] - p3["coords"].mean(axis=0))),
}


# ---------------- views: weekly frames on the shared corpus projection
# Each week of the quarter is clustered independently by the pipeline; the
# slider steps week by week. A frame highlights ONLY that week's clusters;
# that week's remaining documents render as a pale field; everything else in
# the trailing window — including other weeks' clusters — falls back to grey
# dust. Weekly clusters and the corpus's own cluster structure are separate
# namespaces; only the weekly ones are ever drawn.
#
# The same run is read twice:
#   plate II  quarter_without_fable — EVERY cluster, named by raw TF-IDF
#   plate III quarter_with_fable    — ONLY the panel-named clusters, no TF-IDF
# Names restart each week, so a theme may resurface under a different name.


# the panel member whose compression reads as plain description rather than
# synthesis — it introduces each cluster ahead of the Fable metaconcept
GROK = "grok-4.5"


def framed_view(vid, qdir, space, use_fable):
    """Build one weekly-framed view from a quarter export."""
    frames = json.loads((qdir / "frames.json").read_text())
    labels_by_week = comps_by_week = None
    if use_fable:
        labels_by_week = json.loads(
            (qdir / "name_gap_ai_labels_by_week.json").read_text())
        comps_by_week = {
            wk: {str(c["cluster_id"]): c for c in entries}
            for wk, entries in json.loads(
                (qdir / "name_gap_compressions_by_week.json").read_text()).items()
        }
        assert sorted(labels_by_week) == sorted(f["week_end"] for f in frames), \
            f"{vid}: label weeks do not match the frames"

    clusters, pale, wins = [], [], []
    n_skipped = 0
    for fi, frame in enumerate(frames):
        wk_end = frame["week_end"]
        week_idx = frame["week_idx"]
        bg = sorted(frame["background_idx"])
        assert bg == list(range(bg[0], bg[-1] + 1)), f"{vid} frame {fi} window not contiguous"
        assert bg[0] <= min(week_idx) and max(week_idx) == bg[-1], \
            (vid, fi, "week outside window")
        wins.append([bg[0], bg[-1]])

        ai_labels = labels_by_week[wk_end] if use_fable else {}
        compressions = comps_by_week.get(wk_end, {}) if use_fable else {}
        drawn_pts, fr_clusters = set(), []
        for c in frame["clusters"]:
            cid = str(c["cluster_id"])
            if use_fable and (cid not in ai_labels or cid not in compressions):
                n_skipped += 1      # unnamed by the panel: joins the pale field
                continue
            drawn_pts.update(c["points"])
            entry = {
                "name": ai_labels[cid] if use_fable else clean_kw(c["keywords"]),
                "band": c["band"],
                "gap": round(c["name_gap"], 3),
                # the two halves the gap is made of, as percentile ranks within
                # the week: gap = semantic - lexical
                "sem": round(c["semantic_rank"], 3),
                "lex": round(c["lexical_rank"], 3),
                "size": c["size"],
                "fr": fi,
                "rows": [week_idx[p] for p in c["points"]],
                "wk": [c["size"] if k == fi else 0 for k in range(len(frames))],
            }
            if use_fable:
                comp = compressions[cid]
                entry["excerpt"] = excerpt(comp["attractor"])
                entry["attractor"] = comp["attractor"]
                entry["panel"] = list(comp["compressions"].keys())
                # the plain-language read of the cluster, which leads the detail
                # panel: what these documents are, before Fable says what they mean
                assert GROK in comp["compressions"], f"{vid} {wk_end} cluster {cid}: no {GROK}"
                entry["grok"] = comp["compressions"][GROK]
            fr_clusters.append(entry)
        # plate III ranks by name gap; plate II, which shows everything, by size
        fr_clusters.sort(key=lambda c: -c["gap"] if use_fable else -c["size"])
        clusters.extend(fr_clusters)
        pale.append(sorted(week_idx[p] for p in range(len(week_idx))
                           if p not in drawn_pts))

    week_docs = sorted({i for f in frames for i in f["week_idx"]})
    srcmix = {}
    for i in week_docs:
        srcmix[sources[i]] = srcmix.get(sources[i], 0) + 1
    view = {
        "id": vid, "space": space,
        "stops": [f["week_end"] for f in frames], "stopUnit": "week",
        "members": sorted({i for c in clusters for i in c["rows"]}),
        "clusters": clusters,
        "noiseRows": pale,
        "wins": wins,
        "n_docs": len(week_docs),
        "srcmix": srcmix,
        "window": f'{frames[0]["week_end"]}..{frames[-1]["week_end"]}',
    }
    if use_fable:
        view["labeled"] = len(clusters)
        view["unnamed"] = n_skipped
    print(f"{vid}: {len(frames)} weekly frames {view['stops'][0]}..{view['stops'][-1]}, "
          f"{len(clusters)} constellations "
          f"({[sum(1 for c in clusters if c['fr'] == k) for k in range(len(frames))]} per week)"
          + (f", {n_skipped} unnamed clusters folded into the pale field" if use_fable else "")
          + f", pale per week {[len(p) for p in pale]}")
    return view


soc_view = framed_view("social", QUARTER_TFIDF, 0, use_fable=False)
meta_view = framed_view("meta", QUARTER_FABLE, 0, use_fable=True)

# --------- space 2 + view: the 6-month arXiv export (plate I)
# Same pipeline method as the week export (embedding UMAP -> HDBSCAN, TF-IDF
# names); it ships a deterministic 20k-paper sample of the corpus with a
# precomputed 3-D UMAP, so no fitting happens here. EVERY cluster the run
# found becomes a constellation (the key is sorted by size and scrolls; the
# sky's label collision-culling keeps only what fits); noise is dust.

# snapshot the (small) files we read — vectors.bytes is ~120 MB and unused
ARXIV_DIR = ARXIV
ARX_SNAP = Path(__file__).resolve().parent / "arxiv_source"
ARX_FILES = ("arxiv_meta.json", "metadata.tsv", "umap3d.bytes")
if (ARXIV_DIR / "arxiv_meta.json").exists():
    ARX_SNAP.mkdir(exist_ok=True)
    for name in ARX_FILES:
        (ARX_SNAP / name).write_bytes((ARXIV_DIR / name).read_bytes())
else:
    print(f"note: {ARXIV_DIR} missing, using snapshot {ARX_SNAP}")
    ARXIV_DIR = ARX_SNAP

ameta = json.loads((ARXIV_DIR / "arxiv_meta.json").read_text())
with open(ARXIV_DIR / "metadata.tsv") as f:
    rd3 = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    ahdr = next(rd3)
    arows = list(rd3)
# resolve by name, not position: the export grows columns from time to time (an
# `author` column appeared on 2026-08-07), and pinning the exact header meant a
# purely additive upstream change broke the build
acol = {name: i for i, name in enumerate(ahdr)}
for req in ("cluster", "title", "url", "published"):
    assert req in acol, f"arxiv metadata.tsv missing column {req!r}: {ahdr}"
A_CID, A_TITLE, A_URL, A_PUB = (acol["cluster"], acol["title"], acol["url"], acol["published"])
na = len(arows)
assert na == ameta["shipped"], (na, ameta["shipped"])

aum = np.fromfile(ARXIV_DIR / "umap3d.bytes", dtype="<f4")
assert aum.size == na * 3, f"arxiv umap3d size {aum.size} != {na}x3"
aum = aum.reshape(na, 3)

docs_arx6 = {
    "title": [r[A_TITLE][:90] for r in arows],
    "src": [0] * na,
    "pub": [(date.fromisoformat(r[A_PUB]) - BASE).days for r in arows],
    "url": pack_urls([r[A_URL] for r in arows], "arxiv"),
    "umap": flat(unit_rows(aum - aum.mean(axis=0))),
}

arx_months = sorted({r[A_PUB][:7] for r in arows})

def month_end(m):
    y, mo = int(m[:4]), int(m[5:7])
    nxt = date(y + (mo == 12), mo % 12 + 1, 1)
    return (nxt - timedelta(days=1)).isoformat()

arx_stops = [month_end(m) for m in arx_months]

# Activity is a RATE, not a monthly total. Both ends of the window are cut
# short — February starts on the 4th, August stops on the 3rd — so a total
# understates them badly (August ran at 6% of May's total but 71% of its rate).
# Short spans also carry an unrepresentative mix of weekdays: arXiv's weekend
# runs at ~0.58 of a weekday, and 2 of August's 3 days are a weekend. So each
# month gets a denominator in "weekday-equivalent days": every observed day
# counts as its weekday's share of an average day. A 3-day month is then
# measured on the same footing as a 31-day one.
arx_day_n = collections.Counter(r[A_PUB][:10] for r in arows)
_by_wd = collections.defaultdict(list)
for _d, _k in arx_day_n.items():
    _by_wd[date.fromisoformat(_d).weekday()].append(_k)
_mean = lambda xs: sum(xs) / len(xs)
_day_avg = _mean(list(arx_day_n.values()))
wd_factor = {w: _mean(v) / _day_avg for w, v in _by_wd.items()}
arx_day_eq = [
    round(sum(wd_factor[date.fromisoformat(d).weekday()]
              for d in arx_day_n if d[:7] == m), 3)
    for m in arx_months
]
assert all(e > 0 for e in arx_day_eq), arx_day_eq
arx_days_seen = [sum(1 for d in arx_day_n if d[:7] == m) for m in arx_months]
# the export reports its own per-month day counts; the denominator has to match
# the numerator, which is counted off the shipped sample, so the sample's days
# win — but never silently
_meta_days = {m: v["days"] for m, v in ameta.get("months", {}).items()
              if isinstance(v, dict) and "days" in v}
for _m, _seen in zip(arx_months, arx_days_seen):
    if _m in _meta_days and _meta_days[_m] != _seen:
        print(f"  note: {_m}: export reports {_meta_days[_m]} days, the shipped sample "
              f"has papers on {_seen} — using {_seen}, which is what the counts come from")
print("  arxiv activity denominators (weekday-equivalent days per month): "
      + " · ".join(f"{m[5:]} {e:.1f}" for m, e in zip(arx_months, arx_day_eq)))

full_size = {str(c["cluster_id"]): c["size"] for c in ameta["clusters"]}
kw_by_id = {str(c["cluster_id"]): c["keywords"] for c in ameta["clusters"]}
rows_by_cid = {}
for i, r in enumerate(arows):
    rows_by_cid.setdefault(r[A_CID], []).append(i)

arx_clusters = []
for cid in sorted(full_size, key=lambda k: -full_size[k]):
    rows_k = rows_by_cid.get(cid, [])
    if not rows_k:
        continue
    arx_clusters.append({
        "name": clean_kw(kw_by_id[cid]),
        "size": full_size[cid],   # true corpus size; the sky shows the shipped sample
        "rows": rows_k,
        "wk": [sum(1 for k in rows_k if arows[k][A_PUB][:7] == m) for m in arx_months],
    })
arx_members = sorted(i for i, r in enumerate(arows) if r[A_CID] != "noise")
arx_view = {
    "id": "arxiv", "space": 1,
    "stops": arx_stops, "stopUnit": "month",
    # colour divides each cluster's monthly count by this to get papers/day
    "dayEq": arx_day_eq,
    "days": arx_days_seen,
    "partialLast": arx_months[-1] == date.today().isoformat()[:7],
    "members": arx_members,
    "clusters": arx_clusters,
    "docs_total": ameta["papers"],
    "shipped": na,
    "dust_total": ameta["noise"],
    "dust_kept": na - len(arx_members),
    "total_clusters": len(ameta["clusters"]),
}
print(f"arxiv (mindspace export): {ameta['papers']} papers {ameta['window']}, "
      f"shipped {na} ({len(arx_members)} clustered + {na - len(arx_members)} noise dust); "
      f"{len(arx_clusters)} of {len(ameta['clusters'])} clusters on the sky "
      f"(full sizes {[c['size'] for c in arx_clusters][:6]}...{[c['size'] for c in arx_clusters][-3:]})")

# --------------------------------------------------------------------- emit
data = {
    "meta": {
        "built_from": "mindspaceai/data/output/{quarter_without_fable,quarter_with_fable,arxiv}",
        "week_end": soc_view["stops"][-1],
        "base_date": BASE.isoformat(),
        "n_docs": n,
        "sources": ["arxiv", "x", "web"],
    },
    "spaces": [docs, docs_arx6],     # 0 = quarter corpus (plates II & III), 1 = arXiv
    "views": [arx_view, soc_view, meta_view],
}
DEST.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
kb = DEST.stat().st_size / 1024
print(f"wrote {DEST} ({kb:.0f} KB)")
print(f"corpus space: {n} docs {min(published)}..{max(published)}, "
      f"{soc_view['n_docs']} of them inside the 11 weekly windows, "
      f"mix {sorted(soc_view['srcmix'].items(), key=lambda t: -t[1])[:4]}")
