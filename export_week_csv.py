#!/usr/bin/env python3
"""Export one week of the quarter run as CSV — documents and clusters.

    uv run build/export_week_csv.py                 # the latest week
    uv run build/export_week_csv.py --week 2026-07-21
    uv run build/export_week_csv.py --all           # every week, one pair of files

Writes OUTSIDE this repo (~/noosphere-authors by default, same convention as
authors_tui.py): the rows carry an author column in the clear and this repo is
public. Pass --no-authors to drop that column, which makes the output safe to
put anywhere.

Joins what the three plates are built from:
  quarter_without_fable/frames.json          per-week clustering + TF-IDF keywords
  quarter_with_fable/name_gap_*_by_week.json Fable names + panel readings
  quarter_with_fable/projector/metadata.tsv  title, source, author, url, published
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

MS_OUT = Path("/root/michael-folder/mindspaceai/data/output")
csv.field_size_limit(10 ** 9)


def pick(*names):
    for nm in names:
        if (MS_OUT / nm).is_dir():
            return MS_OUT / nm
    raise SystemExit(f"none of {names} found under {MS_OUT}")


QUARTER_TFIDF = pick("quarter_without_fable")
QUARTER_FABLE = pick("quarter_with_fable", "quarter")
QUARTER = QUARTER_TFIDF if (QUARTER_TFIDF / "projector").is_dir() else QUARTER_FABLE

ap = argparse.ArgumentParser()
ap.add_argument("--week", help="week_end, e.g. 2026-08-04 (default: the latest)")
ap.add_argument("--all", action="store_true", help="every week in one file")
ap.add_argument("--out", type=Path, default=Path.home() / "noosphere-authors")
ap.add_argument("--no-authors", action="store_true", help="drop the author column")
args = ap.parse_args()

frames = json.loads((QUARTER_TFIDF / "frames.json").read_text())
assert frames == json.loads((QUARTER_FABLE / "frames.json").read_text()), \
    "the two quarter exports disagree on the weekly frames"

labels = json.loads((QUARTER_FABLE / "name_gap_ai_labels_by_week.json").read_text())
comps = {wk: {str(c["cluster_id"]): c for c in rows}
         for wk, rows in json.loads(
             (QUARTER_FABLE / "name_gap_compressions_by_week.json").read_text()).items()}

with open(QUARTER / "projector" / "metadata.tsv") as f:
    rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    hdr = next(rd)
    meta = list(rd)
col = {name: i for i, name in enumerate(hdr)}
get = lambda row, name: row[col[name]] if name in col else ""

X_SOURCES = {"Twitter/Grok", "Desearch X"}
HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
URL_HANDLE_RE = re.compile(r"^https?://(?:www\.)?(?:x|twitter)\.com/([^/]+)/status/")


def handles_of(row):
    """(@handle, handle-per-the-url, conflict?) for an X post, else blanks.

    The author column and the post URL disagree for a handful of rows upstream,
    and some of the URL handles are malformed (a space, a dot). The author
    column is always a well-formed handle, so it wins — but the conflict is
    reported rather than hidden, because @-ing the wrong account is not a
    mistake you can take back.
    """
    if get(row, "source") not in X_SOURCES:
        return "", "", ""
    author = get(row, "author").strip()
    m = URL_HANDLE_RE.match(get(row, "url"))
    from_url = (m.group(1).strip() if m else "")
    at = "@" + author if HANDLE_RE.match(author) else ""
    conflict = int(bool(from_url) and bool(author)
                   and from_url.lower() != author.lower())
    return at, from_url, conflict


weeks = [f["week_end"] for f in frames]
if args.all:
    want = weeks
elif args.week:
    if args.week not in weeks:
        sys.exit(f"no such week {args.week!r}; have {', '.join(weeks)}")
    want = [args.week]
else:
    want = [weeks[-1]]

DOC_COLS = ["week_end", "published", "source", "title", "url"] \
    + ([] if args.no_authors else ["author"]) \
    + ["handle", "handle_from_url", "handle_conflict",
       "cluster_id", "tfidf_name", "fable_name", "band", "name_gap",
       "semantic_rank", "lexical_rank", "cluster_size", "on_plate_ii", "on_plate_iii"]
CL_COLS = ["week_end", "cluster_id", "size", "tfidf_keywords", "fable_name", "band",
           "name_gap", "semantic_rank", "lexical_rank", "on_plate_ii", "on_plate_iii",
           "x_posts", "x_handles", "grok_summary", "fable_attractor"]

doc_rows, cl_rows, handle_rows = [], [], {}
for frame in frames:
    wk = frame["week_end"]
    if wk not in want:
        continue
    week_idx = frame["week_idx"]
    named = labels.get(wk, {})
    comp_wk = comps.get(wk, {})
    # every document of the week starts unclustered; cluster members overwrite
    by_doc = {i: None for i in week_idx}
    for c in frame["clusters"]:
        cid = str(c["cluster_id"])
        fable = named.get(cid, "")
        comp = comp_wk.get(cid)
        # plate III draws a cluster only when it has BOTH a name and a reading
        on3 = bool(fable) and comp is not None
        cl_rows.append({
            "week_end": wk, "cluster_id": cid, "size": c["size"],
            "tfidf_keywords": c["keywords"], "fable_name": fable, "band": c["band"],
            "name_gap": round(c["name_gap"], 4),
            "semantic_rank": round(c["semantic_rank"], 4),
            "lexical_rank": round(c["lexical_rank"], 4),
            "on_plate_ii": 1, "on_plate_iii": int(on3),
            "grok_summary": (comp or {}).get("compressions", {}).get("grok-4.5", ""),
            "fable_attractor": (comp or {}).get("attractor", ""),
        })
        for p in c["points"]:
            by_doc[week_idx[p]] = (c, fable, on3)

    for i in week_idx:
        hit = by_doc[i]
        c, fable, on3 = hit if hit else (None, "", False)
        row = {
            "week_end": wk,
            "published": get(meta[i], "published"),
            "source": get(meta[i], "source"),
            "title": get(meta[i], "title"),
            "url": get(meta[i], "url"),
            "cluster_id": c["cluster_id"] if c else "",
            "tfidf_name": c["keywords"] if c else "",
            "fable_name": fable,
            "band": c["band"] if c else "",
            "name_gap": round(c["name_gap"], 4) if c else "",
            "semantic_rank": round(c["semantic_rank"], 4) if c else "",
            "lexical_rank": round(c["lexical_rank"], 4) if c else "",
            "cluster_size": c["size"] if c else "",
            "on_plate_ii": int(bool(c)),
            "on_plate_iii": int(on3),
        }
        at, from_url, conflict = handles_of(meta[i])
        row["handle"], row["handle_from_url"], row["handle_conflict"] = at, from_url, conflict
        if not args.no_authors:
            row["author"] = get(meta[i], "author")
        doc_rows.append(row)
        if at:
            h = handle_rows.setdefault(
                (wk, at), {"week_end": wk, "handle": at, "posts": 0,
                           "clusters": set(), "conflicts": 0, "urls": []})
            h["posts"] += 1
            h["conflicts"] += conflict
            h["urls"].append(row["url"])
            if c:
                h["clusters"].add(fable or c["keywords"])

# roll the handles up onto their clusters, paste-ready, most-posted first
per_cluster = {}
for r in doc_rows:
    if r["handle"] and r["cluster_id"] != "":
        per_cluster.setdefault((r["week_end"], str(r["cluster_id"])), []).append(r["handle"])
for c in cl_rows:
    hs = per_cluster.get((c["week_end"], c["cluster_id"]), [])
    seen = {}
    for h in hs:
        seen[h] = seen.get(h, 0) + 1
    c["x_posts"] = len(hs)
    c["x_handles"] = " ".join(h for h, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])))

HANDLE_COLS = ["week_end", "handle", "posts", "first_url", "urls",
               "clusters", "handle_conflicts"]
h_rows = [{"week_end": v["week_end"], "handle": v["handle"], "posts": v["posts"],
           # first_url is the one to quote; urls keeps the rest for the handles
           # that posted more than once, so nothing is dropped
           "first_url": v["urls"][0] if v["urls"] else "",
           "urls": " | ".join(v["urls"]),
           "clusters": " | ".join(sorted(v["clusters"])), "handle_conflicts": v["conflicts"]}
          for v in sorted(handle_rows.values(), key=lambda v: (v["week_end"], -v["posts"], v["handle"]))]

args.out.mkdir(parents=True, exist_ok=True)
tag = "all-weeks" if args.all else want[0]
for name, cols, rows in (("documents", DOC_COLS, doc_rows), ("clusters", CL_COLS, cl_rows),
                         ("x-handles", HANDLE_COLS, h_rows)):
    path = args.out / f"noosphere-{tag}-{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows, {path.stat().st_size // 1024} KB)")

for wk in want:
    d = [r for r in doc_rows if r["week_end"] == wk]
    c = [r for r in cl_rows if r["week_end"] == wk]
    hs = [r for r in h_rows if r["week_end"] == wk]
    bad = sum(r["handle_conflicts"] for r in hs)
    print(f"  {wk}: {len(d)} documents · {len(c)} clusters "
          f"({sum(r['on_plate_iii'] for r in c)} of them named on plate III) · "
          f"{sum(1 for r in d if not r['cluster_id'])} unclustered · "
          f"{len(hs)} distinct X handles"
          + (f"  [{bad} whose author column disagrees with their post url — check before @-ing]"
             if bad else ""))
