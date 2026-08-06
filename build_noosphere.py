#!/usr/bin/env python3
"""Inject noosphere_data.json into noosphere_template.html.

  python3 build_noosphere.py [--data noosphere_data.json] [--out noosphere.html] \
      [--title "Reports from the future"] [--standalone]

Default output is an artifact fragment (the claude.ai wrapper adds the document
shell). --standalone wraps it in a full <!doctype html> document for plain
static hosting (Vercel, nginx, file://).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="noosphere_data.json")
ap.add_argument("--out", default="noosphere.html")
ap.add_argument("--title", default="Reports from the future")
ap.add_argument("--standalone", action="store_true",
                help="emit a complete HTML document (for Vercel/static hosting)")
args = ap.parse_args()

enc = json.dumps(json.loads((HERE / args.data).read_text()),
                 separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
tpl = (HERE / "noosphere_template.html").read_text()
out = tpl.replace("/*__DATA_JSON__*/", enc).replace("__RUN_TITLE__", args.title)

if args.standalone:
    title_tag = f"<title>{args.title}</title>"
    body = out.replace(title_tag, "", 1)
    out = (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
           f"{title_tag}\n</head>\n<body>\n{body}\n</body>\n</html>\n")

(HERE / args.out).write_text(out)
print(f"wrote {HERE / args.out} ({len(out)/1024:.0f} KB){' [standalone]' if args.standalone else ''}")
