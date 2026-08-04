#!/usr/bin/env python3
"""Inject noosphere_data.json into noosphere_template.html.

  python3 build_noosphere.py [--data noosphere_data.json] [--out noosphere.html] \
      [--title "Armillary Noosphere"]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="noosphere_data.json")
ap.add_argument("--out", default="noosphere.html")
ap.add_argument("--title", default="Armillary Noosphere")
args = ap.parse_args()

enc = json.dumps(json.loads((HERE / args.data).read_text()),
                 separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
tpl = (HERE / "noosphere_template.html").read_text()
out = tpl.replace("/*__DATA_JSON__*/", enc).replace("__RUN_TITLE__", args.title)
(HERE / args.out).write_text(out)
print(f"wrote {HERE / args.out} ({len(out)/1024:.0f} KB)")
