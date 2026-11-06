#!/usr/bin/env python3
"""Build the Grabber bookmarklet href from app/grabber.js and inject it into
app/dashboard.html (replaces __GRABBER_HREF__). Run after editing grabber.js."""
import re
import urllib.parse
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = (root / "app" / "grabber.js").read_text(encoding="utf-8")

iife = src[src.index("(function () {"):]
# Collapse real newlines to spaces (all string literals use \n escapes).
mini = re.sub(r"\s*\n\s*", " ", iife).strip()
mini = re.sub(r" {2,}", " ", mini)
href = "javascript:" + urllib.parse.quote(mini, safe="")

dash_path = root / "app" / "dashboard.html"
dash = dash_path.read_text(encoding="utf-8")
if "__GRABBER_HREF__" in dash:
    dash = dash.replace("__GRABBER_HREF__", href)
else:
    # Re-inject: replace the existing javascript: href of the grabber button.
    dash, n = re.subn(r'href="javascript:[^"]*" id="grabber-btn"',
                      f'href="{href}" id="grabber-btn"', dash)
    dash, m = re.subn(r'(id="grabber-code"[^>]*>)javascript:[^<]*',
                      r"\g<1>" + href, dash)
    print(f"re-injected (btn={n}, code={m})")
dash_path.write_text(dash, encoding="utf-8")
print(f"bookmarklet length: {len(href)} chars")
