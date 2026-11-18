"""Headless screenshots of Folium maps / the UI (used for the report figures).

CDN requests (Leaflet, jQuery, Bootstrap, awesome-markers) are served from a local
vendor folder when the machine has no CDN access. Map tiles are optional: if they
cannot be fetched the map renders on a blank background (road network is drawn
from the dataset itself).

usage: python tools/screenshot.py <url-or-html> <out.png> [vendor_dir] [width] [height] [--click selector]...
"""
import os
import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
vendor = sys.argv[3] if len(sys.argv) > 3 else ""
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1400
H = int(sys.argv[5]) if len(sys.argv) > 5 else 900
clicks = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--click"]
view = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--view"]  # "lat,lon,zoom"
if not url.startswith("http"):
    url = "file://" + os.path.abspath(url)

LOCAL = {}
if vendor:
    for root, _, files in os.walk(vendor):
        for f in files:
            LOCAL.setdefault(f, os.path.join(root, f))


def handle(route):
    u = route.request.url
    name = u.split("?")[0].rstrip("/").split("/")[-1]
    if u.startswith(("file://", "http://localhost", "http://127.0.0.1")):
        return route.continue_()
    if name in LOCAL:
        ctype = "text/css" if name.endswith(".css") else "application/javascript" if name.endswith(".js") else None
        return route.fulfill(path=LOCAL[name], content_type=ctype) if ctype else route.fulfill(path=LOCAL[name])
    return route.abort()


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.route("**/*", handle)
    pg.goto(url, wait_until="load", timeout=60000)
    pg.wait_for_timeout(2500)
    if view:
        la, lo, z = view[0].split(",")
        pg.evaluate(f"for (const k in window) {{ try {{ if (window[k] instanceof L.Map) window[k].setView([{la},{lo}],{z}); }} catch(e) {{}} }}")
        pg.wait_for_timeout(1500)
    for sel in clicks:
        pg.click(sel); pg.wait_for_timeout(1500)
    if "--ui" in sys.argv:
        pg.wait_for_function("document.querySelector('#status') && /Done|failed/.test(document.querySelector('#status').textContent)",
                             timeout=300000)
        pg.wait_for_timeout(1500)
    pg.screenshot(path=out)
    b.close()
print("saved", out)
