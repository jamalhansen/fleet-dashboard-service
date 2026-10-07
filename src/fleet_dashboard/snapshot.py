"""Write the dashboard as one self-contained HTML file, for reading away from home.

Every read-only endpoint is evaluated up front -- once per lookback option the page
offers and once per vault -- and baked into the page along with the art image. A
small shim answers the page's fetch() calls from that data, so index.html itself is
unchanged. The page is then rendered once in headless Chromium and saved with its
scripts stripped: iOS opens a local HTML file without running JavaScript, so the
file has to be finished HTML. The file is overwritten each run.
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from fastapi.routing import APIRoute

from . import core, server

INDEX = Path(__file__).parent / "static" / "index.html"
DEFAULT_OUT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Dashboards" / "fleet.html"
ART_IMAGE_PATH = "/api/art/image"
ART_MAX_PX = 900


def lookback_options(html: str) -> list[str]:
    select = re.search(r'<select id="lookback-select">(.*?)</select>', html, re.DOTALL)
    return re.findall(r'<option value="(\d+)"', select.group(1)) if select else []


def _routes() -> dict[str, Callable]:
    return {
        r.path: r.endpoint
        for r in server.app.routes
        if isinstance(r, APIRoute) and r.methods and "GET" in r.methods and r.path != ART_IMAGE_PATH
    }


def collect(lookbacks: list[str], vaults: list[str]) -> dict[str, object]:
    """URL (exactly as index.html builds it) -> JSON body."""
    data: dict[str, object] = {}
    for path, endpoint in _routes().items():
        params = inspect.signature(endpoint).parameters
        if "lookback_hours" in params:
            for h in lookbacks:
                data[f"{path}?lookback_hours={h}"] = endpoint(lookback_hours=float(h))
        elif "vault" in params:
            for v in vaults:
                data[f"{path}?vault={v}"] = endpoint(vault=v)
        else:
            data[path] = endpoint()
    return data


def art_data_url(image: Path, max_px: int = ART_MAX_PX) -> str:
    """The latest art image, downscaled to a JPEG (macOS sips) so the page stays small."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "art.jpg"
        proc = subprocess.run(
            ["sips", "-Z", str(max_px), "-s", "format", "jpeg", str(image), "--out", str(out)],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and out.exists():
            payload, mime = out.read_bytes(), "image/jpeg"
        else:
            payload, mime = image.read_bytes(), mimetypes.guess_type(image.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def tutor_ui_url() -> str | None:
    """Where the snapshot's "Open" link to japanese-tutor should point.

    The live page derives it from its own hostname, but a snapshot is rendered from
    a local file (no hostname: the link came out as "http://:8421") and is then
    opened on a phone or served by the Pi, neither of which runs the tutor. Use
    FLEET_DASHBOARD_TUTOR_URL if set, else this Mac's Bonjour name, which phones
    and the Pi resolve on the home network and which survives a DHCP change.
    """
    explicit = os.environ.get("FLEET_DASHBOARD_TUTOR_URL")
    if explicit:
        return explicit
    try:
        proc = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True, check=False)
    except FileNotFoundError:  # scutil is macOS-only; CI runs on Linux
        return None
    name = proc.stdout.strip()
    return f"http://{name}.local:8421" if proc.returncode == 0 and name else None


_SHIM = """<script>
window.JAPANESE_TUTOR_UI_URL = %s;
window.__SNAPSHOT__ = %s;
window.fetch = async (url) => {
  const key = String(url);
  if (key in window.__SNAPSHOT__) {
    return new Response(JSON.stringify(window.__SNAPSHOT__[key]),
      {status: 200, headers: {"Content-Type": "application/json"}});
  }
  return new Response(JSON.stringify({detail: "not in snapshot"}), {status: 404});
};
</script>
"""


def render(html: str, data: dict[str, object], generated: datetime) -> str:
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    note = (
        '<p style="margin:.25rem 0 0;font-size:.85rem;opacity:.7">'
        f"Snapshot from {generated.strftime('%a %b %-d, %-I:%M %p')}</p>"
    )
    html = html.replace("<h1>Fleet Dashboard</h1>", "<h1>Fleet Dashboard</h1>" + note, 1)
    # "updated <now>" and Refresh describe the live page; in a snapshot the note above is the truth.
    html = html.replace("</head>", "<style>#updated-at, #refresh-btn { display: none; }</style>\n</head>", 1)
    first_script = html.index("<script>")
    tutor = json.dumps(tutor_ui_url()).replace("</", "<\\/")  # JS string, or null to keep the page's default
    return html[:first_script] + _SHIM % (tutor, payload) + html[first_script:]


_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STATIC_CSS = "<style>#lookback-select { display: none; }</style>\n</head>"


def prerender(page: str, timeout_ms: int = 20_000) -> str:
    """Run the page's JavaScript in headless Chromium; return the finished DOM without scripts."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            tab = browser.new_page()
            tab.set_content(page, wait_until="load")
            # Every panel starts as a .skeleton placeholder; wait until all are replaced.
            # A timeout raises, so a half-rendered page never overwrites the last good one.
            tab.wait_for_function("() => document.querySelectorAll('.skeleton').length === 0", timeout=timeout_ms)
            html = tab.content()
        finally:
            browser.close()
    return strip_scripts(html)


def strip_scripts(html: str) -> str:
    """Drop scripts (inert on iOS anyway) and the picker, which can't work without them."""
    return _SCRIPT.sub("", html).replace("</head>", _STATIC_CSS, 1)


def write_snapshot(out: Path = DEFAULT_OUT, now: datetime | None = None) -> Path:
    html = INDEX.read_text(encoding="utf-8")
    data = collect(lookback_options(html), list(server.VAULTS))
    art = data.get("/api/art")
    if isinstance(art, dict) and art.get("available"):
        item = core.get_latest_art(server.ART_ITEMS_DIR)
        if item and item.image_path.exists():
            art["image_url"] = art_data_url(item.image_path)
        else:
            art["available"] = False
    page = prerender(render(html, data, now or datetime.now().astimezone()))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, out)
    return out


def publish(path: Path, target: str) -> bool:
    """Copy the finished snapshot to `host:path` (scp, key auth) so another machine can serve it.

    A failed push never fails the run: the local/iCloud copy is already written, and the
    next scheduled run tries again. Returns whether the copy landed.
    """
    try:
        result = subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(path), target],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"publish to {target} failed: {e}")
        return False
    if result.returncode != 0:
        print(f"publish to {target} failed: {result.stderr.strip() or result.returncode}")
        return False
    return True


def main() -> None:
    out = Path(os.environ.get("FLEET_DASHBOARD_SNAPSHOT_PATH") or DEFAULT_OUT).expanduser()
    path = write_snapshot(out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    # e.g. clifford:dashboard/index.html -- the Raspberry Pi serves whatever lands there.
    target = os.environ.get("FLEET_DASHBOARD_PUBLISH_TO")
    if target and publish(path, target):
        print(f"published to {target}")


if __name__ == "__main__":
    main()
