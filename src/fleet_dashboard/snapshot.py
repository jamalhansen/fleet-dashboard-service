"""Write the dashboard as one self-contained HTML file, for reading away from home.

Every read-only endpoint is evaluated up front -- once per lookback option the page
offers and once per vault -- and baked into the page along with the art image. A
small shim answers the page's fetch() calls from that data, so index.html itself is
unchanged and the lookback picker still works. The file is overwritten each run.
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
        r.path: r.endpoint for r in server.app.routes
        if isinstance(r, APIRoute) and "GET" in r.methods and r.path != ART_IMAGE_PATH
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
            capture_output=True, check=False,
        )
        if proc.returncode == 0 and out.exists():
            payload, mime = out.read_bytes(), "image/jpeg"
        else:
            payload, mime = image.read_bytes(), mimetypes.guess_type(image.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


_SHIM = """<script>
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
    return html[:first_script] + _SHIM % payload + html[first_script:]


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
    page = render(html, data, now or datetime.now().astimezone())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, out)
    return out


def main() -> None:
    out = Path(os.environ.get("FLEET_DASHBOARD_SNAPSHOT_PATH") or DEFAULT_OUT).expanduser()
    path = write_snapshot(out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
