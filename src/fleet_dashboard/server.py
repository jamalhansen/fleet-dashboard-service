"""FastAPI app: a handful of read-only JSON endpoints plus the static
frontend. No auth -- read-only status data, meant for LAN access only (bind
to 0.0.0.0, same shape as japanese-tutor); nothing here holds a secret or
lets a caller change anything."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import core

app = FastAPI(title="Fleet Dashboard")

VAULT_PATH = os.environ.get("FLEET_DASHBOARD_VAULT_PATH") or str(Path.home() / "vaults" / "Contexta")
ART_ITEMS_DIR = os.environ.get("FLEET_DASHBOARD_ART_ITEMS_DIR") or "~/iCloud/ai-artist/items"
JAPANESE_TUTOR_URL = os.environ.get("JAPANESE_TUTOR_URL") or "http://127.0.0.1:8421"

# Jamal's two Obsidian vaults with the ops/ structure vault-health and
# tensions read: Contexta (thinking) and KeySix (work thinking, added
# 2026-09-20 -- was hardcoded to Contexta only, so KeySix's own
# observations/tensions/inbox were invisible from the dashboard).
VAULTS = {
    "Contexta": VAULT_PATH,
    "KeySix": os.environ.get("FLEET_DASHBOARD_KEYSIX_PATH") or str(Path.home() / "vaults" / "KeySix"),
}


@app.get("/api/fleet")
def api_fleet(lookback_hours: float = 24 * 7):
    activity = core.get_fleet_activity(lookback_hours=lookback_hours)
    services = core.get_launch_agents()
    return {
        "activity": [
            {
                "tool_name": a.tool_name,
                "total": a.total,
                "failures": a.failures,
                "failure_rate": round(a.failure_rate, 4),
                "last_call": a.last_call,
                "tables": sorted(set(a.tables)),
            }
            for a in activity
        ],
        "services": [
            {
                "label": s.label,
                "running": s.running,
                "pid": s.pid,
                "keep_alive": s.keep_alive,
                "last_exit_code": s.last_exit_code,
            }
            for s in services
        ],
    }


@app.get("/api/models")
def api_models(lookback_hours: float = 24 * 7):
    usage = core.get_model_usage(lookback_hours=lookback_hours)
    by_provider = core.get_provider_usage(lookback_hours=lookback_hours)
    return {
        "usage": [
            {
                "tool_name": u.tool_name,
                "provider": u.provider,
                "model": u.model,
                "total": u.total,
                "failures": u.failures,
                "failure_rate": round(u.failure_rate, 4),
                "via_gateway": u.via_gateway,
            }
            for u in usage
        ],
        "by_provider": [
            {
                "provider": p.provider,
                "total": p.total,
                "failures": p.failures,
                "failure_rate": round(p.failure_rate, 4),
                "tool_count": p.tool_count,
                "model_count": p.model_count,
            }
            for p in by_provider
        ],
    }


def _resolve_vault_path(vault: str) -> str:
    if vault not in VAULTS:
        raise HTTPException(status_code=404, detail=f"Unknown vault '{vault}'. Known: {sorted(VAULTS)}")
    return VAULTS[vault]


@app.get("/api/tensions")
def api_tensions(vault: str = "Contexta"):
    t = core.get_tension_summary(_resolve_vault_path(vault))
    return {
        "pending_count": t.pending_count,
        "active_count": t.active_count,
        "recent_titles": t.recent_titles,
    }


@app.get("/api/vault-health")
def api_vault_health(vault: str = "Contexta"):
    h = core.get_vault_health(_resolve_vault_path(vault))
    return {
        "observations_pending": h.observations_pending,
        "inbox_count": h.inbox_count,
        "inbox_oldest_days": round(h.inbox_oldest_days, 1) if h.inbox_oldest_days is not None else None,
        "last_health_check": h.last_health_check,
    }


@app.get("/api/frontmatter-validation")
def api_frontmatter_validation():
    v = core.get_frontmatter_validation()
    if not v.available:
        return {"available": False}
    return {
        "available": True,
        "generated_at": v.generated_at,
        "total": v.total,
        "invalid_count": v.invalid_count,
        "invalid_files": v.invalid_files,
        "error_summary": v.error_summary,
        "content_todo": v.content_todo,
    }


@app.get("/api/repo-health")
def api_repo_health():
    h = core.get_repo_health()
    if not h.available:
        return {"available": False}
    return {
        "available": True,
        "generated_at": h.generated_at,
        "total": h.total,
        "healthy": h.healthy,
        "repos": h.repos,
    }


@app.get("/api/gateway-routing")
def api_gateway_routing():
    a = core.get_gateway_routing_audit()
    if not a.available:
        return {"available": False}
    return {
        "available": True,
        "generated_at": a.generated_at,
        "entries": [
            {
                "tool_name": e.tool_name,
                "category": e.category,
                "status": e.status,
                "last_call": e.last_call,
            }
            for e in a.entries
        ],
    }


@app.get("/api/art")
def api_art():
    item = core.get_latest_art(ART_ITEMS_DIR)
    if not item:
        return {"available": False}
    return {
        "available": True,
        "title": item.title,
        "self_score": item.self_score,
        "interest": item.interest,
        "generated_at": item.generated_at,
        "image_url": "/api/art/image",
    }


@app.get("/api/art/image")
def api_art_image():
    item = core.get_latest_art(ART_ITEMS_DIR)
    if not item or not item.image_path.exists():
        raise HTTPException(status_code=404, detail="No image available")
    return FileResponse(item.image_path)


@app.get("/api/japanese-tutor")
def api_japanese_tutor():
    j = core.get_japanese_tutor_summary(JAPANESE_TUTOR_URL)
    return {
        "reachable": j.reachable,
        "cards_due": j.cards_due,
        "new_count": j.new_count,
        "review_count": j.review_count,
        "mastery": j.mastery,
        "reviews_today_attempts": j.reviews_today_attempts,
        "reviews_today_distinct_cards": j.reviews_today_distinct_cards,
    }


def mount_static() -> None:
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


mount_static()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("FLEET_DASHBOARD_PORT", "8422"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
