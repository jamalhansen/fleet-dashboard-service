# fleet-dashboard-service

Read-only local web dashboard aggregating tool fleet health, vault tensions, artist-agent's latest image, and japanese-tutor's study stats -- one page instead of four.

Not a CLI tool -- a FastAPI service (`server.py` + `core.py`), so it has no `cli.py`/`main.py`/`logic.py` entry point.

## Installation
```bash
uv sync
```

## Usage
```bash
uv run fleet-dashboard-service
```

Starts the server on `FLEET_DASHBOARD_PORT` (default `8422`), serving a static frontend plus:

- `GET /api/fleet` -- per-tool health/status across the fleet
- `GET /api/models` -- LLM usage stats by tool
- `GET /api/tensions` -- open vault tensions
- `GET /api/vault-health` -- vault health snapshot
- `GET /api/frontmatter-validation` -- latest frontmatter-validator run, bucketed into an actionable to-do list
- `GET /api/art` / `GET /api/art/image` -- artist-agent's latest generated piece
- `GET /api/japanese-tutor` -- due-card count and today's review stats

Meant to run continuously (see `com.localfirst.fleet-dashboard-service` LaunchAgent) rather than be invoked per-task.
