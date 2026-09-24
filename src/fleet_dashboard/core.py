"""Data-gathering for the dashboard. Every function here is read-only and
never raises -- a data source being unreachable (japanese-tutor not running,
no tensions yet) degrades that section, not the whole page."""
from __future__ import annotations

import json
import plistlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import frontmatter
import httpx
from local_first_common.tracking import get_tracking_db_path

LOCALFIRST_PREFIXES = ("com.localfirst.", "com.jamalhansen.")


# ---------------------------------------------------------------------------
# Fleet health (processing_log / fetch_log / api_call_log + launchctl)
# ---------------------------------------------------------------------------

_STATS_QUERIES: dict[str, str] = {
    # via_gateway IS NOT TRUE excludes llm-gateway-service's own echo of a
    # gateway-routed call -- the calling tool's own timed_run() row already
    # counts that call once; counting both doubles every gateway-routed
    # tool's total (confirmed live 2026-09-20: obsidian-vault-auto-tagger's
    # real ~81 calls showed as ~152).
    "processing_log": """
        SELECT tool_name, COUNT(*) AS total,
               SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failures,
               MAX(created_at) AS last_call
        FROM processing_log WHERE created_at > ? AND via_gateway IS NOT TRUE GROUP BY tool_name
    """,
    "fetch_log": """
        SELECT t.name AS tool_name, COUNT(*) AS total,
               SUM(CASE WHEN NOT fl.success THEN 1 ELSE 0 END) AS failures,
               MAX(fl.attempted_at) AS last_call
        FROM fetch_log fl JOIN tools t ON fl.tool_id = t.id
        WHERE fl.attempted_at > ? GROUP BY t.name
    """,
    "api_call_log": """
        SELECT t.name AS tool_name, COUNT(*) AS total,
               SUM(CASE WHEN NOT acl.success THEN 1 ELSE 0 END) AS failures,
               MAX(acl.attempted_at) AS last_call
        FROM api_call_log acl JOIN tools t ON acl.tool_id = t.id
        WHERE acl.attempted_at > ? GROUP BY t.name
    """,
}


@dataclass
class ToolActivity:
    tool_name: str
    total: int = 0
    failures: int = 0
    last_call: str | None = None
    tables: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


def get_fleet_activity(lookback_hours: float = 24 * 7) -> list[ToolActivity]:
    """Per-tool call activity over the lookback window, merged across all
    three log tables. Never raises: a lock conflict or missing DB yields []."""
    db_path = get_tracking_db_path()
    if not db_path.exists():
        return []

    cutoff = datetime.now() - timedelta(hours=lookback_hours)  # noqa: DTZ005 - must stay naive to match these tables' naive CURRENT_TIMESTAMP columns
    merged: dict[str, ToolActivity] = {}
    try:
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            for table, query in _STATS_QUERIES.items():
                for tool_name, total, failures, last_call in conn.execute(query, [cutoff]).fetchall():
                    entry = merged.setdefault(tool_name, ToolActivity(tool_name=tool_name))
                    entry.total += total
                    entry.failures += failures or 0
                    entry.tables.append(table)
                    last_call_str = str(last_call)
                    if entry.last_call is None or last_call_str > entry.last_call:
                        entry.last_call = last_call_str
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - best-effort read against a DB other tools may be writing to concurrently
        return []

    return sorted(merged.values(), key=lambda t: t.last_call or "", reverse=True)


@dataclass
class ModelUsage:
    tool_name: str
    model: str
    provider: str
    total: int
    failures: int
    via_gateway: bool = False

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


@dataclass
class ProviderUsage:
    provider: str
    total: int
    failures: int
    tool_count: int
    model_count: int

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


_MODEL_USAGE_QUERY = """
    SELECT tool_name, model, provider, via_gateway,
           COUNT(*) AS total, SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failures
    FROM processing_log
    WHERE created_at > ?
    GROUP BY tool_name, model, provider, via_gateway
"""

# processing_log has carried a real `provider` column since 2026-09-20
# (local_first_common.tracking's log_run(provider=...)), populated for
# gateway-routed calls (the dominant path -- llm-gateway-service already
# knows its caller-requested provider for free) and any direct provider
# whose class now declares provider_name. Older rows, and any call site that
# hasn't been updated to pass provider=, still have it NULL -- classify_provider()
# below is the fallback heuristic for exactly those, against real strings
# observed in production before the column existed ("ollama:phi4-mini",
# "anthropic:claude-haiku-4-5-20251001", bare "phi4-mini", "claude-sonnet-5",
# "deepseek-chat", "llama-3.3-70b-versatile", test leakage like MagicMock
# reprs) -- not authoritative, just a best guess for legacy rows. A model
# string that doesn't match any known pattern defaults to "ollama", since
# every uncataloged local model tag observed so far (qwen2.5:7b,
# gemma4:latest, llava:7b, ...) is one.
_KNOWN_PROVIDER_PREFIXES = {"ollama", "local", "anthropic", "gemini", "groq", "deepseek", "mock"}
_KNOWN_GROQ_MODELS = {"llama-3.3-70b-versatile"}


def classify_provider(model: str | None) -> tuple[str, str]:
    """Best-effort (provider, display_model) from a raw processing_log.model
    string, for rows with no real `provider` column value. See the module
    comment above _MODEL_USAGE_QUERY for why this is a fallback heuristic,
    not a lookup against real data."""
    if not model:
        return "(unset)", "(unset)"
    if ":" in model:
        prefix, _, rest = model.partition(":")
        if prefix in _KNOWN_PROVIDER_PREFIXES:
            return prefix, rest or "(unset)"
    lowered = model.lower()
    if lowered.startswith("<"):  # a repr string leaked from a test mock, e.g. "<MagicMock ...>"
        return "mock", model
    if lowered.startswith("claude"):
        return "anthropic", model
    if lowered.startswith("gemini"):
        return "gemini", model
    if lowered.startswith("deepseek"):
        return "deepseek", model
    if lowered.startswith("mock"):
        return "mock", model
    if lowered == "local":
        return "local", model
    if model in _KNOWN_GROQ_MODELS:
        return "groq", model
    return "ollama", model


def get_model_usage(lookback_hours: float = 24 * 7) -> list[ModelUsage]:
    """Per (tool, provider, model) call counts and failure rates -- the "which
    providers are we actually using" view. A FallbackProvider failure is its
    own row here (the primary's real model, success=False), distinct from
    the fallback's own successful row, so a model swap's real cost/reliability
    is directly comparable rather than hidden inside one merged number.
    A raw model string with and without a provider prefix (e.g. "phi4-mini"
    vs "ollama:phi4-mini") collapses to the same row here, since they're the
    same logical model. Never raises: a lock conflict or missing DB yields []."""
    db_path = get_tracking_db_path()
    if not db_path.exists():
        return []

    cutoff = datetime.now() - timedelta(hours=lookback_hours)  # noqa: DTZ005 - must stay naive to match processing_log's naive CURRENT_TIMESTAMP column
    try:
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(_MODEL_USAGE_QUERY, [cutoff]).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - best-effort read against a DB other tools may be writing to concurrently
        return []

    # A gateway-routed call produces two rows for the same logical call: the
    # calling tool's own timed_run() row (provider usually NULL) and
    # llm-gateway-service's own row for the same request (via_gateway=True,
    # provider populated for real). There's no shared id to pair them up
    # directly, but every (tool_name, model) pair that has a via_gateway row
    # is -- in this fleet, where LLM_GATEWAY_URL is set globally -- exactly
    # a pair where the NULL-provider rows are that same pair's redundant
    # client-side echoes, not independent calls. Drop those so the gateway's
    # one accurate row does the counting instead of doubling it.
    gateway_pairs = {(tool_name, raw_model) for tool_name, raw_model, _, via_gateway, _, _ in rows if via_gateway}

    merged: dict[tuple[str, str, str], ModelUsage] = {}
    for tool_name, raw_model, real_provider, via_gateway, total, failures in rows:
        if not via_gateway and real_provider is None and (tool_name, raw_model) in gateway_pairs:
            continue
        if real_provider:
            provider, model = real_provider, (raw_model or "(unset)")
        else:
            provider, model = classify_provider(raw_model)
        key = (tool_name, provider, model)
        entry = merged.setdefault(key, ModelUsage(tool_name=tool_name, model=model, provider=provider, total=0, failures=0))
        entry.total += total
        entry.failures += failures or 0
        entry.via_gateway = entry.via_gateway or bool(via_gateway)

    return sorted(merged.values(), key=lambda u: u.total, reverse=True)


def get_provider_usage(lookback_hours: float = 24 * 7) -> list[ProviderUsage]:
    """Fleet-wide rollup of get_model_usage() by provider alone -- for
    comparing providers directly (cost/reliability trials) rather than
    reading it back out of a long per-tool table. Never raises (delegates to
    get_model_usage(), which already never raises)."""
    by_provider: dict[str, dict] = {}
    for u in get_model_usage(lookback_hours=lookback_hours):
        entry = by_provider.setdefault(u.provider, {"total": 0, "failures": 0, "tools": set(), "models": set()})
        entry["total"] += u.total
        entry["failures"] += u.failures
        entry["tools"].add(u.tool_name)
        entry["models"].add(u.model)

    usage = [
        ProviderUsage(
            provider=provider,
            total=d["total"],
            failures=d["failures"],
            tool_count=len(d["tools"]),
            model_count=len(d["models"]),
        )
        for provider, d in by_provider.items()
    ]
    return sorted(usage, key=lambda p: p.total, reverse=True)


@dataclass
class FetchUsage:
    tool_name: str
    domain: str
    total: int
    failures: int
    avg_duration_ms: float | None = None
    last_call: str | None = None

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


_FETCH_USAGE_QUERY = """
    SELECT t.name AS tool_name, fl.domain AS domain,
           COUNT(*) AS total, SUM(CASE WHEN NOT fl.success THEN 1 ELSE 0 END) AS failures,
           AVG(fl.duration_ms) AS avg_duration_ms, MAX(fl.attempted_at) AS last_call
    FROM fetch_log fl JOIN tools t ON fl.tool_id = t.id
    WHERE fl.attempted_at > ?
    GROUP BY t.name, fl.domain
"""


def get_fetch_usage(lookback_hours: float = 24 * 7) -> list[FetchUsage]:
    """Jamal 2026-09-21: raw HTTP fetches (not LLM calls) read as "49 calls"
    on the old merged Tool Activity total with nothing to tell them apart --
    this is the detail that was missing: per (tool, domain), so "49" becomes
    "12 to arxiv.org, 30 to substackcdn.com, ...". Never raises: a lock
    conflict or missing DB yields []."""
    db_path = get_tracking_db_path()
    if not db_path.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=lookback_hours)  # noqa: DTZ005 - must stay naive to match fetch_log's naive attempted_at column
    try:
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(_FETCH_USAGE_QUERY, [cutoff]).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - best-effort read against a DB other tools may be writing to concurrently
        return []

    usage = [
        FetchUsage(tool_name=tool_name, domain=domain or "(unknown)", total=total, failures=failures or 0, avg_duration_ms=avg_duration_ms, last_call=str(last_call) if last_call else None)
        for tool_name, domain, total, failures, avg_duration_ms, last_call in rows
    ]
    return sorted(usage, key=lambda u: u.total, reverse=True)


@dataclass
class ApiCallUsage:
    tool_name: str
    service: str
    operation: str
    total: int
    failures: int
    last_call: str | None = None

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


_API_CALL_USAGE_QUERY = """
    SELECT t.name AS tool_name, acl.service AS service, acl.operation AS operation,
           COUNT(*) AS total, SUM(CASE WHEN NOT acl.success THEN 1 ELSE 0 END) AS failures,
           MAX(acl.attempted_at) AS last_call
    FROM api_call_log acl JOIN tools t ON acl.tool_id = t.id
    WHERE acl.attempted_at > ?
    GROUP BY t.name, acl.service, acl.operation
"""


def get_api_call_usage(lookback_hours: float = 24 * 7) -> list[ApiCallUsage]:
    """Per (tool, service, operation) call counts -- the third kind of
    external call a tool can make, distinct from an LLM completion or a
    generic URL fetch: a call to a specific third-party service's own API
    (Readwise, Mastodon, Bluesky). Never raises: a lock conflict or missing
    DB yields []."""
    db_path = get_tracking_db_path()
    if not db_path.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=lookback_hours)  # noqa: DTZ005 - must stay naive to match api_call_log's naive attempted_at column
    try:
        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(_API_CALL_USAGE_QUERY, [cutoff]).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - best-effort read against a DB other tools may be writing to concurrently
        return []

    usage = [
        ApiCallUsage(tool_name=tool_name, service=service, operation=operation, total=total, failures=failures or 0, last_call=str(last_call) if last_call else None)
        for tool_name, service, operation, total, failures, last_call in rows
    ]
    return sorted(usage, key=lambda u: u.total, reverse=True)


def _launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _is_keep_alive(label: str) -> bool:
    path = _launch_agents_dir() / f"{label}.plist"
    if not path.exists():
        return False
    try:
        with path.open("rb") as f:
            data = plistlib.load(f)
    except Exception:  # noqa: BLE001 - a malformed/unreadable plist should not be treated as KeepAlive
        return False
    keep_alive = data.get("KeepAlive")
    if isinstance(keep_alive, dict):
        return True
    return bool(keep_alive)


@dataclass
class ServiceStatus:
    label: str
    running: bool
    pid: int | None
    keep_alive: bool
    last_exit_code: int | None


def get_launch_agents() -> list[ServiceStatus]:
    """Every com.localfirst.*/com.jamalhansen.* LaunchAgent's current state,
    via `launchctl list` (no sudo, no privileged calls). Never raises."""
    try:
        result = subprocess.run(["launchctl", "list"], capture_output=True, text=True, check=True, timeout=5)
    except Exception:  # noqa: BLE001 - best-effort; a launchctl hiccup shouldn't break the page
        return []

    statuses = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        pid_str, status_str, label = parts
        if not label.startswith(LOCALFIRST_PREFIXES):
            continue
        pid = None if pid_str == "-" else int(pid_str)
        try:
            exit_code = int(status_str)
        except ValueError:
            exit_code = None
        statuses.append(
            ServiceStatus(
                label=label,
                running=pid is not None,
                pid=pid,
                keep_alive=_is_keep_alive(label),
                last_exit_code=exit_code,
            )
        )
    return sorted(statuses, key=lambda s: s.label)


# ---------------------------------------------------------------------------
# Vault tensions (Contexta's ops/tensions/, via tension-triage-dashboard)
# ---------------------------------------------------------------------------


@dataclass
class TensionSummary:
    pending_count: int
    active_count: int
    recent_titles: list[str]


@dataclass
class VaultHealth:
    observations_pending: int
    inbox_count: int
    inbox_oldest_days: float | None
    last_health_check: str | None


def get_vault_health(vault_path: str | Path) -> VaultHealth:
    """Mirrors the maintenance thresholds CLAUDE.md already tracks (10+
    observations, 3+ day inbox age, 7+ day stale health check) -- visible
    from the dashboard without opening a session. Never raises: a missing
    directory or malformed note is skipped, not fatal."""
    vault = Path(vault_path).expanduser()

    observations_pending = 0
    obs_dir = vault / "ops" / "observations"
    if obs_dir.exists():
        for f in obs_dir.glob("*.md"):
            try:
                post = frontmatter.load(f)
                if post.metadata.get("status") == "pending":
                    observations_pending += 1
            except Exception:  # noqa: BLE001, S112 - a malformed observation note shouldn't break the count
                continue

    inbox_count = 0
    inbox_oldest_days: float | None = None
    inbox_dir = vault / "inbox"
    if inbox_dir.exists():
        files = list(inbox_dir.glob("*.md"))
        inbox_count = len(files)
        if files:
            oldest_mtime = min(f.stat().st_mtime for f in files)
            inbox_oldest_days = (datetime.now().timestamp() - oldest_mtime) / 86400  # noqa: DTZ005 - comparing against a local mtime, not persisting

    last_health_check = None
    health_dir = vault / "ops" / "health"
    if health_dir.exists():
        reports = sorted(health_dir.glob("*-report.md"))
        if reports:
            last_health_check = reports[-1].name.removesuffix("-report.md")

    return VaultHealth(
        observations_pending=observations_pending,
        inbox_count=inbox_count,
        inbox_oldest_days=inbox_oldest_days,
        last_health_check=last_health_check,
    )


def get_tension_summary(vault_path: str | Path) -> TensionSummary:
    """Never raises: an unreadable tensions dir or a bad note is skipped,
    not fatal, matching tension_triage_dashboard's own read-only stance."""
    try:
        from tension_triage_dashboard.clustering import scan_tensions

        tensions_dir = Path(vault_path).expanduser() / "ops" / "tensions"
        tensions = scan_tensions(tensions_dir)
    except Exception:  # noqa: BLE001 - a read-only vault scan should degrade, not 500 the page
        return TensionSummary(pending_count=0, active_count=0, recent_titles=[])

    pending = [t for t in tensions if getattr(t, "status", None) == "pending"]
    active = [t for t in tensions if getattr(t, "status", None) == "active"]
    recent = sorted(tensions, key=lambda t: getattr(t, "created", "") or "", reverse=True)[:5]
    return TensionSummary(
        pending_count=len(pending),
        active_count=len(active),
        recent_titles=[getattr(t, "title", "untitled") for t in recent],
    )


# ---------------------------------------------------------------------------
# Artist-agent's latest image
# ---------------------------------------------------------------------------


@dataclass
class ArtItem:
    title: str
    self_score: float | None
    interest: str | None
    generated_at: str | None
    image_path: Path


def get_latest_art(items_dir: str | Path = "~/iCloud/ai-artist/items") -> ArtItem | None:
    """Never raises: a missing directory or malformed item note returns None."""
    try:
        items_path = Path(items_dir).expanduser()
        if not items_path.exists():
            return None
        item_files = sorted(items_path.glob("*.md"))
        if not item_files:
            return None
        latest = item_files[-1]
        post = frontmatter.load(latest)

        images_dir = items_path.parent / "images"
        image_path = images_dir / f"{latest.stem}.png"
        if not image_path.exists():
            candidates = list(images_dir.glob(f"{latest.stem}.*"))
            if not candidates:
                return None
            image_path = candidates[0]

        return ArtItem(
            title=post.metadata.get("title") or latest.stem,
            self_score=post.metadata.get("self_score"),
            interest=post.metadata.get("interest"),
            generated_at=post.metadata.get("generated_at"),
            image_path=image_path,
        )
    except Exception:  # noqa: BLE001 - best-effort; a malformed item note shouldn't break the page
        return None


# ---------------------------------------------------------------------------
# japanese-tutor (calls its own already-running API, doesn't touch its DB)
# ---------------------------------------------------------------------------


@dataclass
class JapaneseTutorSummary:
    reachable: bool
    cards_due: int = 0
    new_count: int = 0
    review_count: int = 0
    mastery: list[dict] = field(default_factory=list)
    reviews_today_attempts: int = 0
    reviews_today_distinct_cards: int = 0


def get_japanese_tutor_summary(base_url: str = "http://127.0.0.1:8421") -> JapaneseTutorSummary:
    """Never raises: the server not being up is a normal, expected state.

    Reads /api/cards/due/count, not len(/api/cards/due) -- found live
    2026-09-20: the latter always returns up to a fixed limit (backfilled
    with not-yet-due cards to keep a study session full), so its length
    never reflects real review progress. The count endpoint is the true
    number of cards overdue right now.

    Splits cards_due into new_count/review_count -- also found live
    2026-09-20: a flat due count reads as "overdue re-reviews" but can be
    entirely never-touched cards, which a learner should treat differently.

    Also reads /api/reviews/today -- due_count alone couldn't explain why
    reviewing ~15 cards barely moved it; attempts (every submission,
    including retries) vs distinct_cards (unique characters touched) is
    the real answer, and the two together show it.
    """
    try:
        with httpx.Client(timeout=3.0) as client:
            due = client.get(f"{base_url}/api/cards/due/count").json()
            mastery = client.get(f"{base_url}/api/mastery").json()
            reviews_today = client.get(f"{base_url}/api/reviews/today").json()
        return JapaneseTutorSummary(
            reachable=True,
            cards_due=due.get("due_count", 0),
            new_count=due.get("new_count", 0),
            review_count=due.get("review_count", 0),
            mastery=mastery,
            reviews_today_attempts=reviews_today.get("attempts", 0),
            reviews_today_distinct_cards=reviews_today.get("distinct_cards", 0),
        )
    except Exception:  # noqa: BLE001 - the server simply not running is expected, not an error to surface
        return JapaneseTutorSummary(reachable=False)


# ---------------------------------------------------------------------------
# Frontmatter validation (personal-infra's frontmatter-validation-run script,
# not yet on a schedule as of 2026-09-20 pending two real findings: the
# validator's own Category-field check is case-sensitive against content
# that's genuinely lowercase, and blog/posts/ turned out to be the wrong
# target -- see BrainSync tool doc for frontmatter-validator)
# ---------------------------------------------------------------------------


@dataclass
class FrontmatterValidationSummary:
    available: bool
    generated_at: str | None = None
    total: int = 0
    invalid_count: int = 0
    invalid_files: list[dict] = field(default_factory=list)
    error_summary: list[dict] = field(default_factory=list)
    content_todo: list[dict] = field(default_factory=list)


_ERROR_SUMMARY_FILES_CAP = 20  # per error type -- some real groups run 100+ files, showing all isn't "a summary"
_TODO_BUCKET_FILES_CAP = 20

# Ordered most-urgent first: a file already live with incomplete metadata
# matters more than one that's still a draft. A file can land in more than
# one bucket (e.g. missing both status and tags) -- these aren't mutually
# exclusive categories, they're independent things that need doing.
_TODO_BUCKETS: list[tuple[str, str, Callable[[str], bool]]] = [
    (
        "published_incomplete",
        "Published but incomplete",
        lambda err: "'status' is 'published'" in err,
    ),
    (
        "needs_status",
        "Needs a status decision",
        lambda err: "Missing universal field: 'status'" in err,
    ),
    (
        "needs_category",
        "Needs a category decision",
        lambda err: "Missing 'category' field" in err,
    ),
    (
        "parse_error",
        "Malformed YAML -- needs a manual fix",
        lambda err: err.startswith("Failed to parse frontmatter"),
    ),
    (
        "auto_fixable",
        "Auto-fixable (tags/created)",
        lambda err: "Missing universal field: 'tags'" in err or "Missing universal field: 'created'" in err,
    ),
]


def get_frontmatter_validation(
    snapshot_path: str | Path = "~/sync/local-first/frontmatter-validation-latest.json",
) -> FrontmatterValidationSummary:
    """Reads the snapshot frontmatter-validation-run writes -- never raises:
    the job not having run yet (or ever) is a normal, expected state.

    error_summary groups every invalid file by its exact error text (one
    file can appear under more than one error) so issues can be addressed
    together -- "170 files are missing canonical_url" is actionable in a way
    a 256-row flat file list isn't. Built from the full invalid_files list in
    the snapshot, not the truncated one this function also returns for the
    per-file detail view.
    """
    try:
        path = Path(snapshot_path).expanduser()
        if not path.exists():
            return FrontmatterValidationSummary(available=False)
        data = json.loads(path.read_text())
        all_invalid = data.get("invalid_files", [])

        by_error: dict[str, list[str]] = {}
        for entry in all_invalid:
            for err in entry.get("errors", []):
                by_error.setdefault(err, []).append(entry.get("file", ""))

        error_summary = [
            {
                "error": err,
                "count": len(files),
                "files": files[:_ERROR_SUMMARY_FILES_CAP],
                "more": max(0, len(files) - _ERROR_SUMMARY_FILES_CAP),
            }
            for err, files in sorted(by_error.items(), key=lambda kv: len(kv[1]), reverse=True)
        ]

        # Re-slice the same data by what kind of action it needs, not just
        # raw error text -- "23 files need a status decision" is something
        # to go do, "Missing universal field: 'status'" is just a string. A
        # file lands in every bucket its errors match (not mutually exclusive:
        # a file can need both a status decision and a tags/created auto-fill).
        bucket_files: dict[str, list[str]] = {key: [] for key, _, _ in _TODO_BUCKETS}
        for entry in all_invalid:
            errors = entry.get("errors", [])
            file_name = entry.get("file", "")
            for key, _, matches in _TODO_BUCKETS:
                if any(matches(err) for err in errors):
                    bucket_files[key].append(file_name)

        content_todo = [
            {
                "key": key,
                "label": label,
                "count": len(bucket_files[key]),
                "files": bucket_files[key][:_TODO_BUCKET_FILES_CAP],
                "more": max(0, len(bucket_files[key]) - _TODO_BUCKET_FILES_CAP),
            }
            for key, label, _ in _TODO_BUCKETS
            if bucket_files[key]
        ]

        return FrontmatterValidationSummary(
            available=True,
            generated_at=data.get("generated_at"),
            total=data.get("total", 0),
            invalid_count=data.get("invalid_count", 0),
            invalid_files=all_invalid[:10],
            error_summary=error_summary,
            content_todo=content_todo,
        )
    except Exception:  # noqa: BLE001 - a malformed/partial snapshot file shouldn't break the page
        return FrontmatterValidationSummary(available=False)


# ---------------------------------------------------------------------------
# Writing cadence -- the blog's publishing rhythm against the one-post-a-week
# target in BrainSync/blog/content-strategy-plan-2026.md, plus what's in the
# vault pipeline. Read live from the files, like vault health: no job needed.
# ---------------------------------------------------------------------------

_WEEKS_SHOWN = 8
# Statuses that mean "this could become a post": outline counts because the
# plan's bottleneck is outline -> draft, so outlines are the queue to watch.
_PIPELINE_STATUSES = ("draft", "outline", "idea", "brainstorm")


@dataclass
class WritingCadence:
    available: bool
    last_published: str | None = None
    last_published_title: str | None = None
    days_since_last: int | None = None
    weeks: list[dict] = field(default_factory=list)  # oldest first: {"week_of", "posts"}
    weeks_on_target: int = 0
    pipeline: dict[str, int] = field(default_factory=dict)
    freshest_draft: dict | None = None


def _as_date(value) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "year") and hasattr(value, "month"):  # datetime.date
        return datetime(value.year, value.month, value.day)  # noqa: DTZ001 - frontmatter dates are naive calendar dates
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def get_writing_cadence(
    blog_dir: str | Path = "~/projects/jamalhansen.com/content/blog",
    vault_blog_dir: str | Path = "~/vaults/BrainSync/blog",
    today: datetime | None = None,
) -> WritingCadence:
    """Never raises: a missing blog checkout or vault degrades this card only."""
    try:
        today = (today or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)  # noqa: DTZ005 - compared against naive frontmatter dates
        blog = Path(blog_dir).expanduser()
        if not blog.exists():
            return WritingCadence(available=False)

        published: list[tuple[datetime, str]] = []
        for index in blog.rglob("index.md"):
            try:
                meta = frontmatter.load(index).metadata
            except Exception:  # noqa: BLE001, S112 - one malformed post shouldn't hide the rest
                continue
            posted = _as_date(meta.get("date"))
            if meta.get("draft") is True or posted is None or posted > today:
                continue
            published.append((posted, str(meta.get("title") or index.parent.name)))
        published.sort()

        week_start = today - timedelta(days=today.weekday())  # Monday
        weeks = []
        for i in range(_WEEKS_SHOWN - 1, -1, -1):
            start = week_start - timedelta(weeks=i)
            end = start + timedelta(days=7)
            weeks.append({
                "week_of": start.date().isoformat(),
                "posts": sum(1 for d, _ in published if start <= d < end),
            })

        pipeline = dict.fromkeys(_PIPELINE_STATUSES, 0)
        freshest: tuple[float, Path, str] | None = None
        vault = Path(vault_blog_dir).expanduser()
        if vault.exists():
            for note in vault.rglob("*.md"):
                rel = note.relative_to(vault).parts
                if "posts" not in rel or note.name == "promo.md":
                    continue
                try:
                    status = str(frontmatter.load(note).metadata.get("status") or "")
                except Exception:  # noqa: BLE001, S112 - hand-edited frontmatter fails to parse in many ways
                    continue
                if status in pipeline:
                    pipeline[status] += 1
                    mtime = note.stat().st_mtime
                    if status == "draft" and (freshest is None or mtime > freshest[0]):
                        freshest = (mtime, note, status)

        last_date, last_title = published[-1] if published else (None, None)
        return WritingCadence(
            available=True,
            last_published=last_date.date().isoformat() if last_date else None,
            last_published_title=last_title,
            days_since_last=(today - last_date).days if last_date else None,
            weeks=weeks,
            weeks_on_target=sum(1 for w in weeks if w["posts"] >= 1),
            pipeline=pipeline,
            freshest_draft=(
                {
                    "name": freshest[1].stem,
                    "modified": datetime.fromtimestamp(freshest[0]).date().isoformat(),  # noqa: DTZ006 - local mtime shown as a local date
                }
                if freshest
                else None
            ),
        )
    except Exception:  # noqa: BLE001 - a read-only scan should degrade, not 500 the page
        return WritingCadence(available=False)


# ---------------------------------------------------------------------------
# Code-block validation (personal-infra's blog-validate-run script, weekly
# via com.localfirst.blog-validation) -- do the code samples embedded in
# posts still run? One row per target: the published blog, the vault drafts
# they were written in, and the newsletter patterns.
# ---------------------------------------------------------------------------

_FAILED_POSTS_CAP = 20


@dataclass
class BlogValidationSummary:
    available: bool
    generated_at: str | None = None
    targets: list[dict] = field(default_factory=list)
    posts: int = 0
    posts_failed: int = 0
    blocks_failed: int = 0


def get_blog_validation(
    snapshot_path: str | Path = "~/sync/local-first/blog-validate-latest.json",
) -> BlogValidationSummary:
    """Reads the snapshot blog-validate-run writes -- never raises: the job
    not having run yet (or ever) is a normal, expected state."""
    try:
        path = Path(snapshot_path).expanduser()
        if not path.exists():
            return BlogValidationSummary(available=False)
        data = json.loads(path.read_text())

        targets = []
        for t in data.get("targets", []):
            summary = t.get("summary", {})
            coverage = t.get("coverage", {})
            failed_posts = t.get("failed_posts", [])
            targets.append(
                {
                    "name": t.get("name", ""),
                    "posts": summary.get("posts", 0),
                    "posts_passed": summary.get("posts_passed", 0),
                    "posts_failed": summary.get("posts_failed", 0),
                    "blocks_passed": summary.get("blocks_passed", 0),
                    "blocks_failed": summary.get("blocks_failed", 0),
                    "blocks_skipped": summary.get("blocks_skipped", 0),
                    "posts_skipped_by_status": summary.get("posts_skipped_by_status", 0),
                    "fully_covered": coverage.get("fully_covered", 0),
                    "needs_attention": coverage.get("needs_attention", 0),
                    "assertion_pct": coverage.get("assertion_pct", 0),
                    "failed_posts": failed_posts[:_FAILED_POSTS_CAP],
                    "more_failed": max(0, len(failed_posts) - _FAILED_POSTS_CAP),
                }
            )

        return BlogValidationSummary(
            available=True,
            generated_at=data.get("generated_at"),
            targets=targets,
            posts=sum(t["posts"] for t in targets),
            posts_failed=sum(t["posts_failed"] for t in targets),
            blocks_failed=sum(t["blocks_failed"] for t in targets),
        )
    except Exception:  # noqa: BLE001 - a malformed/partial snapshot file shouldn't break the page
        return BlogValidationSummary(available=False)


# ---------------------------------------------------------------------------
# Repo health (personal-infra's repo-health-run script, daily via
# com.localfirst.repo-health) -- per-repo lint/tests/git/hooks status across
# the fleet, so a stalled or broken repo surfaces without running `make
# verify` by hand and reading the log.
# ---------------------------------------------------------------------------


@dataclass
class RepoHealthSummary:
    available: bool
    generated_at: str | None = None
    total: int = 0
    healthy: int = 0
    repos: list[dict] = field(default_factory=list)


def get_repo_health(
    snapshot_path: str | Path = "~/sync/local-first/repo-health-latest.json",
) -> RepoHealthSummary:
    """Reads the snapshot repo-health-run writes -- never raises: the job
    not having run yet (or ever) is a normal, expected state.

    repos is sorted unhealthy-first so the card surfaces what needs
    attention without the viewer scanning a full alphabetical list.
    """
    try:
        path = Path(snapshot_path).expanduser()
        if not path.exists():
            return RepoHealthSummary(available=False)
        data = json.loads(path.read_text())

        repos = []
        for name, r in data.get("repos", {}).items():
            lint = r.get("lint", {})
            tests = r.get("tests", {})
            git = r.get("git", {})
            hooks = r.get("hooks", {})
            install = r.get("install", {})
            ok = (
                bool(lint.get("ok"))
                and bool(tests.get("ok"))
                and bool(hooks.get("ok"))
                and bool(git.get("has_remote", True))
                and bool(install.get("ok", True))
            )
            repos.append(
                {
                    "name": name,
                    "ok": ok,
                    "lint_ok": lint.get("ok", True),
                    "lint_errors": lint.get("error_count", 0),
                    "tests_ok": tests.get("ok", True),
                    "tests_passed": tests.get("passed", 0),
                    "tests_failed": tests.get("failed", 0),
                    "hooks_ok": hooks.get("ok", True),
                    "dirty": git.get("dirty", False),
                    "unpushed": git.get("unpushed", 0),
                    "has_remote": git.get("has_remote", True),
                    "install_ok": install.get("ok", True),
                    "install_stale_files": install.get("stale_files", 0),
                }
            )
        repos.sort(key=lambda r: (r["ok"], r["name"]))

        return RepoHealthSummary(
            available=True,
            generated_at=data.get("generated_at"),
            total=data.get("total", 0),
            healthy=data.get("healthy", 0),
            repos=repos,
        )
    except Exception:  # noqa: BLE001 - a malformed/partial snapshot file shouldn't break the page
        return RepoHealthSummary(available=False)


@dataclass
class GatewayRoutingEntry:
    tool_name: str
    category: str  # "gateway" | "direct_pydantic_ai" | "direct_forced" | "direct_unclassified" | "none"
    status: str  # "ok" | "unconfirmed" | "review" | "expected_direct" | "no_llm_calls"
    last_call: str | None = None


@dataclass
class GatewayRoutingSummary:
    available: bool
    generated_at: str | None = None
    entries: list[GatewayRoutingEntry] = field(default_factory=list)


def get_gateway_routing_audit(
    snapshot_path: str | Path = "~/sync/local-first/repo-health-latest.json",
    lookback_hours: float = 24 * 30,
) -> GatewayRoutingSummary:
    """Cross-references repo-health-run's static source classification
    (does this repo's code call resolve_provider(), a pydantic-ai Agent, or
    construct a provider directly?) against real processing_log activity, so
    a repo that LOOKS gateway-routed in its own source but has never actually
    logged a via_gateway=True row shows up distinctly from one that's simply
    never been run. Jamal 2026-09-21: "I would like to see if something isn't
    using the gateway that should be" -- the DB alone can't answer that (it
    has no notion of intent), so this needs the static category too.

    Status meanings:
    - "ok": source calls resolve_provider() and has a confirmed via_gateway
      row in the lookback window.
    - "unconfirmed": source calls resolve_provider() but no via_gateway row
      shows up -- either it hasn't run recently, or something's wrong
      (LLM_GATEWAY_URL unset, gateway down when it last ran).
    - "review": constructs a provider directly with no documented reason --
      gets none of the gateway's fallback/cost/reliability tracking; a real
      migration candidate.
    - "expected_direct": either pydantic-ai Agent(retries=N) for auto-retry
      on invalid structured output, a capability resolve_provider() doesn't
      have (persona-counsel, marketing-persona-counsel, pedantic-troll), or
      resolve_provider(..., use_gateway=False) for a deliberate, documented
      reason (pebble, 2026-09-22: baby photos/journal text must stay local,
      and the gateway's own internal resolve_provider() call always defaults
      to fallback=True with no way to override it). Either way: deliberate,
      not a gap.
    - "no_llm_calls": no .complete()/.acomplete() found in source.

    Never raises: a missing/malformed snapshot or unreachable tracking DB
    yields an empty, unavailable summary rather than breaking the page.
    """
    try:
        path = Path(snapshot_path).expanduser()
        if not path.exists():
            return GatewayRoutingSummary(available=False)
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - a malformed/partial snapshot file shouldn't break the page
        return GatewayRoutingSummary(available=False)

    usage = get_model_usage(lookback_hours=lookback_hours)
    confirmed_gateway = {u.tool_name for u in usage if u.via_gateway}
    last_call_by_tool = {a.tool_name: a.last_call for a in get_fleet_activity(lookback_hours=lookback_hours)}

    entries = []
    for name, r in data.get("repos", {}).items():
        category = r.get("gateway_routing", {}).get("category", "none")
        if category == "gateway":
            status = "ok" if name in confirmed_gateway else "unconfirmed"
        elif category == "direct_unclassified":
            status = "review"
        elif category in ("direct_pydantic_ai", "direct_forced"):
            status = "expected_direct"
        else:
            status = "no_llm_calls"
        entries.append(GatewayRoutingEntry(tool_name=name, category=category, status=status, last_call=last_call_by_tool.get(name)))

    # Surface what needs a look first: review > unconfirmed > ok > expected_direct > no_llm_calls.
    order = {"review": 0, "unconfirmed": 1, "ok": 2, "expected_direct": 3, "no_llm_calls": 4}
    entries.sort(key=lambda e: (order.get(e.status, 9), e.tool_name))

    return GatewayRoutingSummary(available=True, generated_at=data.get("generated_at"), entries=entries)
