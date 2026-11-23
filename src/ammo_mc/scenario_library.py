"""SQLite-backed Persistent Scenario Library (Save, Load, Tag, Diff & JSON Sync)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import uuid


def resolve_scenario_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve writable SQLite path with safe fallback to system temp directory."""
    if db_path is not None:
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    env_path = os.environ.get("AMMO_MC_SCENARIO_DB", "").strip()
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    home_dir = Path.home() / ".ammo_mc"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        probe = home_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return home_dir / "scenarios.db"
    except Exception:
        return Path(tempfile.gettempdir()) / "ammo_mc_scenarios.db"


def _flatten_dict(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, prefix=key))
        else:
            flat[key] = v
    return flat


class ScenarioLibrary:
    """Thread-safe SQLite store for saving, reloading, diffing, and backing up scenarios."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = resolve_scenario_db_path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "domain": str(row["domain"]),
            "tags": json.loads(row["tags_json"]),
            "notes": str(row["notes"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]),
        }

    def save_scenario(
        self,
        name: str,
        domain: str,
        config: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        scenario_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a named scenario snapshot and return its stored record."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sid = str(scenario_id).strip() if scenario_id else uuid.uuid4().hex[:10]
        clean_tags = sorted({str(t).strip() for t in (tags or []) if str(t).strip()})
        cfg_str = json.dumps(config, sort_keys=True, default=str)
        met_str = json.dumps(metrics or {}, sort_keys=True, default=str)
        tags_str = json.dumps(clean_tags)

        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM scenarios WHERE id = ?", (sid,)).fetchone()
            created = str(existing["created_at"]) if existing else now_iso
            conn.execute(
                """
                INSERT INTO scenarios (id, name, domain, tags_json, notes, created_at, updated_at, config_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    domain = excluded.domain,
                    tags_json = excluded.tags_json,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at,
                    config_json = excluded.config_json,
                    metrics_json = excluded.metrics_json
                """,
                (sid, str(name).strip() or f"Scenario-{sid}", str(domain).strip() or "general", tags_str, str(notes), created, now_iso, cfg_str, met_str),
            )
            conn.commit()
            saved = conn.execute("SELECT * FROM scenarios WHERE id = ?", (sid,)).fetchone()
        assert saved is not None
        return self._row_to_dict(saved)

    def get_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scenarios WHERE id = ?", (scenario_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_scenarios(
        self,
        domain: str | None = None,
        tag: str | None = None,
        search: str = "",
    ) -> list[dict[str, Any]]:
        """List stored scenarios filtered by optional domain, tag, and substring query."""
        with self._connect() as conn:
            if domain and domain.lower() != "all":
                rows = conn.execute(
                    "SELECT * FROM scenarios WHERE lower(domain) = ? ORDER BY updated_at DESC",
                    (domain.strip().lower(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scenarios ORDER BY updated_at DESC").fetchall()

        q = search.strip().lower()
        t_filter = tag.strip().lower() if tag and tag.lower() != "all" else None
        out: list[dict[str, Any]] = []
        for r in rows:
            item = self._row_to_dict(r)
            if t_filter and t_filter not in [tg.lower() for tg in item["tags"]]:
                continue
            if q:
                hay = f"{item['id']} {item['name']} {item['domain']} {item['notes']} {' '.join(item['tags'])}".lower()
                if q not in hay:
                    continue
            out.append(item)
        return out

    def delete_scenario(self, scenario_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM scenarios WHERE id = ?", (scenario_id,))
            conn.commit()
            return cur.rowcount > 0

    def diff_scenarios(self, id_a: str, id_b: str) -> dict[str, Any]:
        """Compare configuration parameters and quantitative outcome metrics between two scenarios."""
        sa = self.get_scenario(id_a)
        sb = self.get_scenario(id_b)
        if sa is None or sb is None:
            raise KeyError("Both scenario IDs must exist in ScenarioLibrary to diff.")

        cfg_a = _flatten_dict(sa["config"])
        cfg_b = _flatten_dict(sb["config"])
        met_a = _flatten_dict(sa["metrics"])
        met_b = _flatten_dict(sb["metrics"])

        config_diffs: list[dict[str, Any]] = []
        for k in sorted(set(cfg_a) | set(cfg_b)):
            va, vb = cfg_a.get(k), cfg_b.get(k)
            if va != vb:
                delta = None
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
                    delta = round(float(vb) - float(va), 4)
                config_diffs.append({"parameter": k, "scenario_a": va, "scenario_b": vb, "delta": delta})

        metric_diffs: list[dict[str, Any]] = []
        for k in sorted(set(met_a) | set(met_b)):
            va, vb = met_a.get(k), met_b.get(k)
            delta = None
            pct = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
                d_val = float(vb) - float(va)
                delta = round(d_val, 4)
                if abs(float(va)) > 1e-9:
                    pct = round(100.0 * d_val / float(va), 2)
            metric_diffs.append({"metric": k, "scenario_a": va, "scenario_b": vb, "delta": delta, "pct_change": pct})

        return {
            "scenario_a": {"id": sa["id"], "name": sa["name"], "domain": sa["domain"]},
            "scenario_b": {"id": sb["id"], "name": sb["name"], "domain": sb["domain"]},
            "config_diffs": config_diffs,
            "metric_diffs": metric_diffs,
        }

    def export_library_json(self, domain: str | None = None) -> str:
        items = self.list_scenarios(domain=domain)
        return json.dumps({"schema": "ammo_mc_scenario_library_v1", "scenarios": items}, indent=2)

    def import_library_json(self, raw_json: str) -> int:
        payload = json.loads(raw_json)
        items = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("Scenario library JSON must contain a list of scenarios.")
        imported = 0
        for row in items:
            if not isinstance(row, dict) or "name" not in row or "config" not in row:
                continue
            self.save_scenario(
                name=str(row["name"]),
                domain=str(row.get("domain", "general")),
                config=dict(row["config"]),
                metrics=dict(row.get("metrics", {})),
                tags=list(row.get("tags", [])),
                notes=str(row.get("notes", "")),
                scenario_id=row.get("id"),
            )
            imported += 1
        return imported
