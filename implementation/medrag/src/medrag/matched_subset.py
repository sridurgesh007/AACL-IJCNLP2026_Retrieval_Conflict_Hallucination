from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_label_audit(path: str | Path) -> dict[str, str]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out: dict[str, str] = {}
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("item_id", "")).strip()
            status = str(row.get("label_status", "")).strip().lower()
            if item_id and status in {"clean", "ambiguous", "likely_wrong"}:
                out[item_id] = status
    elif isinstance(payload, dict):
        rows = payload.get("items", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item_id = str(row.get("item_id", "")).strip()
                status = str(row.get("label_status", "")).strip().lower()
                if item_id and status in {"clean", "ambiguous", "likely_wrong"}:
                    out[item_id] = status
    return out


def build_label_audit_template(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": rows}
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_matched_subset_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    included_ids: list[str] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for item in items:
        item_id = str(item.get("item_id", "")).strip()
        reasons = [str(x) for x in item.get("exclude_reasons", []) if str(x).strip()]
        if reasons:
            excluded.append({"item_id": item_id, "reasons": reasons})
            for reason in reasons:
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        elif item_id:
            included_ids.append(item_id)
    return {
        "n_items_total": int(len(items)),
        "n_items_included": int(len(included_ids)),
        "n_items_excluded": int(len(excluded)),
        "matched_item_ids": sorted(included_ids),
        "excluded_items": excluded,
        "exclude_reason_counts": reason_counts,
    }


def clean_subset_item_ids(matched_payload: dict[str, Any], label_audit: dict[str, str]) -> list[str]:
    matched_ids = [str(x) for x in matched_payload.get("matched_item_ids", [])]
    return sorted([item_id for item_id in matched_ids if str(label_audit.get(item_id, "")).lower() == "clean"])
