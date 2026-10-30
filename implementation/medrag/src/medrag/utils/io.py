from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON at {p}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash(start_path: str | Path) -> str:
    cwd = Path(start_path).expanduser().resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return str(out)
    except Exception:
        return "unknown"


def build_run_metadata(
    *,
    args: Any,
    input_file: str | Path,
    seed: int,
    wall_clock_seconds: float,
    project_root: str | Path,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "git_commit_hash": git_commit_hash(project_root),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cli_args": {str(k): v for k, v in vars(args).items()},
        "input_file": str(Path(input_file).expanduser().resolve()),
        "input_file_sha256": sha256_file(input_file),
    }


def write_summary_json(
    path: str | Path,
    payload: dict[str, Any],
    *,
    rate_fields: list[str],
) -> None:
    missing: list[str] = []
    for field in rate_fields:
        if field not in payload:
            continue
        ci_key = f"{field}_ci95_wilson"
        if ci_key not in payload:
            missing.append(field)
    if missing:
        raise RuntimeError(f"Refusing to write summary without Wilson CI for: {', '.join(sorted(missing))}")
    write_json(path, payload)
