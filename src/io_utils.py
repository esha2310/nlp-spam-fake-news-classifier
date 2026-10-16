from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
