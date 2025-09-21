"""Configuration helpers for the Fantasy Manager application."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Application level configuration loaded from environment variables."""

    odds_api_key: Optional[str]
    sleeper_base_url: str = "https://api.sleeper.app/v1"
    fantasypros_base_url: str = "https://www.fantasypros.com/nfl/projections"

    @classmethod
    def load(cls, env_file: Optional[Path] = None) -> "AppConfig":
        if env_file is None:
            env_file = Path(".env")
        load_dotenv(env_file)
        from os import getenv

        return cls(
            odds_api_key=getenv("ODDS_API_KEY"),
        )


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_metadata(path: Path, metadata: dict) -> None:
    payload = {
        **metadata,
        "last_updated": metadata.get("last_updated"),
        "last_window": metadata.get("last_window"),
        "generated_at": datetime.utcnow().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
