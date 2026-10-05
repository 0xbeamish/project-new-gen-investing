"""Load KEY=VALUE lines from the repo's .env into the environment (never printed, never committed)."""

import os

from engine.config import ROOT

ENV_FILE = ROOT / ".env"


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def has(key: str) -> bool:
    return bool(os.environ.get(key))
