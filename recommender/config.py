import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_KEYS_PATH = _ROOT / "keys.md"


def _load_keys() -> dict:
    if not _KEYS_PATH.exists():
        return {}
    return dict(
        line.strip().split("=", 1)
        for line in _KEYS_PATH.read_text().splitlines()
        if "=" in line
    )


_keys = _load_keys()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_KEY") or _keys.get("ANTHROPIC_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API") or _keys.get("BRAVE_API")

if not ANTHROPIC_API_KEY or not BRAVE_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_KEY / BRAVE_API missing. Set them as Fly secrets "
        "(fly secrets set ANTHROPIC_KEY=... BRAVE_API=...) or, for local dev, "
        "put them in keys.md as ANTHROPIC_KEY=... / BRAVE_API=..."
    )

MODEL = "claude-sonnet-5-5"
INPUT_COST_PER_MTOK = 2.0
OUTPUT_COST_PER_MTOK = 10.0
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = _ROOT / "data" / "events.db"
