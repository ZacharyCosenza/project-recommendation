import os
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
_settings = yaml.safe_load(Path(os.environ.get("CONFIG_PATH", ROOT / "config.yaml")).read_text())

paths = SimpleNamespace(**_settings["paths"])
search = SimpleNamespace(**_settings["search"])
model = SimpleNamespace(**_settings["model"])
event_map = SimpleNamespace(**_settings["event_map"])
app = SimpleNamespace(**_settings["app"])
auth = SimpleNamespace(**_settings["auth"])
server = SimpleNamespace(**_settings["server"])

DB_PATH = ROOT / paths.db
HEARTBEAT_PATH = Path(paths.heartbeat)


def _load_keys():
    keys_path = ROOT / "keys.md"
    if not keys_path.exists():
        return {}
    return dict(line.strip().split("=", 1) for line in keys_path.read_text().splitlines() if "=" in line)


_keys = _load_keys()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_KEY") or _keys.get("ANTHROPIC_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API") or _keys.get("BRAVE_API")
# Optional: without it the site is read-only for everyone, including you.
APP_PASSWORD = os.environ.get("APP_PASSWORD") or _keys.get("APP_PASSWORD")

if not ANTHROPIC_API_KEY or not BRAVE_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_KEY / BRAVE_API missing. Set them as Fly secrets "
        "(fly secrets set ANTHROPIC_KEY=... BRAVE_API=...) or, for local dev, "
        "put them in keys.md as ANTHROPIC_KEY=... / BRAVE_API=..."
    )
