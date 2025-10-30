from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "backend" / "data_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


