import io
import urllib.request
import zipfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[1]
_EXP_CFG_PATH = _ROOT / "configs" / "experiments.yaml"


def _load_cfg() -> tuple[str, Path]:
    with open(_EXP_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    ns = cfg["datasets"]["natural_stories"]
    return ns["url"], _ROOT / ns["path"]


def load_natural_stories():
    url, out_dir = _load_cfg()

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Already exists: {out_dir}")
        return

    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"Downloaded {len(data) / 1024:.1f} KB")

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
        zf.extractall(out_dir)

    print(f"Extracted {len(members)} files -> {out_dir}")
    for p in sorted(out_dir.rglob("*"))[:30]:
        print(f"  {p.relative_to(out_dir)}")
