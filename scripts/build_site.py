"""Build the hosted site (web/site/) and the Windows download zip.

    python scripts/build_site.py

The landing page is the app's own landing.html; served from the hosted site it switches
to "Get HeadVoice" mode by itself. The zip holds the source, config, eye model and a
one-click launcher; the face models are downloaded on first run (size and licenses).
"""

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "headvoice" / "app" / "static"
WEB = ROOT / "web"
SITE = WEB / "site"

SHARED = ["ui.css", "landing.css", "common.js", "landing.js"]
ZIP_TREES = ["src", "config", "scripts"]
ZIP_FILES = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "RULES.md",
    "Start HeadVoice.bat",
    "models/eye_state_svm.yml",
    "models/eye_state.json",
]
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def judge_url() -> str:
    out = ROOT / "cdk-outputs.json"
    if out.exists():
        return json.loads(out.read_text())["HeadVoice"]["JudgeUrl"]
    return "judge"


def build_zip(dest: Path) -> int:
    n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for tree in ZIP_TREES:
            for p in sorted((ROOT / tree).rglob("*")):
                if p.is_file() and not SKIP.intersection(p.parts):
                    z.write(p, Path("HeadVoice") / p.relative_to(ROOT))
                    n += 1
        for f in ZIP_FILES:
            z.write(ROOT / f, Path("HeadVoice") / f)
            n += 1
        z.writestr("HeadVoice/data/sessions/.gitkeep", "")
    return n


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    for f in SHARED:
        shutil.copy(STATIC / f, SITE / f)
    landing = (STATIC / "landing.html").read_text(encoding="utf-8")
    landing = landing.replace("<body>", f'<body data-judge-url="{judge_url()}">', 1)
    (SITE / "index.html").write_text(landing, encoding="utf-8")
    for f in ("download.html", "privacy.html", "pages.css"):
        shutil.copy(WEB / f, SITE / f)
    n = build_zip(SITE / "headvoice-windows.zip")
    size = (SITE / "headvoice-windows.zip").stat().st_size
    print(f"site: {sorted(p.name for p in SITE.iterdir())}")
    print(f"zip: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
