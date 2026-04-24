from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script_name)], check=True)


def main() -> int:
    run("build_api_identity.py")
    run("build_ms_coverage.py")
    run("build_bundles.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
