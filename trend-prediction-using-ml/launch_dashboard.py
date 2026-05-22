"""Launcher for the Streamlit dashboard using local project packages."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"

if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

from streamlit.web.cli import main as streamlit_main


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_ROOT / "dashboard.py"),
        "--global.developmentMode",
        "false",
        "--server.headless",
        "true",
        *sys.argv[1:],
    ]
    raise SystemExit(streamlit_main())
