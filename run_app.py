from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import bootstrap


def main() -> None:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    bootstrap.run(
        str(base_path / "app.py"),
        is_hello=False,
        args=sys.argv[1:],
        flag_options={},
    )


if __name__ == "__main__":
    main()