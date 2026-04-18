"""Convenience launcher so users can run the pipeline from the project root.

Equivalent to ``python -m scorecard.cli``, but keeps the CWD at the project
root so relative dataset paths like ``data/heloc_dataset_v1.parquet`` resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from scorecard.cli import main as _main

    return _main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
