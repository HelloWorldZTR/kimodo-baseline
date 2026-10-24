"""Command9 entrypoint; see docs/command9_benchmark.md."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimodo_benchmark.score import main


if __name__ == "__main__":
    main()
