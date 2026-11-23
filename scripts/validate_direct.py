"""Run the portable V6 validate entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.training import main

if __name__ == "__main__":
    main("direct_correction", evaluation=True)
