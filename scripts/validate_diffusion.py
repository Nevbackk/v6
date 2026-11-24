"""Run Round 5 Genuine Diffusion (Route A: VP-OrbitDiff) validation entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.training_diffusion import main

if __name__ == "__main__":
    main(evaluation=True)
