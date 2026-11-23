"""Run Model Plan V2 (ASD-SC) progressive self-correcting training entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.training_v2 import main

if __name__ == "__main__":
    main(evaluation=False)
