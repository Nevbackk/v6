"""Run Model Plan V1 training entry point (LTR-LM, JF-BR, FD-ASF, Posterior-CNF)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.training_v1 import main

if __name__ == "__main__":
    main(evaluation=False)
