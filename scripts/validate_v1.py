"""Run Model Plan V1 validation entry point (CL-GN1, CL-WindowLM, LTR-LM, JF-BR, FD-ASF)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.training_v1 import main

if __name__ == "__main__":
    main(evaluation=True)
