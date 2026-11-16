"""Ablation: optimized baseline WITHOUT the global pooled branch.
The Conv front-end and nonlinear per-point branch are kept; the linear
branch is already absent in the optimized baseline.

Run from the repository root:
    python experiments/ablation/no_global_branch.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run

if __name__ == "__main__":
    run(
        interpolation_count=200,
        use_global=False,
        use_ln=False,
        use_nl=True,
        use_conv=True,
        layer_size=12,
        hidden_layers=2,
        conv_filters=8,
        num_epochs=3000,
        out_dir="experiments/ablation/results_no_global",
    )
