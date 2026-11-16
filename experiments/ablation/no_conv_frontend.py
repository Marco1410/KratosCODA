"""Ablation: optimized baseline WITHOUT the Conv1d front-end.
The nonlinear per-point branch and global pooled branch are kept; the linear
branch is already absent in the optimized baseline.  With no Conv front-end,
the global branch pools the raw five input features instead.

Run from the repository root:
    python experiments/ablation/no_conv_frontend.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run

if __name__ == "__main__":
    run(
        interpolation_count=200,
        use_global=True,
        use_ln=False,
        use_nl=True,
        use_conv=False,
        layer_size=12,
        hidden_layers=2,
        conv_filters=8,
        num_epochs=3000,
        out_dir="experiments/ablation/results_no_conv_opt",
    )
