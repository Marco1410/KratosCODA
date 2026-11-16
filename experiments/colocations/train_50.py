"""Optimized baseline with 50 collocation points.

Run from the repository root:
    python experiments/colocations_opt/train_50.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run

if __name__ == "__main__":
    run(
        interpolation_count=50,
        use_global=True,
        use_ln=False,
        use_nl=True,
        use_conv=True,
        layer_size=12,
        hidden_layers=2,
        conv_filters=8,
        num_epochs=3000,
        out_dir="experiments/colocations_opt/results_50",
    )
