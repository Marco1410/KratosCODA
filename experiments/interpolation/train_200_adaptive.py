"""Optimized baseline, 200 points, gradient-adaptive interpolation.

In low-gradient LF regions this uses RBF interpolation.  In high-gradient
LF regions (top quartile of local LF gradient) it uses the nearest raw LF
sample, avoiding RBF smoothing across sharp changes.

Run from the repository root:
    python experiments/interpolation/train_200_adaptive.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_interp import run

if __name__ == "__main__":
    run(
        interpolation_count=200,
        use_global=True,
        use_ln=False,
        use_nl=True,
        use_conv=True,
        layer_size=12,
        hidden_layers=2,
        conv_filters=8,
        num_epochs=3000,
        interp_method="adaptive",
        out_dir="experiments/interpolation/results_adaptive",
    )
