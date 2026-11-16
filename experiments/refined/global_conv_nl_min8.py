"""Minimal tested variant: global + conv + nonlinear branch only,
with 8 Conv filters and layer_size 8.
Run from the repository root:
    python experiments/refined/global_conv_nl_min8.py
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
        use_conv=True,
        layer_size=8,
        hidden_layers=2,
        conv_filters=8,
        num_epochs=3000,
        out_dir="experiments/refined/results_global_conv_nl_min8",
    )
