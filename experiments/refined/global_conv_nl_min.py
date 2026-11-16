"""New compact experiment:
    - global pooled branch is ON by default
    - keep the dilated Conv1d front-end
    - remove the redundant linear per-point branch
    - reduce layer size / hidden layers
Run from the repository root:
    python experiments/refined/global_conv_nl_min.py
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
        layer_size=24,
        hidden_layers=2,
        conv_filters=16,
        num_epochs=3000,
        out_dir="experiments/refined/results_global_conv_nl_min",
    )
