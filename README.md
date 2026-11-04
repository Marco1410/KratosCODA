# CP Error-Correction Network

A neural network that learns to correct the pressure-coefficient (Cp) error of a
low-fidelity (LF) simulation by predicting the high-fidelity (HF) result.

## How to run

1 - 
Download and extract the dataset from: [datasets.zip](https://zenodo.org/records/23916811/files/datasets.zip?download=1)

2 - 
Install the necessary packages in `requirements.txt` 

3 - 
Run the main script:

```bash
python train_predict_cp.py
```

### Model

`HF_Predictor` (PyTorch) — input `(B, 200, 5)` channels-last:

- **Dilated Conv1d front-end** (16 filters, kernel 5): four blocks with dilations
  `1, 2, 4, 8` (padding chosen to keep the 200-point length), ReLU after each.
- **Global branch** — `AdaptiveAvgPool1d(1)` → `Linear(16, 40)` + ReLU. Computed in
  `forward()`, but **not** added to the output in the current implementation.
- **Point projection** — `Conv1d(16 → 40, kernel 1)` maps the conv features to the
  per-point `layer_size` (40), reshaped to `(B, 200, 40)`.
- **Two identical 3-layer per-point branches** (sizes `5 → 44 → 42 → 40`):
  - `nl`: ReLU after each layer.
  - `ln`: no activation (purely linear).
- **Fusion** — element-wise `z = ln + nl + conv` → `Linear(40, 1)` → `(B, 200, 1)`.
- Init: `kaiming_normal_` (HeNormal) on all Linear/Conv1d weights, zero biases.
- Loss: `F.smooth_l1_loss(pred, y, beta=0.05)`.


### Notes

- The train/validation split is a **sequential** index split (first 80 / last 20)
  applied after shuffling (`train_fract = 0.80`).
- Leading parameters are scaled by `/max` only (no min shift), whereas the LF/HF
  `(x, y, cp)` arrays are min-shifted then `/max`.

## Result viewer

A standalone Node/Express + ECharts web app (`result_viewer/`) to browse the
`results/` CSVs, compare PRED vs LF vs HF, and read off shock locations and errors.
This code of the visualizer has been partially generated with AI assistance.

```bash
cd result_viewer
npm install          # once
npm start            # -> http://localhost:8050
```

It scans `results/` for `*_xyz_*.csv` files and, per
simulation, computes and shows:

- The Cp curve PRED / LF / HF with a vertical marker at each shock location.
- chi² error vs HF and vs LF
- RMSE pred-vs-HF.
- Shock location (x) for PRED / LF / HF, and the location error PRED-vs-HF and
  PRED-vs-LF.
- A per-simulation table, plus aggregate stats over all simulations.

The shock is detected as the largest adjacent Cp jump on the chordwise-ordered
upper surface.

The `results/` directory is resolved from the script's parent dir and can be
overridden with the `RESULTS_DIR` env var.
