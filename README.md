# CP Error-Correction Network

A neural network that learns to correct the pressure-coefficient (Cp) error of a
low-fidelity (LF) simulation by predicting the high-fidelity (HF) result.

## How to run

A virtual environment is provided in `.venv` (Python 3.14). It already contains
`torch`, `numpy`, `scipy`, and `matplotlib`.

```bash
.venv/bin/python train_predict_cp.py
```

### Pipeline

1. **Load** — for each dataset, `prepare_data_from_db` reads the `simulations` table
   from the SQLite DB and joins the LF and HF Cp arrays per simulation into a flat
   7-column row `[m, p, t, M, a, LF_CP, HF_CP]` (LF = framework
   `KratosFpotentialtransonic`, HF = `CODAEuler`). All datasets are vstacked into
   `flatten_data`.
2. **Shuffle** — `np.random.shuffle(flatten_data)` (mandatory when loading multiple DBs).
3. **Filter no-shock** — `filter_non_shock` keeps rows where a shock is present. This is calculated with the condition: `has_shock(HF_CP) > 0`.
4. **Interpolate** — `interpolate_cp_coordinates` resamples both LF and HF Cp onto a
   common 200-point upper-surface grid (HF x/y, `y > 0`) using a 2-D RBF
   interpolator per fidelity.
5. **Filter outliers** — `filter_outlayer_results(…, 2000)` drops rows whose
   LF-vs-HF chi² error exceeds 2000.
6. **Build model** — `generate_model()` creates the `HF_Predictor`.

7. **Assemble tensors** — `data_x` is `(N, 200, 5)` = `[M, a, lf_x, lf_y, lf_cp]`
   per point; `data_y` is `(N, 200)` = the scaled HF cp.
9. **Train** — `train_model` runs Adam (lr `1e-3`, weight decay `1e-4` on weights
    only) with a `ReduceLROnPlateau` scheduler (factor `0.5`, patience `50`,
    min `1e-6`) over `4000` epochs, batch size `N_train/8`; training loss is
    smooth-L1 (beta `0.05`).

Data is split `80/20` into training and validation. Separate results are generated for validation and training and stored in the `results` folder with the process-id that originated them. 

## Model

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

## Result viewer

A standalone Node/Express + ECharts web app (`result_viewer/`) to browse the
`results/` CSVs, compare PRED vs LF vs HF, and read off shock locations and errors.

```bash
cd result_viewer
npm install          # once
npm start            # -> http://localhost:8050
```

It scans `results/` for `*_xyz_*.csv` files (no hardcoding of filenames) and, per
simulation, computes and shows:

- The Cp curve PRED / LF / HF with a vertical marker at each shock location.
- chi² error vs HF and vs LF (from the matching `*_chi_*.csv` when present,
  otherwise recomputed from the xyz data).
- RMSE pred-vs-HF.
- Shock location (x) for PRED / LF / HF, and the location error PRED-vs-HF and
  PRED-vs-LF.
- A per-simulation table, plus aggregate stats over all simulations.

The shock is detected as the largest adjacent Cp jump on the chordwise-ordered
upper surface; the leading-edge suction peak is a curved rise spread over several
tiny steps, so the genuine shock discontinuity has the largest single-step jump.

The `results/` directory is resolved from the script's parent dir and can be
overridden with the `RESULTS_DIR` env var.

## Notes / caveats

- The train/validation split is a **sequential** index split (first 80 / last 20)
  applied after shuffling (`train_fract = 0.80`).
- Leading parameters are scaled by `/max` only (no min shift), whereas the LF/HF
  `(x, y, cp)` arrays are min-shifted then `/max`.
- `interpolate_cp_coordinates` resamples to **200** upper-surface points (the HF
  grid is truncated to 400, filtered to `y > 0`, then the first 200 kept).
- `plot_control_sim` evaluates over the **entire** dataset (train + validation) and
  writes four CSVs (full `chi`, full `xyz`, `xyz_train`, `xyz_valid`); validation
  samples are flagged via the `is_validation` column in the chi CSV.
- **Bug:** the validation step in `train_model` calls `curve_loss(…,
  lambda_deriv=lambda_deriv)`, but neither `curve_loss` nor `lambda_deriv` is defined
  in this file — so training with a validation set (the default) raises a
  `NameError` on the first validation step. The training loss is
  `F.smooth_l1_loss(…, beta=0.05)`.
- **Dead code:** `forward()` computes a `global_features` vector (global pool +
  dense) that is never used in the returned output.
- **No-op check:** the post-training `# Check with validation` block calls `predict`
  on `train_x` (first sample) and its print loop is commented out, so it does nothing.
- **Off-by-one:** the `is_validation` flag uses `i > split_threshold`, while the
  validation slice is `data_x[split_threshold:]` — so index `split_threshold` is in
  the validation set but marked as not-validation.
