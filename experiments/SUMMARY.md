# Experiment summary

All experiments use the same data, fixed shuffle seed `42`, the same
shock/chi filters, and the same 80/20 sequential simulation split.  The
colocation experiments change the number of selected points, while the
interpolation-method audit separately changes the LF/HF mapping method.

## Metric definitions

All curve-fit metrics are computed from the saved `_xyz_*.csv` result files
without retraining.  In the tables, metric values are means over simulations.

- **RMSE**: root mean square pointwise prediction error,
  `sqrt(mean((pred_i - HF_i)^2))`.  It measures overall magnitude of error.
- **MAE**: mean absolute pointwise prediction error,
  `mean(|pred_i - HF_i|)`.  It is less sensitive to large outliers than RMSE.
- **Pearson correlation**: linear correlation coefficient between prediction
  and HF, `cov(pred, HF)/(sigma_pred sigma_HF)`.  It measures monotonic
  association, not error magnitude; a high correlation can coexist with
  large pointwise error.
- **Total variation ratio**: `TV(pred)/TV(HF)`, where
  `TV(x) = sum_i |x_{i+1} - x_i|`.  A ratio near 1 means similar total
  variation; below 1 means the prediction is smoother, above 1 means it
  oscillates more.
- **Derivative sign changes**: number of sign changes in the first differences
  `pred_{i+1} - pred_i`; this counts oscillations/extrema in the prediction.
  The raw count scales with the number of points, so compare counts at the
  same point count or use density/TV ratio for cross-count comparisons.
- **Residual total variation**: `TV(pred - HF) = sum_i |(pred_{i+1}-HF_{i+1})
  - (pred_i-HF_i)|`.  A large residual TV means the prediction wiggles
  around the HF curve even if RMSE is moderate.
- **Shock error**: absolute difference between the predicted shock location
  and the HF shock location.  The value is in the normalized chordwise
  coordinate used by `train_predict_cp.py` and the result CSV, not in
  physical metres.

## Files

| experiment | script | result directory |
|---|---|---|
| optimized baseline 200 points | `colocations_opt/train_200.py` | `colocations_opt/results_200` |
| optimized baseline 100 points | `colocations_opt/train_100.py` | `colocations_opt/results_100` |
| optimized baseline 50 points | `colocations_opt/train_50.py` | `colocations_opt/results_50` |
| optimized baseline model | `refined/global_conv_nl_min12.py` | `refined/results_global_conv_nl_min12` |
| no global branch | `ablation/no_global_branch.py` | `ablation/results_no_global` |
| no nonlinear branch | `ablation/no_nonlinear_branch.py` | `ablation/results_no_nonlinear` |
| no Conv front-end | `ablation/no_conv_frontend.py` | `ablation/results_no_conv_opt` |
| interpolation RBF | `interpolation/train_200_rbf.py` | `interpolation/results_rbf` |
| interpolation nearest | `interpolation/train_200_nearest.py` | `interpolation/results_nearest` |
| interpolation linear | `interpolation/train_200_linear.py` | `interpolation/results_linear` |
| interpolation gradient-adaptive | `interpolation/train_200_adaptive.py` | `interpolation/results_adaptive` |

Shared corrected runner: `experiments/common.py`.

## Colocation results

All three runs use the same optimized baseline:
global branch ON, Conv front-end ON, nonlinear per-point branch ON,
linear branch OFF, `layer_size=12`, `hidden_layers=2`, `conv_filters=8`,
3000 epochs.

The selected segment is preserved.  50 and 100 use evenly spaced subsets
of the same 200-point segment.

### All simulations

| points | MAE | Pearson | TV ratio | deriv. sign changes pred | deriv. sign changes HF | residual TV | shock | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.0237156 | 0.9938520 | 1.0072585 | 7.0417 | 3.1786 | 1.5111428 | 0.0137687 | 0.0565412 |
| 100 | 0.0293230 | 0.9901879 | 1.0210297 | 14.9524 | 3.8452 | 2.7558136 | 0.0544947 | 0.0832451 |
| 200 | 0.0276002 | 0.9905176 | 1.0066010 | 27.9107 | 9.4524 | 3.6881450 | 0.0212028 | 0.0817696 |

### Validation subset

| points | MAE | Pearson | TV ratio | deriv. sign changes pred | deriv. sign changes HF | residual TV | shock | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.0274892 | 0.9921440 | 1.0132975 | 7.9412 | 3.1176 | 1.7658543 | 0.0180249 | 0.0702429 |
| 100 | 0.0328120 | 0.9865971 | 1.0315599 | 15.7647 | 3.7647 | 3.0281767 | 0.0406967 | 0.1007897 |
| 200 | 0.0310718 | 0.9867896 | 1.0227078 | 28.7941 | 9.3824 | 3.9766648 | 0.0249132 | 0.0987101 |

Interpretation:

- In the low-vs-low comparison, 50 points looks best on RMSE and shock
  error, but this is not a valid full-curve comparison because the low-point
  HF is only a sampled version of the raw HF.
- The 100-point low-vs-low run is worse than the 200-point run in RMSE
  and shock error.
- Pearson correlation is high for all runs, but derivative sign changes
  show that predictions oscillate more than the sampled HF.  Raw derivative
  counts scale with the number of points; use density or TV ratio for
  cross-count comparisons.
- Total-variation ratios are near 1, while residual TV is substantial.
  RMSE and shock error alone do not capture curve-level oscillation.

Detailed CSV: `experiments/colocations_opt/shape_metrics.csv`.

## Corrected comparison against raw 200-point HF

The low-vs-low comparison uses the lower-count HF sampled from the same
raw 200-point HF, so it makes the low-point errors artificially favorable.
To check properly, I linearly upsampled the 50/100-point prediction back
to the raw 200 x positions and compared it with the full raw 200-point
HF.

| points (evaluated on 200 raw points) | split | MAE | Pearson | TV ratio | deriv. sign changes pred | deriv. sign changes HF | residual TV | shock | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | all | 0.0272403 | 0.9909024 | 0.9104344 | 7.0417 | 9.4524 | 3.4270526 | 0.1724907 | 0.0831175 |
| 50 | valid | 0.0298498 | 0.9900799 | 0.9202839 | 7.9412 | 9.3824 | 3.5944731 | 0.1533324 | 0.0887810 |
| 100 | all | 0.0291912 | 0.9901032 | 0.9517827 | 14.9524 | 9.4524 | 3.5754282 | 0.0835497 | 0.0862958 |
| 100 | valid | 0.0318279 | 0.9879457 | 0.9660380 | 15.7647 | 9.3824 | 3.7522886 | 0.0891972 | 0.0971810 |
| 200 | all | 0.0276002 | 0.9905176 | 1.0066010 | 27.9107 | 9.4524 | 3.6881450 | 0.0212028 | 0.0817696 |
| 200 | valid | 0.0310718 | 0.9867896 | 1.0227078 | 28.7941 | 9.3824 | 3.9766648 | 0.0249132 | 0.0987101 |

Interpretation:

- The 200-point baseline is best on shock error and all-sample RMSE.
- The 50-point prediction has better validation RMSE but much worse
  shock error when compared with the full raw 200-point HF.
- The 100-point prediction is worse than the 200-point baseline on both
  RMSE and shock error.
- The low-point advantage is therefore an artifact of comparing against
  the coarse low-point HF representation.

CSV: `experiments/colocations_opt/raw200_comparison.csv`.

## Preprocessing-error audit

The selected points come from the HF grid, so HF values at those points are
exact.  LF values are interpolated onto the HF-derived points, so the LF
input is smoothed.

Using nearest raw LF as a grid-mismatch diagnostic:

| quantity | mean | median | max |
|---|---:|---:|---:|
| LF interpolation mismatch, absolute | 0.001763 | 0.000674 | 0.395204 |
| LF interpolation mismatch, relative | 0.011495 | 0.001283 | 35.688 |
| nearest LF-grid distance | 0.000425 | not reported | 0.001244 |

High-gradient subsets:

- LF high-gradient quartile: mean absolute mismatch `0.005012`, max `0.395204`.
- HF high-gradient quartile: mean absolute mismatch `0.002510`, max `0.194264`.
- HF interpolation at selected points: essentially exact, about `1e-12`.

Interpretation: the mismatch is small on average but locally large in sharp-gradient regions.  This supports the reviewer's concern and argues against treating the interpolation as universally harmless.

## Interpolation-method audit (N=200, optimized baseline)

This is a separate preprocessing-sensitivity experiment.  It keeps the
optimized 1685-parameter architecture but changes the LF/HF mapping method.

The linear run uses nearest-neighbour fallback when `LinearNDInterpolator`
returns NaN, so all runs keep the same 168/34 split.

### All simulations

| method | MAE | Pearson | TV ratio | deriv. sign changes pred | deriv. sign changes HF | residual TV | shock | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RBF | 0.0298395 | 0.9900481 | 1.0265847 | 29.5000 | 9.4524 | 3.8976596 | 0.0325735 | 0.0845496 |
| nearest | 0.0280187 | 0.9911963 | 1.0062727 | 26.3869 | 9.4524 | 3.7243780 | 0.0259928 | 0.0807042 |
| linear | 0.0269129 | 0.9917892 | 0.9966397 | 27.3929 | 9.4524 | 3.5621815 | 0.0137661 | 0.0759354 |
| adaptive | 0.0246434 | 0.9920480 | 0.9700641 | 22.6190 | 9.4524 | 3.2202822 | 0.0100838 | 0.0734726 |

### Validation subset

| method | MAE | Pearson | TV ratio | deriv. sign changes pred | deriv. sign changes HF | residual TV | shock | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RBF | 0.0333538 | 0.9863632 | 1.0472670 | 29.9118 | 9.3824 | 4.2033595 | 0.0273740 | 0.1008936 |
| nearest | 0.0302256 | 0.9890673 | 1.0263851 | 26.0000 | 9.3824 | 3.9578369 | 0.0308606 | 0.0919980 |
| linear | 0.0287469 | 0.9894184 | 1.0137340 | 29.8235 | 9.3824 | 3.8331546 | 0.0108951 | 0.0881596 |
| adaptive | 0.0266630 | 0.9898101 | 0.9863850 | 23.4706 | 9.3824 | 3.5139156 | 0.0103091 | 0.0858114 |

Interpretation:

- Interpolation choice materially changes the aggregate metrics.
- Gradient-adaptive interpolation gives the lowest aggregate RMSE and shock
  error in this run, and the lowest MAE and residual total variation.
- Linear interpolation is close, and nearest-neighbour gives the best
  all-sample RMSE among the simple alternatives but worse validation
  shock error.
- RBF is worse on aggregate RMSE/shock and still produces many more
  derivative sign changes than HF.
- The interpolation change reduces error but does not eliminate the
  oscillation problem.

The preprocessing-error audit is in `experiments/interpolation/interpolation_error_audit.md`.  The gradient-adaptive method is documented in `experiments/interpolation/adaptive.md`.
The detailed CSV is `experiments/interpolation/shape_metrics.csv`, and a combined all-method table is in `experiments/interpolation/combined_table.md`.





## Architecture-ablation results (all at 200 points)

The optimized baseline is the compact model with the global branch ON,
the Conv front-end ON, the nonlinear per-point branch ON, and the linear
branch OFF.  The baseline row uses the new 200-point run.

| ablation | params | mean RMSE PRED-HF all | mean RMSE PRED-HF valid | mean shock error PRED-HF all | mean shock error PRED-HF valid |
|---|---:|---:|---:|---:|---:|
| optimized baseline | 1685 | 0.0817696 | 0.0987101 | 0.0212028 | 0.0249132 |
| no global branch | 1577 | 0.085716 | 0.096008 | 0.018131 | 0.025770 |
| no nonlinear branch | 1421 | 0.078097 | 0.094972 | 0.006552 | 0.007290 |
| no Conv front-end | 349 | 0.193608 | 0.208761 | 0.063203 | 0.068157 |

Interpretation:

- Removing the global branch does not collapse the result.  It slightly
  worsens all-sample RMSE and validation shock error, but slightly improves
  validation RMSE and all-sample shock error.  The global branch is a
  refinement, not an essential component.
- Removing the nonlinear per-point branch improves RMSE and shock-location
  error while reducing parameters.  In this compact baseline, the Conv
  front-end plus global branch appear to carry most of the useful correction.
- Removing the Conv front-end roughly doubles RMSE and substantially
  worsens shock-location error.  The dilated Conv1d front-end is the main
  spatial component.

## Refined compact model (global ON by default)

The optimized baseline keeps the Conv front-end, enables the global branch,
removes the redundant linear branch, and reduces the per-point network.

Chosen model: `layer_size=12`, `hidden_layers=2`, `conv_filters=8`, `num_epochs=3000`.

| metric | original baseline | optimized baseline |
|---|---:|---:|
| params | 13453 | 1685 |
| mean RMSE PRED-HF all | 0.087668 | 0.0817696 |
| mean RMSE PRED-HF valid | 0.102082 | 0.0987101 |
| mean shock error PRED-HF all | 0.063071 | 0.0212028 |
| mean shock error PRED-HF valid | 0.051072 | 0.0249132 |

The loss was flat at the end of the 3000-epoch run, so reducing epochs was
safe.  Smaller variants were tested serially; `layer_size=8` fell back to
baseline-level RMSE and worse shock error, so 12 was the best compact
trade-off before the new ablation set.

Files: `experiments/refined/global_conv_nl_min12.py` and
`experiments/refined/global_conv_nl_min.md`.

## Overall conclusions

- The interpolation error is small on average but locally large in
  high-gradient regions.  The current RBF mapping is probably acceptable for
  aggregate inviscid-case metrics, but it is not uniformly harmless.
- Interpolation choice materially changes the aggregate metrics.  In this
  single-seed comparison, gradient-adaptive interpolation gives the best
  aggregate RMSE, shock error, MAE, and residual total variation, while
  linear is close and nearest-neighbour is intermediate.
- Adaptive interpolation reduces preprocessing error but does not
  eliminate prediction oscillation: the prediction still has far more
  derivative sign changes than HF.
- The low-vs-low colocation comparison is misleading because the low-point
  HF is only a sampled version of the raw 200-point HF.  Against the full
  raw 200-point HF, the 200-point baseline is best on shock error and all-sample
  RMSE.
- In the architecture ablations, the Conv front-end is the most important
  component; the global branch is a refinement; the nonlinear per-point branch
  is not essential in the compact baseline.

## Running

From the repository root, run the experiments serially:

```bash
python experiments/colocations_opt/train_200.py
python experiments/colocations_opt/train_100.py
python experiments/colocations_opt/train_50.py
python experiments/refined/global_conv_nl_min12.py
python experiments/ablation/no_global_branch.py
python experiments/ablation/no_nonlinear_branch.py
python experiments/ablation/no_conv_frontend.py
```

The result files are in each experiment's `results/` subdirectory.  The
`result_viewer` can be pointed at one of them with `RESULTS_DIR`:

```bash
cd result_viewer
RESULTS_DIR=../experiments/colocations_opt/results_50/results npm start
```
