import os
import csv
import json
import pickle

import numpy as np
import scipy as sp
import sqlite3 as sql3

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LF_DATA = 0
HF_DATA = 1

SRC_MAP = {
    "XfoilPanel": LF_DATA,
    "KratosFpotentialcompressible": HF_DATA,
    "KratosFpotentialtransonic": HF_DATA,
    "CODAEuler": HF_DATA
}

LF_FRWK = "KratosFpotentialtransonic"
HF_FRWK = "CODAEuler"


def load_simulations_from_db(db_name):
    ''' Read the data of the db and returns a flatten_vector where:
        Dumps the sql database into an array
    '''
    conn = sql3.connect(db_name)
    return conn.execute("SELECT * FROM simulations").fetchall()


def prepare_data_from_db(db_mame="mysqlite3.db"):
    ''' Read the data of the dumped db and returns a flatten_vector where every row
        represents a simulation with the following info:

        0 -> "m"
        1 -> "p"
        2 -> "t"
        3 -> "M"
        4 -> "a"
        5 -> "LFDATA CP"
        6 -> "HFDATA CP"

        db_mame: Name of the db.
    '''

    data_map = {}
    db_data = load_simulations_from_db(os.path.join(db_mame, "mysqlite3.db"))

    for row in db_data:
        src_file = row[0]
        src_frwk = row[1]
        data_target = None
        if src_frwk == LF_FRWK:
            data_target = 0
        if src_frwk == HF_FRWK:
            data_target = 1

        if data_target is not None:
            with open(os.path.join(db_mame, "output", f"{src_file}.json"), "r") as f:
                file_data = json.load(f)["Cp"]

            if row[2:] not in data_map:
                datum = [None, None]
                datum[data_target] = file_data
                data_map[row[2:]] = datum
            else:
                data_map[row[2:]][data_target] = file_data

    return np.asarray(
        [[k[0], k[1], k[2], k[3], k[4], v[0], v[1]] for k, v in data_map.items()],
        dtype=object,
    )


def remove_duplicates(data):
    ''' Tries to remove duplicated entried from the db if present
    '''

    np.sort(data, axis=0)
    unique_rows, _, _ = np.unique(data, axis=0, return_index=True, return_inverse=True)
    return unique_rows


def has_shock(row):
    ''' Returns true if a shock is detected in the profile or false if not.

        @slope_fract: Minimum value of the slope in any given point to be considered a shock
    '''

    slope_fract = 1
    try:
        top = row[row[:, 1] > 0]
        sort_ord = np.argsort(top[:, 0])
        slop_ord = np.abs(np.ediff1d(top[sort_ord][:, 2], to_end=None, to_begin=None))
        return np.count_nonzero(slop_ord > slope_fract)
    except:
        return 0

def shock_location(x, cp):
    ''' Aproximates the shock location.
    '''

    order = np.argsort(x)
    xs = x[order]
    cps = cp[order]
    if len(cps) < 2:
        return float("nan")
    jumps = np.abs(np.ediff1d(cps))
    i = int(np.argmax(jumps))

    return float(xs[i + 1])


def filter_outlayer_results(flatten_data, chi=2):
    ''' Filter results with chi crtieria.
    '''

    print(
        f"Excluding {len([row for row in flatten_data if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) > chi])} lf results with high (> {chi}) chi error..."
    )
    return np.asarray(
        [row for row in flatten_data
         if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) < chi]
    )


def filter_non_shock(flatten_data):
    ''' Filter results with shock crtieria.
    '''

    return [row for row in flatten_data if has_shock(np.asarray(row[-1])) > 0]


def interpolation_points_for_row(hfd_unique, interpolation_count):
    ''' Interpolates the values from both LF and HF data for a single row.
    '''

    base = hfd_unique[0:400, 0:2]
    base = base[base[:, 1] > 0]
    segment = base[:200]

    if len(segment) == 0:
        return segment

    if interpolation_count == 200:
        return segment.copy()

    if interpolation_count <= len(segment):
        idx = np.linspace(0, len(segment) - 1, interpolation_count, dtype=int)
        return segment[idx].copy()

    # Upsample along the same selected segment (linear interpolation of the 2-D chordwise/airfoil coordinates).
    t = np.linspace(0.0, 1.0, len(segment))
    u = np.linspace(0.0, 1.0, interpolation_count)
    return np.column_stack([
        np.interp(u, t, segment[:, 0]),
        np.interp(u, t, segment[:, 1]),
    ])


def interpolate_cp_coordinates(flatten_data, interpolation_count=200):
    ''' Interpolates the values from both LF and HF data so the training sizes match.

        flatten_data: Parsed data from the DB
    '''

    print(f"Interpolation count: {interpolation_count}")
    for row in range(len(flatten_data)):
        lfd_cp = np.asarray(flatten_data[row][-2])
        hfd_cp = np.asarray(flatten_data[row][-1])

        lfd_unique = remove_duplicates(lfd_cp)
        hfd_unique = remove_duplicates(hfd_cp)

        d_points = interpolation_points_for_row(hfd_unique, interpolation_count)

        d_idx_lf = lfd_unique[:, 0:2][lfd_unique[:, 1] > 0]
        d_idx_hf = hfd_unique[:, 0:2][hfd_unique[:, 1] > 0]

        cp_intrpl1d_lf = sp.interpolate.RBFInterpolator(
            d_idx_lf, lfd_unique[:, 2][lfd_unique[:, 1] > 0]
        )
        cp_intrpl1d_hf = sp.interpolate.RBFInterpolator(
            d_idx_hf, hfd_unique[:, 2][hfd_unique[:, 1] > 0]
        )

        lf_new_cp = np.column_stack((d_points, cp_intrpl1d_lf(d_points)))
        hf_new_cp = np.column_stack((d_points, cp_intrpl1d_hf(d_points)))

        flatten_data[row][-2] = np.asarray(lf_new_cp)
        flatten_data[row][-1] = np.asarray(hf_new_cp)

        if row % 10 == 0:
            print(f"Interpolating row {row}: {flatten_data[row][-2].shape}")
    return flatten_data


class HF_Predictor(nn.Module):
    ''' PyTorch port of the Keras model in train_predict_cp_working.py:

        - Conv1D front-end (16 filters, kernel 5) -> Flatten -> Dense(8000) -> Reshape(200, 40)
        - One per-point branch (3 layers each):
              * nl: ReLU activations
        - Elementwise Add of [ln, nl, conv]  ->  Dense(1)
    '''

    def __init__(
        self,
        input_points=200,
        input_features=5,
        layer_size=40,
        hidden_layers=3,
        conv_filters=16,
        conv_kernel=5,
        use_global=False,
        use_ln=True,
        use_nl=True,
        use_conv=True,
    ):
        super().__init__()
        self.input_points = input_points
        self.layer_size = layer_size
        self.use_global = use_global
        self.use_ln = use_ln
        self.use_nl = use_nl
        self.use_conv = use_conv

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        if use_global:
            global_in = conv_filters if use_conv else input_features
            self.global_dense = nn.Sequential(
                nn.Linear(global_in, layer_size),
                nn.ReLU(),
            )
        else:
            self.global_dense = None

        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(input_features, conv_filters, conv_kernel, padding=conv_kernel // 2),
                nn.ReLU(),
                nn.Conv1d(conv_filters, conv_filters, conv_kernel, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(conv_filters, conv_filters, conv_kernel, padding=8, dilation=4),
                nn.ReLU(),
                nn.Conv1d(conv_filters, conv_filters, conv_kernel, padding=16, dilation=8),
                nn.ReLU(),
            )
            self.conv_project = nn.Conv1d(conv_filters, layer_size, kernel_size=1)
        else:
            self.conv = None
            self.conv_project = None

        sizes = []
        ls = layer_size + (hidden_layers - 1) * 2
        in_dim = input_features
        for _ in range(hidden_layers):
            sizes.append((in_dim, ls))
            in_dim = ls
            ls -= 2

        self.nl_layers = nn.ModuleList([nn.Linear(a, b) for a, b in sizes]) if use_nl else None
        self.ln_layers = nn.ModuleList([nn.Linear(a, b) for a, b in sizes]) if use_ln else None
        self.head = nn.Linear(layer_size, 1)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        L = x.size(1)

        if self.use_conv:
            cv_raw = self.conv(x.permute(0, 2, 1))
            cv = self.conv_project(cv_raw).permute(0, 2, 1)
        else:
            cv_raw = None
            cv = torch.zeros((B, L, self.layer_size), device=x.device)

        if self.use_nl:
            nl = x
            for layer in self.nl_layers:
                nl = F.relu(layer(nl))
        else:
            nl = torch.zeros((B, L, self.layer_size), device=x.device)

        if self.use_ln:
            ln = x
            for layer in self.ln_layers:
                ln = layer(ln)
        else:
            ln = torch.zeros((B, L, self.layer_size), device=x.device)

        z = ln + nl + cv

        if self.use_global:
            if self.use_conv:
                global_features = self.global_dense(self.global_pool(cv_raw).squeeze(-1))
            else:
                global_features = self.global_dense(self.global_pool(x.permute(0, 2, 1)).squeeze(-1))
            z = z + global_features.unsqueeze(1)

        return self.head(z)


def loss_f(pred, y):
    if pred.ndim == 3:
        pred = pred.squeeze(-1)
    if y.ndim == 3:
        y = y.squeeze(-1)
    return F.smooth_l1_loss(pred, y, beta=0.05)


def train_model(model, train_x, train_y, batch_size, num_epochs,
              lr=1e-3, valid_x=None, valid_y=None):
    ''' Train the HF predictor using curve loss.
    '''
    
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if name.endswith("weight"):
            weights.append(param)
        else:
            biases.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": weights, "weight_decay": 1e-4},
            {"params": biases, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    train_x = torch.as_tensor(train_x, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y, dtype=torch.float32, device=DEVICE)
    if valid_x is not None:
        valid_x = torch.as_tensor(valid_x, dtype=torch.float32, device=DEVICE)
        valid_y = torch.as_tensor(valid_y, dtype=torch.float32, device=DEVICE)

    if train_y.ndim == 3:
        train_y = train_y.squeeze(-1)
    if valid_y is not None and valid_y.ndim == 3:
        valid_y = valid_y.squeeze(-1)

    n_train = train_x.size(0)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(train_x[idx]).squeeze(-1)
            y = train_y[idx]
            loss = loss_f(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        epoch_train_loss = total_loss / n_train

        if valid_x is not None and valid_x.size(0) > 0:
            model.eval()
            with torch.no_grad():
                vpred = model(valid_x).squeeze(-1)
                vloss = loss_f(vpred, valid_y).item()
            monitored_loss = vloss
        else:
            vloss = None
            monitored_loss = epoch_train_loss

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(monitored_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            if vloss is not None:
                print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_train_loss:.6f} - val_loss: {vloss:.6f} - lr: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_train_loss:.6f} - lr: {current_lr:.2e}")

    return model


def predict(model, x):
    ''' Runs inference and returns a numpy array of shape (B, 200, 1).
    '''

    model.eval()
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        out = model(x)
    return out.cpu().numpy()


def evaluate(model, flatten_data, minval, maxval, split_threshold):
    results_data = []
    results_coords = []

    for i, sim_data in enumerate(flatten_data):
        row_x = []
        for p_idx in range(len(sim_data[-2])):
            row_x.append([
                sim_data[-4],
                sim_data[-3],
                (sim_data[-2][p_idx, 0] - minval[-3]) / maxval[-3],
                (sim_data[-2][p_idx, 1] - minval[-2]) / maxval[-2],
                (sim_data[-2][p_idx, 2] - minval[-1]) / maxval[-1],
            ])

        input_data = np.asarray([row_x])
        LF_X = sim_data[-2][:, 0]
        HF_X = sim_data[-1][:, 0]
        LF_Z = -sim_data[-2][:, 2]
        HF_Z = -sim_data[-1][:, 2]

        PZ = predict(model, input_data)[0, :, 0]
        PZ = -((PZ * maxval[-1]) + minval[-1])

        chi_pred = float(sum((HF_Z - PZ) ** 2))
        chi_lf = float(sum((HF_Z - LF_Z) ** 2))
        rmse_pred = float(np.sqrt(np.mean((HF_Z - PZ) ** 2))) if len(PZ) else float("nan")
        rmse_lf = float(np.sqrt(np.mean((HF_Z - LF_Z) ** 2))) if len(LF_Z) else float("nan")

        sx_pred = shock_location(LF_X, PZ)
        sx_hf = shock_location(HF_X, HF_Z)
        sx_lf = shock_location(LF_X, LF_Z)

        results_data.append({
            "sim": i,
            "is_valid": i >= split_threshold,
            "chi_pred": chi_pred,
            "chi_lf": chi_lf,
            "rmse_pred": rmse_pred,
            "rmse_lf": rmse_lf,
            "shock_x_pred": sx_pred,
            "shock_x_hf": sx_hf,
            "shock_x_lf": sx_lf,
            "shock_err_pred_hf": abs(sx_pred - sx_hf),
            "shock_err_pred_lf": abs(sx_pred - sx_lf),
        })
        results_coords.append([LF_X.tolist(), PZ.tolist(), LF_Z.tolist(), HF_Z.tolist()])

    return results_data, results_coords


def summarize(results_data):
    def mean(key, subset=None):
        vals = [r[key] for r in results_data if subset is None or r["is_valid"] == subset]
        return float(np.mean(vals)) if vals else float("nan")

    def median(key, subset=None):
        vals = [r[key] for r in results_data if subset is None or r["is_valid"] == subset]
        return float(np.median(vals)) if vals else float("nan")

    return {
        "n_all": len(results_data),
        "n_valid": sum(1 for r in results_data if r["is_valid"]),
        "mean_chi_pred_all": mean("chi_pred"),
        "mean_chi_lf_all": mean("chi_lf"),
        "mean_rmse_pred_all": mean("rmse_pred"),
        "mean_rmse_lf_all": mean("rmse_lf"),
        "mean_shock_err_pred_hf_all": mean("shock_err_pred_hf"),
        "mean_shock_err_pred_lf_all": mean("shock_err_pred_lf"),
        "mean_chi_pred_valid": mean("chi_pred", True),
        "mean_chi_lf_valid": mean("chi_lf", True),
        "mean_rmse_pred_valid": mean("rmse_pred", True),
        "mean_rmse_lf_valid": mean("rmse_lf", True),
        "mean_shock_err_pred_hf_valid": mean("shock_err_pred_hf", True),
        "mean_shock_err_pred_lf_valid": mean("shock_err_pred_lf", True),
        "median_chi_pred_all": median("chi_pred"),
        "median_chi_lf_all": median("chi_lf"),
        "median_chi_pred_valid": median("chi_pred", True),
        "median_chi_lf_valid": median("chi_lf", True),
    }


def run(interpolation_count=200,
        use_global=False,
        use_ln=True,
        use_nl=True,
        use_conv=True,
        out_dir="experiments/results",
        num_epochs=4000,
        seed=42,
        layer_size=40,
        hidden_layers=3,
        conv_filters=16,
        dataset_names=["dataset_02_10_2023_0", "dataset_02_10_2023_1"]):
        
    print(f"\n=== Experiment: interpolation={interpolation_count}, "
          f"use_global={use_global}, use_ln={use_ln}, use_nl={use_nl}, use_conv={use_conv} ===")

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_fract = 0.80

    cache_dir = "experiments/cache"
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"flatten_{interpolation_count}_seed{seed}.pkl"
    )

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            flatten_data = pickle.load(f)
        print(f"Loaded cached interpolated data from {cache_path}")
    else:
        flatten_data = np.empty([0, 7])
        for dataset_name in dataset_names:
            db_path = os.path.join("datasets", dataset_name)
            flatten_data = np.vstack([flatten_data, prepare_data_from_db(db_path)])
            print(flatten_data.shape)

        # Original shuffling: same seed in all variants.
        np.random.shuffle(flatten_data)

        # Filters are applied exactly as in the original script.
        flatten_data = filter_non_shock(flatten_data)
        print("After shock filter:", len(flatten_data))

        flatten_data = interpolate_cp_coordinates(flatten_data, interpolation_count)
        flatten_data = filter_outlayer_results(flatten_data, 2000)

        with open(cache_path, "wb") as f:
            pickle.dump(flatten_data, f)
        print(f"Saved interpolated data cache to {cache_path}")

    model = HF_Predictor(
        input_points=interpolation_count,
        input_features=5,
        layer_size=layer_size,
        hidden_layers=hidden_layers,
        conv_filters=conv_filters,
        conv_kernel=5,
        use_global=use_global,
        use_ln=use_ln,
        use_nl=use_nl,
        use_conv=use_conv,
    ).to(DEVICE)
    print(model)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    minval = []
    maxval = []

    num_features = len(flatten_data[0]) - 2
    for col in range(num_features):
        min_val = np.min(flatten_data[:, col])
        max_val = np.max(flatten_data[:, col])
        flatten_data[:, col] /= max_val
        minval.append(min_val)
        maxval.append(max_val)

    lf_data = np.asarray([r for r in flatten_data[:, -2]])
    hf_data = np.asarray([r for r in flatten_data[:, -1]])

    for i in range(3):
        min_val_x = np.min(lf_data[:, :, i])
        min_val_k = np.min(hf_data[:, :, i])
        min_val = np.minimum(min_val_x, min_val_k)
        minval.append(min_val)

        lf_data[:, :, i] -= min_val
        hf_data[:, :, i] -= min_val

        max_val_lf = np.max(lf_data[:, :, i])
        max_val_hf = np.max(hf_data[:, :, i])
        max_val = np.maximum(max_val_lf, max_val_hf)
        maxval.append(max_val)

        lf_data[:, :, i] /= max_val
        hf_data[:, :, i] /= max_val

    data_x = []
    data_y = []
    for s_idx in range(flatten_data.shape[0]):
        row_x = []
        row_y = []
        for p_idx in range(lf_data.shape[1]):
            row_x.append([
                flatten_data[s_idx, -4],
                flatten_data[s_idx, -3],
                lf_data[s_idx, p_idx, 0],
                lf_data[s_idx, p_idx, 1],
                lf_data[s_idx, p_idx, 2],
            ])
            row_y.append(hf_data[s_idx, p_idx, 2])
        data_x.append(row_x)
        data_y.append(row_y)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    print(f"data_x.shape={data_x.shape}, data_y.shape={data_y.shape}")

    split_threshold = int(data_x.shape[0] * train_fract)
    train_x = data_x[:split_threshold]
    valid_x = data_x[split_threshold:]
    train_y = data_y[:split_threshold]
    valid_y = data_y[split_threshold:]

    train_model(
        model,
        train_x,
        train_y,
        batch_size=max(1, int(train_x.shape[0] / 8)),
        num_epochs=num_epochs,
        lr=1e-3,
        valid_x=valid_x,
        valid_y=valid_y,
    )

    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "models", f"train_{num_epochs}.pt"))

    results_data, results_coords = evaluate(
        model, flatten_data, minval, maxval, split_threshold
    )
    summary = summarize(results_data)

    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    out_name = os.path.basename(out_dir)
    pid = os.getpid()

    # Keep the original result_viewer-compatible chi CSV: chi_pred, chi_lf, is_valid.
    with open(os.path.join(out_dir, "results", f"{out_name}_chi_{pid}.csv"), "w") as f:
        wr = csv.writer(f)
        for r in results_data:
            wr.writerow([r["chi_pred"], r["chi_lf"], r["is_valid"]])

    # Detailed per-simulation metrics used for the MD tables.
    with open(os.path.join(out_dir, "metrics_detail.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerow([
            "sim", "is_valid", "chi_pred", "chi_lf", "rmse_pred", "rmse_lf",
            "shock_x_pred", "shock_x_hf", "shock_x_lf",
            "shock_err_pred_hf", "shock_err_pred_lf"
        ])
        for r in results_data:
            wr.writerow([
                r["sim"], r["is_valid"], r["chi_pred"], r["chi_lf"],
                r["rmse_pred"], r["rmse_lf"], r["shock_x_pred"], r["shock_x_hf"],
                r["shock_x_lf"], r["shock_err_pred_hf"], r["shock_err_pred_lf"]
            ])

    with open(os.path.join(out_dir, "results", f"{out_name}_xyz_{pid}.csv"), "w") as f:
        json.dump({
            "meta": [float(flatten_data[-1, -4]), float(flatten_data[-1, -3])],
            "data": results_coords,
        }, f)

    with open(os.path.join(out_dir, "results", f"{out_name}_xyz_train_{pid}.csv"), "w") as f:
        json.dump({"data": results_coords[:split_threshold]}, f)

    with open(os.path.join(out_dir, "results", f"{out_name}_xyz_valid_{pid}.csv"), "w") as f:
        json.dump({"data": results_coords[split_threshold:]}, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    run()
