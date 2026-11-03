import os
import sys
import csv
import json

import numpy as np
import scipy as sp
import sqlite3 as sql3
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LF_DATA = 0
HF_DATA = 1

SRC_MAP = {
    "XfoilPanel"                        : LF_DATA,
    "KratosFpotentialcompressible"      : HF_DATA,
    "KratosFpotentialtransonic"         : HF_DATA,
    "CODAEuler"                         : HF_DATA
}

LF_FRWK = "KratosFpotentialtransonic"
HF_FRWK = "CODAEuler"

def load_simulations_from_db(db_name):
    ''' Read the data of the db and returns a flatten_vector where:
        Dumps the sql database into an array
    '''
    conn = sql3.connect(db_name)

    query = f"SELECT * FROM simulations"

    cursor = conn.execute(query)

    return cursor.fetchall()

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
    db_data = load_simulations_from_db(os.path.join(db_mame,"mysqlite3.db"))

    for row in db_data:
        src_file = row[0]
        src_frwk = row[1]
        data_target = None
        # Retrieve the data
        if src_frwk == LF_FRWK:
            data_target = 0
        if src_frwk == HF_FRWK:
            data_target = 1

        if data_target is not None:
            with open(os.path.join(db_mame,"output",f"{src_file}.json"), "r") as src_file_data:
                file_data = json.load(src_file_data)["Cp"]

                if row[2:] not in data_map:
                    datum = [None, None]
                    datum[data_target] = file_data
                    data_map[row[2:]] = datum
                else:
                    data_map[row[2:]][data_target] = file_data

    # Flatten the data and return
    return np.asarray([[k[0], k[1], k[2], k[3], k[4], v[0], v[1]] for k,v in data_map.items()], dtype=object)

def prepare_data_raw(raw_path):
    ''' Read the data of the dumped db and returns a flatten_vector where every row
        represents a simulation with the following info:

        0  -> "R_le"
        1  -> "L_le"
        2  -> "rr_le"
        3  -> "psi1_ledeg"
        4  -> "psi2_ledeg"
        5  -> "L1_te"
        6  -> "L2_te"
        7  -> "theta1_tedeg"
        8  -> "theta2_tedeg"
        9  -> "M"
        10 -> "a"
        11 -> "xfoil cp"
        12 -> "kratos cp"

        db_mame: Name of the db.
    '''

    print(f"Directory {raw_path} has {len(os.listdir(raw_path))} files")

    data_map = {}

    for src_file_data_name in os.listdir(raw_path):
        with open(os.path.join(raw_path,f"{src_file_data_name}"), "r") as src_file_data:
            as_json = json.load(src_file_data)

            simulation_setting = as_json["Settings"]
            simulation_model   = as_json["Model"]

            file_data = as_json["Cp"]

            control = np.max(np.abs(np.asarray(file_data)[:,2]))

            if control >= 10:
                print(f"Excluding: {src_file_data_name} -- {control}...")

            key = tuple(simulation_setting.values())

            if key not in data_map:
                data_map[key] = [None, None]
            if control < 10:
                data_map[key][SRC_MAP[simulation_model]] = file_data

    inva_map = [k for k,v in data_map.items() if v[0] == None or  v[1] == None]

    print(f"The following invalid entries were found: {len(inva_map)}")

    data_map = {k:v for k,v in data_map.items() if v[0] != None and v[1] != None}

    # Flatten the data and return
    return np.asarray([[k[-11], k[-10], k[-9], k[-8], k[-7], k[-6], k[-5], k[-4], k[-3], k[-2], k[-1], v[0], v[1]] for k,v in data_map.items()], dtype=object)

def remove_duplicates(data):
    ''' Tries to remove duplicated entried from the db if present
    '''

    np.sort(data, axis=0)
    unique_rows, indices, inverse_indices = np.unique(data, axis=0, return_index=True, return_inverse=True)

    return unique_rows

def has_shock(row):
    ''' Returns true if a shock is detected in the profile or false if not.

        @slope_fract: Minimum value of the slope in any given point to be considered a shock
    '''

    slope_fract = 1

    try:
        top = row[row[:,1] > 0]
        sort_ord = np.argsort(top[:,0])
        slop_ord = np.abs(np.ediff1d(top[sort_ord][:,2], to_end=None, to_begin=None))
        return np.count_nonzero(slop_ord > slope_fract)
    except:
        return 0

def filter_outlayer_results(flatten_data, chi=2):
    ''' Filter results with chi crtieria.
    '''

    print(f"Excluding {len([row for row in flatten_data if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) > chi])} lf results with high (>{chi}) chi error...")

    return np.asarray([row for row in flatten_data if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) < chi])

def filter_non_shock(flatten_data):
    ''' Filter results with shock crtieria.
    '''

    return [row for row in flatten_data if has_shock(np.asarray(row[-1])) > 0]

def interpolate_cp_coordinates(flatten_data):
    ''' Interpolates the values from both LF and HF data so the training sizes match.

        flatten_data: Parsed data from the DB
    '''

    # Get the points from both simulations
    lfd_cp = np.asarray(flatten_data[0][-2])
    hfd_cp = np.asarray(flatten_data[0][-1])

    # Get the interpolation points
    lfd_x0, lfd_y0 = lfd_cp[:, 0], lfd_cp[:, 1]
    hfd_x0, hfd_y0 = hfd_cp[:, 0], hfd_cp[:, 1]

    # Create a filter of valid LF and HF values (remove the ones that fall outside the interpolation range)
    idx_min_x = np.min(hfd_x0)
    idx_max_x = np.max(hfd_x0)

    valid_idx = np.where((lfd_x0>idx_min_x) & (lfd_x0<idx_max_x))

    # Make the interpolation
    for row in range(len(flatten_data)):
        # Split the low fidelity and high fidelity data
        lfd_cp = np.asarray(flatten_data[row][-2])
        hfd_cp = np.asarray(flatten_data[row][-1])

        # Remove duplicates from the data
        lfd_unique = remove_duplicates(lfd_cp)
        hfd_unique = remove_duplicates(hfd_cp)

        # Select the number of points
        n_points = 400 # min(lfd_cp.shape[0], hfd_cp.shape[0])
        d_points = hfd_unique[0:n_points, 0:2]
        d_points = d_points[:,:][d_points[:,1] > 0 ]
        d_points = d_points[:200]

        # Make an interpolator using the HF data
        try:
            d_idx_lf = lfd_unique[:,0:2][lfd_unique[:,1] > 0]
            d_idx_hf = hfd_unique[:,0:2][hfd_unique[:,1] > 0]
        except Exception as e:
            print("Failed inserting", flatten_data[row])

        cp_intrpl1d_lf = sp.interpolate.RBFInterpolator(d_idx_lf, lfd_unique[:, 2][lfd_unique[:,1] > 0])
        cp_intrpl1d_hf = sp.interpolate.RBFInterpolator(d_idx_hf, hfd_unique[:, 2][hfd_unique[:,1] > 0])

        # Interpolate both lf and hf data
        lf_new_cp = np.column_stack((d_points, cp_intrpl1d_lf(d_points)))
        hf_new_cp = np.column_stack((d_points, cp_intrpl1d_hf(d_points)))

        # Assign the Interpolated values
        flatten_data[row][-2] = np.asarray(lf_new_cp)
        flatten_data[row][-1] = np.asarray(hf_new_cp)

        if row % 10 == 0:
            print(f"Interpolating row: {row}:{flatten_data[row][-2].shape}")

class HF_Predictor(nn.Module):
    ''' PyTorch port of the Keras model in train_predict_cp_working.py:

        - Conv1D front-end (16 filters, kernel 5) -> Flatten -> Dense(8000) -> Reshape(200, 40)
        - two per-point branches (3 layers each, 5->44->42->40):
              * nl: ReLU activations
              * ln: linear (no activation)
        - elementwise Add of [ln, nl, conv]  ->  Dense(1)
    '''

    def __init__(self, input_points=200, input_features=5, layer_size=40,
                 hidden_layers=3, conv_filters=16, conv_kernel=5):
        super().__init__()
        self.input_points = input_points
        self.layer_size = layer_size

        conv_out_len = input_points - conv_kernel + 1
        # self.conv = nn.Conv1d(input_features, conv_filters, conv_kernel, padding=conv_kernel // 2)  # padding=0 (valid)
        # self.conv_dense = nn.Linear(conv_filters * conv_out_len, input_points * layer_size)
        
        # Replaces Flatten -> Linear(3136, 8000)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.global_dense = nn.Sequential(
            nn.Linear(16, 40),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 16, 5, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=4, dilation=2),
            nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=8, dilation=4),
            nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=16, dilation=8),
            nn.ReLU()
        )
        
        self.conv_project = nn.Conv1d(
            conv_filters,
            layer_size,
            kernel_size=1
        )

        sizes = []
        ls = layer_size + (hidden_layers - 1) * 2
        in_dim = input_features

        for _ in range(hidden_layers):
            sizes.append((in_dim, ls))
            in_dim = ls
            ls -= 2

        self.nl_layers = nn.ModuleList([nn.Linear(a, b) for a, b in sizes])
        self.ln_layers = nn.ModuleList([nn.Linear(a, b) for a, b in sizes])
        self.head = nn.Linear(layer_size, 1)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, input_points, input_features)  channels-last, matching the Keras input
        B = x.size(0)
        act_sq = torch.zeros((), device=x.device)

        cv = x.permute(0, 2, 1)
        cv = self.conv(cv)

        global_features = self.global_pool(cv).squeeze(-1)
        global_features = self.global_dense(global_features)

        cv = self.conv_project(cv)
        cv = cv.permute(0, 2, 1)

        # nonlinear branch (ReLU)
        nl = x
        for layer in self.nl_layers:
            nl = F.relu(layer(nl))
            act_sq = act_sq + torch.sum(nl ** 2) / B

        # linear branch (no activation)
        ln = x
        for layer in self.ln_layers:
            ln = layer(ln)
            act_sq = act_sq + torch.sum(ln ** 2) / B

        # elementwise fusion (Keras layers.Add)
        z = ln + nl + cv                                     # (B, L, layer_size)
        out = self.head(z)                                   # (B, L, 1)

        self._act_sq = act_sq
        return out

def loss_f(pred, y):

    if pred.ndim == 3:
        pred = pred.squeeze(-1)

    if y.ndim == 3:
        y = y.squeeze(-1)

    return F.smooth_l1_loss(
        pred, y,
        beta=0.05
    )

def generate_model():
    ''' Creates the PyTorch model to train
    '''
    model = HF_Predictor(
        input_points=200,
        input_features=5,
        layer_size=40,
        hidden_layers=3,
        conv_filters=16,
        conv_kernel=5,
    )
    model.to(DEVICE)
    print(model)
    return model

def train_model(model, train_x, train_y, batch_size, num_epochs,
                lr=1e-3, valid_x=None, valid_y=None):

    """Train the HF predictor using curve loss."""

    # Separate weights and biases
    weights = []
    biases = []

    for name, param in model.named_parameters():
        if name.endswith("weight"):
            weights.append(param)
        else:
            biases.append(param)

    # Adam with L2 regularization on weights only
    optimizer = torch.optim.Adam(
        [
            {"params": weights, "weight_decay": 1e-4},
            {"params": biases, "weight_decay": 0.0}
        ],
        lr=lr
    )

    # Learning-rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
        min_lr=1e-6
    )

    # Convert data to tensors
    train_x = torch.as_tensor(
        train_x, dtype=torch.float32, device=DEVICE
    )
    train_y = torch.as_tensor(
        train_y, dtype=torch.float32, device=DEVICE
    )

    if valid_x is not None:
        valid_x = torch.as_tensor(
            valid_x, dtype=torch.float32, device=DEVICE
        )
        valid_y = torch.as_tensor(
            valid_y, dtype=torch.float32, device=DEVICE
        )

    # Accept targets of shape (B, 200) or (B, 200, 1)
    if train_y.ndim == 3:
        train_y = train_y.squeeze(-1)

    if valid_y is not None and valid_y.ndim == 3:
        valid_y = valid_y.squeeze(-1)

    n_train = train_x.size(0)

    for epoch in range(num_epochs):

        # ---------------- TRAINING ----------------
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

            # Weighted by batch size for accurate epoch mean
            total_loss += loss.item() * len(idx)

        epoch_train_loss = total_loss / n_train

        # ---------------- VALIDATION ----------------
        if valid_x is not None and valid_x.size(0) > 0:

            model.eval()

            with torch.no_grad():

                vpred = model(valid_x).squeeze(-1)

                vloss = curve_loss(
                    vpred, valid_y,
                    lambda_deriv=lambda_deriv
                ).item()

            monitored_loss = vloss

        else:
            vloss = None
            monitored_loss = epoch_train_loss

        # ---------------- SCHEDULER ----------------
        old_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(monitored_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr < old_lr:
            print(
                f"Learning rate reduced: "
                f"{old_lr:.2e} -> {current_lr:.2e}"
            )

        # ---------------- LOGGING ----------------
        if epoch % 10 == 0 or epoch == num_epochs - 1:

            if vloss is not None:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"loss: {epoch_train_loss:.6f} - "
                    f"val_loss: {vloss:.6f} - "
                    f"lr: {current_lr:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"loss: {epoch_train_loss:.6f} - "
                    f"lr: {current_lr:.2e}"
                )

    return model

def predict(model, x):
    ''' Runs inference and returns a numpy array of shape (B, 200, 1).
    '''
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        out = model(x)
    return out.cpu().numpy()

def plot_control_sim(model, flatten_data, minval, maxval, split_threshold, dataset_name):
    # Generate the data for the training process (Simulation parameters + coords + pvalue)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # chi LF, chi PRED, is_validation
    results_data = []

    # x, y_pred, y_LF, x_HF
    results_coords = []

    for i in range(len(flatten_data[:])):
        sim_data = flatten_data[i]

        row_x = []
        for p_idx in range(len(sim_data[-2])):
            row_x.append([
                sim_data[-4],
                sim_data[-3],
                (sim_data[-2][p_idx,0] - minval[-3]) / maxval[-3],
                (sim_data[-2][p_idx,1] - minval[-2]) / maxval[-2],
                (sim_data[-2][p_idx,2] - minval[-1]) / maxval[-1]
            ])

        # As np array
        input_data = np.asarray([row_x])

        LF_X = sim_data[-2][:, 0]
        HF_X = sim_data[-1][:, 0]

        LF_Z = -sim_data[-2][:, 2]
        HF_Z = -sim_data[-1][:, 2]

        x_idx = np.argsort(LF_X)

        print(f"{input_data.shape=}")

        PZ = predict(model, input_data)

        print(f"{PZ[0].shape=}")

        PZ = -((PZ[0,:,0] * maxval[-1]) + minval[-1])

        # plot C over X
        if sum((HF_Z - LF_Z)**2) > 10:
            ax.plot(LF_X, HF_Z, 'o', markersize=4, label=f"HFDAT_{i}")
            ax.plot(LF_X, LF_Z, 'x', markersize=4, label=f"LFDAT_{i}")
            ax.plot(LF_X, PZ, 'v', markersize=4, label=f"PREDI_{i}")

        print(f"Chi PREDI({i}): {sum((HF_Z - PZ)**2)}")
        print(f"CHI LFDAT({i}): {sum((HF_Z - LF_Z)**2)}")

        results_data.append([sum((HF_Z - PZ)**2), sum((HF_Z - LF_Z)**2), i > split_threshold])
        results_coords.append([LF_X.tolist(), PZ.tolist(), LF_Z.tolist(), HF_Z.tolist()])

    ax.legend(loc="upper left")

    training_coords = results_coords[:split_threshold]
    validation_coords = results_coords[split_threshold:]

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results",dataset_name.replace("/","_") + "_chi_" + str(os.getpid()) + ".csv"), 'w') as results_file:
        wr = csv.writer(results_file)
        wr.writerows(results_data)

    with open(os.path.join("results",dataset_name.replace("/","_") + "_xyz_" + str(os.getpid()) + ".csv"), 'w') as results_file:
        json.dump({"meta":[sim_data[-4], sim_data[-3]], "data":results_coords}, results_file)

    # Save training curves
    with open(os.path.join(
        "results",
        dataset_name.replace("/", "_") + "_xyz_train_" + str(os.getpid()) + ".csv"
    ), 'w') as results_file:

        json.dump({
            "data": training_coords
        }, results_file)

    # Save validation curves
    with open(os.path.join(
        "results",
        dataset_name.replace("/", "_") + "_xyz_valid_" + str(os.getpid()) + ".csv"
    ), 'w') as results_file:

        json.dump({
            "data": validation_coords
        }, results_file)

    # plt.show()

def plot_result_space(flatten_data, train_fract):

    to_ = int(flatten_data.shape[0] * (1-train_fract))

    # a,M axis
    a_values = np.array([t[3] for t in flatten_data])
    b_values = np.array([t[4] for t in flatten_data])

    # Markers
    flatten_data = np.array(flatten_data)
    s_values = np.array([has_shock(np.asarray(row[-1])) for row in flatten_data])

    # Plot train data
    train_a = a_values[:-to_]
    train_b = b_values[:-to_]
    train_s = s_values[:-to_]

    plt.scatter(train_a[train_s == 0], train_b[train_s == 0], color='blue', label='Training', marker='x')
    plt.scatter(train_a[train_s != 0], train_b[train_s != 0], color='cyan', label='Training', marker='o')

    # Plot valid data
    valid_a = a_values[-to_:]
    valid_b = b_values[-to_:]
    valid_s = s_values[-to_:]

    plt.scatter(valid_a[valid_s == 0], valid_b[valid_s == 0], color='red', label='Training', marker='x')
    plt.scatter(valid_a[valid_s != 0], valid_b[valid_s != 0], color='orange', label='Training', marker='o')

    plt.xlabel('M')
    plt.ylabel('a')
    plt.title('Scatterplot of M vs. a')
    plt.show()

    exit()

if __name__ == "__main__":

    dataset_names = [
        "dataset_02_10_2023_0",
        "dataset_02_10_2023_1",
        # "dataset_05_10_2023_0",
        # "dataset_05_10_2023_1",
    ]

    flatten_data = np.empty([0, 7])
    for dataset_name in dataset_names:
        db_path = os.path.join("datasets",dataset_name)
        flatten_data = np.vstack([flatten_data,prepare_data_from_db(db_path)])
        print(flatten_data.shape)

    # Shuffle the simulations. This is mandatory while loading multiple db
    np.random.shuffle(flatten_data)

    # Define the training data fraction
    train_fract = 0.80

    # Select only N simulations from the DB ( usefull for prototyping )
    flatten_data = flatten_data[:, :]

    # Filter simulations without shock (or with a very small error)
    flatten_data = filter_non_shock(flatten_data)
    print(len(flatten_data))

    # Prepare the results for processing
    interpolate_cp_coordinates(flatten_data)

    # Filter results with abnormal values
    flatten_data = filter_outlayer_results(flatten_data,2000)

    predict_model = generate_model()

    # Scale the data simulation input data
    minval = []
    maxval = []

    num_features = len(flatten_data[0]) - 2

    for col in range(num_features):
        min_val = np.min(flatten_data[:,col])
        max_val = np.max(flatten_data[:,col])

        flatten_data[:,col] /= (max_val)

        minval.append(min_val)
        maxval.append(max_val)

    # For the object type data
    lf_data = np.asarray([r for r in flatten_data[:,-2]])
    hf_data = np.asarray([r for r in flatten_data[:,-1]])

    # Scale the values
    for i in range(3):
        min_val_x = np.min(lf_data[:,:,i])
        min_val_k = np.min(hf_data[:,:,i])

        min_val = np.minimum(min_val_x, min_val_k)
        minval.append(min_val)

        lf_data[:,:,i] -= min_val
        hf_data[:,:,i] -= min_val

        max_val_lf = np.max(lf_data[:,:,i])
        max_val_hf = np.max(hf_data[:,:,i])

        max_val = np.maximum(max_val_lf, max_val_hf)
        maxval.append(max_val)

        lf_data[:,:,i] /= max_val
        hf_data[:,:,i] /= max_val

    # Generate the data for the training process (Simulation parameters + coords + pvalue)
    data_x = []
    data_y = []

    print(f"{flatten_data.shape=}")

    for s_idx in range(flatten_data.shape[0]):
        row_x = []
        row_y = []

        for p_idx in range(lf_data.shape[1]):
            row_x.append([
                flatten_data[s_idx,-4],
                flatten_data[s_idx,-3],
                lf_data[s_idx,p_idx,0],
                lf_data[s_idx,p_idx,1],
                lf_data[s_idx,p_idx,2]
            ])

            row_y.append(
                hf_data[s_idx,p_idx,2]
            )

        data_x.append(row_x)
        data_y.append(row_y)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    print(f"{data_x.shape=}, {data_y.shape=}")

    # Split the data into training and validation
    split_threshold = int(data_x.shape[0] * train_fract)

    # Split the data into input and expected outputs
    train_x = data_x[:split_threshold,:]
    valid_x = data_x[split_threshold:,:]

    train_y = data_y[:split_threshold,:]
    valid_y = data_y[split_threshold:,:]

    # Set num epochs
    num_epochs = 4000

    # Train the model
    train_model(
        predict_model,
        train_x,
        train_y,
        batch_size=int(train_x.shape[0]/8),
        num_epochs=num_epochs,
        lr=1e-3,
        valid_x=valid_x,
        valid_y=valid_y,
    )

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(predict_model.state_dict(), f"models/train_{num_epochs}.pt")

    # Check with validation
    predi_y = predict(predict_model, train_x)

    print(f"{predi_y.shape=}")

    for i in range(0, 30):
        orig = train_x[0][i][-1] * max_val + min_val
        expc = train_y[0][i]     * max_val + min_val
        pred = predi_y[0][i]     * max_val + min_val

        # print(f"PREDI: {pred:.4f} ({np.abs(pred-expc):.4f}), LFDAT: {orig:.4f} ({np.abs(orig-expc):.4f}), HFDAT: {expc:.4f}")

    plot_control_sim(predict_model, flatten_data, minval, maxval, split_threshold, dataset_name)
