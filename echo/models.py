import gc
import random
import json
import glob
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score



def train_elasticNet(instance_path) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    csv_name = geneID + f'_{window/(10**6)}Mb_gene_specific_probes.csv'
    gene_data_path = instance_path / csv_name
    df = pd.read_csv(gene_data_path, index_col=0)
    df = df.iloc[1:]
    print(f"Loaded {csv_name} with shape {df.shape}")

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    cpg_names = df.columns[1:]
    print("Data shape:", X.shape, " | Features:", len(cpg_names), " | Y shape:", y.shape)

    elastic_net = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        max_iter=5000,
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', elastic_net)
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    pipeline.fit(X, y)
    enet_model = pipeline.named_steps['model']

    print()
    print("Cross-validated R² scores:", scores)
    print("Mean R²:", np.mean(scores))
    print("Best alpha (λ):", enet_model.alpha_)
    print("Best l1_ratio:", enet_model.l1_ratio_)

    coefs = enet_model.coef_

    coef_df = pd.DataFrame({
        "CpG": cpg_names,
        "Coefficient": coefs
    })

    csv_name = geneID + f'_{window/(10**6)}Mb_gene_specific_elasticNet_weights.csv'
    coef_df_path = instance_path / csv_name
    coef_df.to_csv(coef_df_path)
    print(f"Exporeted coefficients df of shape {coef_df.shape} to {coef_df_path}")

def plot_elasticNet(instance_path, use_dist, save_dist) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
        geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    csv_name = geneID + f'_{window/(10**6)}Mb_gene_specific_elasticNet_weights.csv'
    coef_df_path = instance_path / csv_name
    coef_df = pd.read_csv(coef_df_path, index_col=0)

    wt_cg_coefs = coef_df

    if use_dist :
        file_path = Path(__file__).parent.parent / "data"
        meth_map_path = file_path / "probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
        meth_map = pd.read_csv(meth_map_path, sep='\t')
        meth_map = meth_map.drop(columns=['gene', 'chromEnd', 'strand'])
        meth_map = meth_map.set_index('#id')

        rel_map = meth_map.loc[wt_cg_coefs['CpG'].to_frame().T.iloc[0].to_list()]
        wt_cg_coefs['chromStart'] = rel_map['chromStart'].values
        wt_cg_coefs['chromStart'] = wt_cg_coefs['chromStart'] - start
        plt.plot(wt_cg_coefs['chromStart'].values, wt_cg_coefs['Coefficient'].values)
        plt.xlabel('Distance from gene')

        if save_dist :
            wt_cg_coefs.to_csv(coef_df_path)
            print(f"Saved CpG distances to {coef_df_path}")
    else : 
        plt.plot(range(len(wt_cg_coefs)), wt_cg_coefs['Coefficient'].values)
        plt.xlabel('CpG in order')
    
    plt.title(geneID + f'_{window/(10**6)}Mb')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.show()

# --- Model Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.leaky_relu(out)
        return out

class AdaptiveRegressionCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        out_channels_conv1 = min(64, input_size // 10)
        out_channels_conv2 = min(32, input_size // 20)
        self.conv1 = nn.Conv1d(1, out_channels_conv1, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(out_channels_conv1)
        self.conv2 = nn.Conv1d(out_channels_conv1, out_channels_conv2, kernel_size=3, padding=1)
        self.resblock2 = ResidualBlock(out_channels_conv2)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self._to_linear = None
        self._calculate_to_linear(input_size)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)

    def _calculate_to_linear(self, L):
        x = torch.randn(1, 1, L)
        x = self.leaky_relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.resblock2(x)
        self._to_linear = x.numel() // x.size(0)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.resblock2(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_arCNN(instance_path) -> None :
    def load_igf_like_csv(path):
        df = pd.read_csv(path)
        df_samples = df.iloc[1:].reset_index(drop=True)
        df_samples_numeric = df_samples.apply(pd.to_numeric, errors="coerce")

        y = df_samples_numeric.iloc[:, 1].values.astype(np.float32)
        X = df_samples_numeric.iloc[:, 2:].values.astype(np.float32)

        print(f"Using input length {X.shape[1]}.")
        input_length_expected = X.shape[1]

        return X, y, input_length_expected

    def make_model(input_size):
        return AdaptiveRegressionCNN(input_size=input_size)

    def train_one_split(model, x_train, y_train, x_val, y_val, lr=1e-3, epochs=500, patience=25, device="cpu"):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        best_state = None
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).cpu().numpy()
        r2 = r2_score(y_val.cpu().numpy(), val_pred)
        return model, r2, best_val_loss

    def cross_validate_and_report(X, Y, input_size, n_splits=5, lr=1e-3, epochs=500, patience=25, device="cpu"):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2s = []
        fold = 1
        for tr_idx, va_idx in kf.split(X):
            x_tr, x_va = X[tr_idx], X[va_idx]
            y_tr, y_va = Y[tr_idx], Y[va_idx]
            model = make_model(input_size)
            _, r2, best_val = train_one_split(model, x_tr, y_tr, x_va, y_va,
                                            lr=lr, epochs=epochs, patience=patience, device=device)
            print(f"Fold {fold}: R2 = {r2:.4f}, best val MSE = {best_val:.6f}")
            r2s.append(r2)
            fold += 1
            torch.cuda.empty_cache()
        print(f"Mean R2 across {n_splits} folds: {np.mean(r2s):.4f}")
        return np.mean(r2s)

    def train_full_and_save(X, Y, input_size, models_dir, gene_name, lr=1e-3, epochs=400, patience=30, device="cpu"):
        model = make_model(input_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)

        # Use a small validation split from the tail to enable early stopping while training on "full" data
        n = X_t.shape[0]
        val_count = max(1, int(0.1 * n))  # 10% for ES
        idx_perm = np.random.RandomState(42).permutation(n)
        tr_idx = idx_perm[:-val_count]
        va_idx = idx_perm[-val_count:]

        X_tr, Y_tr = X_t[tr_idx], Y_t[tr_idx]
        X_va, Y_va = X_t[va_idx], Y_t[va_idx]

        best_state = None
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_tr)
            loss = criterion(pred, Y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_va)
                val_loss = criterion(val_pred, Y_va).item()

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        weights_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_cnn.pth")
        meta_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_meta.json")

        torch.save(model.state_dict(), weights_path)
        with open(meta_path, "w") as f:
            json.dump({"input_size": int(input_size)}, f)

        # Report fit on all data
        model.eval()
        with torch.no_grad():
            full_pred = model(X_t).cpu().numpy()
        full_r2 = r2_score(Y_t.cpu().numpy(), full_pred)
        print(f"Saved: {weights_path}")
        print(f"Saved: {meta_path}")
        print(f"Full-data R2 (on same data; optimistic): {full_r2:.4f}")

    #### CHECKPOINT 4.0 ####

    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    # --- GPU SETUP ---
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Set seeds for reproducibility ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = instance_path / f'{geneID}_{window/(10**6)}Mb_gene_specific_probes.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models_dir = instance_path
    os.makedirs(models_dir, exist_ok=True)

    X, y, input_size = load_igf_like_csv(csv_path)

    mean_cv_r2 = cross_validate_and_report(
        X, y, input_size, n_splits=5, lr=1e-3, epochs=200, patience=20, device=device
    )

    train_full_and_save(X, y, input_size, models_dir, geneID,
                        lr=1e-3, epochs=400, patience=30, device=device)

def get_gradCAM(instance_path) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    models_dir = instance_path
    os.makedirs(models_dir, exist_ok=True)

    weights_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_cnn.pth")
    meta_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_meta.json")
    csv_path = instance_path / f'{geneID}_{window/(10**6)}Mb_gene_specific_probes.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    input_size = int(meta["input_size"])

    model = AdaptiveRegressionCNN(input_size=input_size).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    def load_igf_like_csv(path):
        df = pd.read_csv(path)
        df_samples = df.iloc[2:].reset_index(drop=True)
        df_samples_numeric = df_samples.apply(pd.to_numeric, errors="coerce")

        y = df_samples_numeric.iloc[:, 1].values.astype(np.float32)
        X = df_samples_numeric.iloc[:, 2:].values.astype(np.float32)
        return X, y

    X, y = load_igf_like_csv(csv_path)
    assert X.shape[1] == input_size, f"Input width {X.shape[1]} != meta input_size {input_size}"

    def find_last_conv1d(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv1d):
                last = m
        return last

    target_layer = find_last_conv1d(model)
    if target_layer is None:
        raise RuntimeError("No Conv1d layer found in AdaptiveRegressionCNN. Please set target_layer manually.")
    
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def gradcam_for_batch(model, x_batch, target_index=None):
        model.zero_grad()
        out = model(x_batch)                  # [N,1]
        if target_index is None:
            # scalar regression: sum over batch so grad flows to all
            target = out.sum()
        else:
            target = out[:, target_index].sum()
        target.backward()

        A = activations["value"]              # [N, C, L]
        dYdA = gradients["value"]             # [N, C, L]

        # Global-average-pool gradients over spatial dimension to get channel weights
        weights = dYdA.mean(dim=2, keepdim=True)   # [N, C, 1]
        cam = (weights * A).sum(dim=1)             # [N, L]
        cam = torch.relu(cam)

        # Normalize each sample CAM to [0,1]
        cam_min = cam.amin(dim=1, keepdim=True)
        cam_max = cam.amax(dim=1, keepdim=True).clamp(min=1e-12)
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
        return cam_norm  # [N, L]

    def batch_iter(arr, bs=128):
        n = arr.shape[0]
        for s in range(0, n, bs):
            e = min(n, s+bs)
            yield s, e

    all_cam_sum = np.zeros(input_size, dtype=np.float64)
    count = 0

    with torch.no_grad():
        pass  # keep model in eval

    for s, e in batch_iter(X, bs=256):
        xb = torch.tensor(X[s:e], dtype=torch.float32).unsqueeze(1).to(device)
        cam_b = gradcam_for_batch(model, xb).cpu().numpy()  # [B, L]
        all_cam_sum = all_cam_sum + cam_b.sum(axis=0)
        count += cam_b.shape[0]

    mean_cam = (all_cam_sum / max(1, count)).astype(np.float32)

    # Smooth optional (moving average) to reduce noise on 1D signal
    def smooth(x, k=5):
        if k <= 1: return x
        k = int(k)
        ker = np.ones(k, dtype=np.float32) / k
        return np.convolve(x, ker, mode="same")

    mean_cam_smooth = smooth(mean_cam, k=7)

    df_full = pd.read_csv(csv_path, nrows=1)   # header row
    probe_cols = df_full.columns.tolist()[2:2+input_size]  # skip first 3 columns
    cam_export = pd.DataFrame({
        "probe": probe_cols,
        "gradcam_mean": mean_cam,
        "gradcam_mean_smoothed": mean_cam_smooth
    })

    cam_export_path = instance_path / f"{geneID}_{window/(10**6)}Mb_gradCAM_importance.csv"
    cam_export.to_csv(cam_export_path, index=False)
    print(f"Wrote probe-level Grad-CAM importances to {cam_export_path}")

def plot_gradCAM(instance_path, use_dist, save_dist) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    cam_path = instance_path / f"{geneID}_{window/(10**6)}Mb_gradCAM_importance.csv"
    grad_data = pd.read_csv(cam_path)

    if use_dist :
        if getattr(sys, 'frozen', False):
            # Running as compiled .exe
            base_path = Path(sys.executable).parent
        else:
            # Running as normal script
            base_path = Path(__file__).resolve().parent.parent

        file_path = base_path / "data"

        meth_map_path = file_path / "probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
        meth_map = pd.read_csv(meth_map_path, sep='\t')
        meth_map = meth_map.drop(columns=['gene', 'chromEnd', 'strand'])
        meth_map = meth_map.set_index('#id')

        rel_map = meth_map.loc[grad_data['probe'].to_frame().T.iloc[0].to_list()]
        grad_data['chromStart'] = rel_map['chromStart'].values
        grad_data['chromStart'] = grad_data['chromStart'] - start
        plt.scatter(grad_data['chromStart'].values, grad_data['gradcam_mean'].values)
        plt.xlabel('Distance from gene')

        if save_dist :
            grad_data.to_csv(cam_path)
            print(f"Saved CpG distances to {cam_path}")
    else : 
        plt.scatter(range(len(grad_data)), grad_data['gradcam_mean'].values)
        plt.xlabel('CpG in order')
    
    plt.title(geneID + f'_{window/(10**6)}Mb')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.show()

def compare_weights(instance_path) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    cam_path = instance_path / f"{geneID}_{window/(10**6)}Mb_gradCAM_importance.csv"
    grad_data = pd.read_csv(cam_path, index_col=0)

    coef_df_path = instance_path / f'{geneID}_{window/(10**6)}Mb_gene_specific_elasticNet_weights.csv'
    coef_df = pd.read_csv(coef_df_path, index_col=0)

    x = range(0, len(coef_df))

    y1 = np.abs(coef_df['Coefficient'].values)
    y1 = (y1 - min(y1))/(max(y1) - min(y1))

    y2 = grad_data['gradcam_mean'].values
    y2[np.where(y2 < np.mean(y2))] = np.mean(y2)
    y2 = (y2 - min(y2))/(max(y2) - min(y2))

    fig, ax1 = plt.subplots()
    ax1.scatter(x, y1, color='g', s=10)
    ax1.set_xlabel('Distance from gene')
    ax1.set_ylabel('elasticNet', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.scatter(x, y2, color='b', s =10, alpha=0.2)
    ax2.set_ylabel('arCNN', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, 1)

    y3 = y2 - y1
    ax3 = ax1.twinx()
    ax3.scatter(x, y3, color='orange', s=10, alpha=0.2)
    ax3.set_ylabel('Error', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    print(f"Mean error is {np.mean(y3)}.")
    print(f"Fraction of probes with error > {0.2} is {len([x for x in np.abs(y3) if x > 0.2])/len(y3)}.")

    plt.show()