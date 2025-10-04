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
        out += residual
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

def calc_imp_cpgs(instance_path, name, dir) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    #### CHECKPOINT 1.0 ####

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

    # --- Cross-validation training ---
    def cnn_cross_validation(Dmodel, x, y, patience, epochs, device, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        r2_scores_per_fold = []

        for train_idx, val_idx in kf.split(x):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

            model = Dmodel
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            count = 0

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_pred_val = model(x_val)
                    val_loss = criterion(y_pred_val, y_val)
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        count = 0
                        best_model_state = model.state_dict()
                    else:
                        count += 1
                        if count >= patience:
                            model.load_state_dict(best_model_state)
                            break

            model.eval()
            with torch.no_grad():
                y_pred_val = model(x_val)
                r2 = r2_score(y_val.cpu().numpy(), y_pred_val.cpu().numpy())
                r2_scores_per_fold.append(r2)

            del model, optimizer, criterion, x_train, x_val, y_train, y_val, y_pred_val
            torch.cuda.empty_cache()
            gc.collect()

        return np.mean(r2_scores_per_fold)
    
    def load_gene_model(gene_name, models_dir, device="cpu"):
        # --- Load metadata first ---
        meta_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata for gene {gene_name} not found at {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        input_size = meta["input_size"]

        # --- Rebuild model architecture ---
        model = AdaptiveRegressionCNN(input_size=input_size)

        # --- Load weights ---
        model_path = os.path.join(models_dir, f"{geneID}_{window/(10**6)}Mb_arCNN_cnn.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights for gene {gene_name} not found at {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # --- Ready for inference ---
        model.to(device)
        model.eval()
        return model, input_size
    
    #### CHECKPOINT 2.0 ####

    cpgs_path = instance_path / f'{geneID}_{window/(10**6)}Mb_gene_specific_elasticNet_weights.csv'
    cpgs = pd.read_csv(cpgs_path, index_col=0)

    cam_path = instance_path / f"{geneID}_{window/(10**6)}Mb_gradCAM_importance.csv"
    grad_data = pd.read_csv(cam_path)

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

    rel_map = meth_map.loc[cpgs['CpG'].to_frame().T.iloc[0].to_list()]
    cpgs['chromStart'] = rel_map['chromStart'].values
    cpgs['chromStart'] = cpgs['chromStart'] - start

    model, input_size = load_gene_model(geneID, models_dir=instance_path, device="cpu")
    print(f"arCNN loaded with input size {input_size}")

    #### CHECKPOINT 3.0 ####

    gene_data = pd.read_csv(instance_path / f'{geneID}_{window/(10**6)}Mb_gene_specific_probes.csv', index_col=0)
    print(gene_data.shape)

    gene_data = gene_data.iloc[1:]
    x_vals = gene_data.drop(geneID, axis=1)
    y_vals = gene_data[geneID]

    x_avg = x_vals.mean()
    print(f"Min gene expression : {min(y_vals)} | Max gene expression : {max(y_vals)}")

    if name == 'standard_sequential' or name == 'ss':
        cpgs['Weights'] = range(input_size)
        cpgs['Index'] = range(input_size)
        name = 'standard_sequential'
    elif name == 'circular_sequential' or name == 'cs' :
        cpgs['Weights'] = np.abs(cpgs['chromStart'].values)
        cpgs['Index'] = np.flip(np.argsort(cpgs["Weights"].values))
        name = 'circular_sequential'
    elif name == 'gradCAM_sequential' or name == 'gs' :
        cpgs['Weights'] = grad_data['gradcam_mean']
        cpgs['Index'] = np.flip(np.argsort(cpgs["Weights"].values))
        name = 'gradCAM_sequential'
    elif name == 'elasticNet_sequential' or name == 'es' :
        cpgs['Weights'] = cpgs['Coefficient']
        cpgs['Index'] = np.flip(np.argsort(cpgs["Weights"].values))
        name = 'elasticNet_sequential'
    else :
        print(f"CRITICAL ERROR : Algorithm '{name}' not defined")
        return

    px_vals = range(input_size)
    py_vals = np.zeros(input_size)

    x_cur = np.array(x_avg, dtype=np.float32)
    x_pred = torch.tensor(x_cur.reshape([1, input_size])).unsqueeze(1).to("cpu")
    with torch.no_grad() :
        print(f"Predicted value for standard input is {model(x_pred).cpu().numpy().item()}")

    count = 0
    if dir == 1 :
        delta = 1
        iter = cpgs['Index'].values
    elif dir == -1 :
        delta = -1
        iter = np.flip(cpgs["Index"].values)
    else:
        print(f"CRITICAL ERROR : Direction '{dir}' is not defined")
        return
    
    for i in iter :
        if (cpgs['Coefficient'].values[i] - 0)*delta > 0 :
            x_cur[i] = 1

        x_pred = torch.tensor(x_cur.reshape([1, input_size])).unsqueeze(1).to("cpu")
        with torch.no_grad() :
            py_vals[count] = model(x_pred).cpu().numpy().item()
        count += 1

    output = cpgs.loc[iter]
    output['Gene_expression'] = py_vals
    output = output.drop(columns=['Weights', 'Index'])
    output_name = f'{geneID}_{window/(10**6)}Mb_imp_cpgs_{name}_{dir}.csv'
    output.to_csv(instance_path / output_name, index = False)
    print(f"Exported ordered cpgs list to {output_name}")

    plt.plot(px_vals, py_vals)
    plt.title(geneID + f'_{window/(10**6)}Mb | Algorithm : ' + name)
    plt.xlabel('CpG site')
    plt.ylabel('Gene Expression')
    plt.show()
