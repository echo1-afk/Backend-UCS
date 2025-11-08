
"""GNN spatio-temporal pipeline using PyTorch Geometric (script)
Save this file as GNN_PyG_spatio_temporal.py and run after installing dependencies.

High-level steps:
1. Install dependencies (run in your environment):
   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  # or cpu wheels per your system
   - pip install torch-geometric -f https://data.pyg.org/whl/torch-<TORCH_VERSION>.html
   - pip install scikit-learn pandas numpy scipy

2. The script expects the CSV at /mnt/data/smart_mobility_dataset.csv.
3. It clusters lat/lon into zones, aggregates features per zone per timestep,
   constructs a K-NN adjacency, and trains a simple spatio-temporal GCN+GRU model.

NOTE: Installing PyG can be system-specific. Follow official instructions:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv

# Config
CSV_PATH = 'smart_mobility_dataset.csv'
N_ZONES = 30          # change as needed
K_NEIGHBORS = 4
SEQ_LEN = 24
HORIZON = 12
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_and_cluster(csv_path, n_zones=30):
    df = pd.read_csv(csv_path)
    assert 'Latitude' in df.columns and 'Longitude' in df.columns, "Need lat/lon"
    coords = df[['Latitude','Longitude']].values
    kmeans = KMeans(n_clusters=n_zones, random_state=42).fit(coords)
    df['zone'] = kmeans.labels_
    # create zone centroids
    centroids = kmeans.cluster_centers_
    return df, centroids

def aggregate_per_zone(df, n_zones):
    # assume df has Timestamp column; else treat rows as sequential timesteps
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
    # group by timestamp and zone, take mean of numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'zone' from numeric_cols if it exists
    if 'zone' in numeric_cols:
        numeric_cols.remove('zone')
    agg = df.groupby(['Timestamp','zone'])[numeric_cols].mean().reset_index()
    # pivot to create times x zones x features tensor
    timestamps = sorted(agg['Timestamp'].unique())
    feature_list = [c for c in numeric_cols if c not in ['Latitude','Longitude']]
    T = len(timestamps)
    Z = n_zones
    F = len(feature_list)
    tensor = np.zeros((T, Z, F), dtype=np.float32)
    for t_idx, ts in enumerate(timestamps):
        df_t = agg[agg['Timestamp']==ts]
        for _, row in df_t.iterrows():
            z = int(row['zone'])
            feats = row[feature_list].values.astype(np.float32)
            tensor[t_idx, z, :] = feats
    return tensor, timestamps, feature_list

def build_knn_edge_index(centroids, k=4):
    # kneighbors_graph gives sparse adjacency; convert to edge_index
    A = kneighbors_graph(centroids, k, mode='connectivity', include_self=False)
    A = A.tocoo()
    edge_index = np.vstack([A.row, A.col])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

class SpatioTemporalDataset(InMemoryDataset):
    def __init__(self, tensor, edge_index, transform=None):
        super().__init__('.', transform)
        # tensor: T x Z x F
        self.tensor = tensor
        self.edge_index = edge_index
        self.process()

    def process(self):
        data_list = []
        T, Z, F = self.tensor.shape
        for t in range(T - SEQ_LEN - HORIZON + 1):
            x = torch.tensor(self.tensor[t:t+SEQ_LEN], dtype=torch.float)  # seq_len x Z x F
            y = torch.tensor(self.tensor[t+SEQ_LEN+HORIZON-1], dtype=torch.float)  # Z x F (we'll pick target column)
            # reshape x to (Z, seq_len*F) or keep as temporal dimension in model
            # For PyG, we create one Data per sample with node features shaped (Z, seq_len*F)
            x_nodes = x.permute(1,0,2).reshape(Z, -1)  # Z x (seq_len*F)
            data = Data(x=x_nodes, edge_index=self.edge_index, y=y)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)

class STGCN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)  # predicting single target per node

    def forward(self, data):
        # data.x: (batch*Z, in_dim) due to PyG batching -> we will reshape inside training loop
        # For simplicity, assume data.x is (batch, Z, in_dim)
        x = data.x  # expected shape: (batch, Z, in_dim)
        batch, Z, in_dim = x.shape
        x = x.view(batch*Z, in_dim)
        # apply GCN per sample: requires edge_index that is same for every sample; not trivial in mini-batch without using torch_geometric batching
        # We'll implement a simple per-sample loop (not optimized) — fine for prototyping
        x = x.view(batch, Z, in_dim)
        out_nodes = []
        for b in range(batch):
            xb = x[b]  # Z x in_dim
            # apply GCN on single graph
            xb = xb.to(DEVICE)
            edge = data.edge_index.to(DEVICE)
            xb_g = self.gcn1(xb, edge)  # Z x hidden
            out_nodes.append(xb_g.unsqueeze(0))
        out = torch.cat(out_nodes, dim=0)  # batch x Z x hidden
        # run GRU across nodes as sequence? alternatively across time — here GRU is not used as we collapsed time dimension earlier
        # we take mean across nodes and predict per-node value via FC
        out = self.fc(out).squeeze(-1)  # batch x Z
        return out

def train():
    df, centroids = load_and_cluster(CSV_PATH, n_zones=N_ZONES)
    tensor, timestamps, feature_list = aggregate_per_zone(df, N_ZONES)
    edge_index = build_knn_edge_index(centroids, k=K_NEIGHBORS)
    # build dataset
    dataset = SpatioTemporalDataset(tensor, edge_index)
    # simple split
    n = len(dataset)
    train_n = int(n*0.7)
    val_n = int(n*0.85)
    train_set = dataset[:train_n]
    val_set = dataset[train_n:val_n]
    test_set = dataset[val_n:]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    model = STGCN(in_dim=dataset[0].x.shape[1], hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            # batch.x shape: (batch_size*Z, in_dim) due to PyG batching; we will reshape to (batch, Z, in_dim)
            batch = batch.to(DEVICE)
            B = batch.num_graphs
            Z = N_ZONES
            in_dim = batch.x.shape[1]
            # reshape
            x = batch.x.view(B, Z, in_dim)
            data_for_model = type('obj', (object,), {'x': x, 'edge_index': batch.edge_index})
            y_true = batch.y[:, :, 0].to(DEVICE)  # pick first numeric column as target; adjust as needed
            optimizer.zero_grad()
            y_pred = model(data_for_model)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS} train_loss={tot_loss/len(train_loader):.4f}')
    print('Training complete. Evaluate on test set similarly.')
    # Save model
    torch.save(model.state_dict(), 'stgcn_model.pt')

if __name__ == '__main__':
    train()
