
Files created in /mnt/data:

1) Capstone_LSTM_CNN-GRU_Notebook.ipynb
   - Jupyter notebook with data loading, preprocessing, sliding-window creation,
     LSTM and CNN-GRU model definitions, and evaluation utilities.
   - Configure TARGET, seq_len, horizon near the top of the notebook.
   - Run in an environment with tensorflow installed.

2) GNN_PyG_spatio_temporal.py
   - Prototype PyTorch Geometric script to cluster lat/lon into zones, build K-NN graph,
     prepare spatio-temporal dataset, and train a simple GCN+GRU model.
   - Installing PyG can be system-specific; follow official instructions.
   - Edit N_ZONES, SEQ_LEN, HORIZON at top of script to suit your use-case.

Quick setup commands (example):

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install core libs for notebook
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib jupyterlab

# For TensorFlow models:
pip install tensorflow

# For PyTorch Geometric (example for CPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then follow PyG install guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Run notebook:
jupyter lab /mnt/data/Capstone_LSTM_CNN-GRU_Notebook.ipynb

Run GNN script (after PyG installed):
python /mnt/data/GNN_PyG_spatio_temporal.py

Notes on RMSE target:
- The notebook trains on standardized target and inverts predictions for RMSE reporting in original units.
- Your original requirement RMSE <= 1.12 is likely referring to a scaled metric. Check whether that's in original units or normalized units. Use NRMSE (RMSE divided by target range or std) for more interpretable comparisons across targets.

