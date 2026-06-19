import torch

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gcn import MolecularGCN


# ==========================
# LOAD DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[5000:6000]

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False
)


# ==========================
# LOAD MODEL
# ==========================

model = MolecularGCN()

model.load_state_dict(
    torch.load(
        "models/molecular_gcn_best.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Model loaded successfully!")


# ==========================
# EVALUATION
# ==========================

total_mae = 0.0
num_batches = 0

with torch.no_grad():

    for batch in loader:

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        target = batch.y[:, 0].view(-1, 1)

        mae = torch.mean(
            torch.abs(pred - target)
        )

        total_mae += mae.item()
        num_batches += 1


# ==========================
# RESULTS
# ==========================

final_mae = total_mae / num_batches

print("\n==========================")
print("Evaluation Results")
print("==========================")
print(f"Average MAE = {final_mae:.4f}")