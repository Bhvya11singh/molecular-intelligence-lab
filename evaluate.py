import torch
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gcn import MolecularGCN


dataset = QM9(root="data/QM9")

subset = dataset[5000:6000]

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False
)

model = MolecularGCN()

model.load_state_dict(
    torch.load("models/molecular_gcn.pth")
)

model.eval()

total_mae = 0
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

print(
    f"Average MAE = {total_mae / num_batches:.4f}"
)