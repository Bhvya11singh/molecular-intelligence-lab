import torch
import pandas as pd

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gin import MolecularGIN


# ==========================
# PROPERTY NAMES
# ==========================

property_names = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
    "a",
    "b",
    "c"
]


# ==========================
# DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[5000:6000]

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False
)


# ==========================
# NORMALIZATION STATS
# ==========================

stats = torch.load(
    "models/normalization_stats.pth",
    map_location="cpu"
)

target_mean = stats["mean"]
target_std = stats["std"]


# ==========================
# MODEL
# ==========================

model = MolecularGIN()

model.load_state_dict(
    torch.load(
        "models/molecular_gin_multitask_best.pth",
        map_location="cpu"
    )
)

model.eval()


# ==========================
# PROPERTY-WISE MAE
# ==========================

property_mae = torch.zeros(19)
num_batches = 0

with torch.no_grad():

    for batch in loader:

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        pred = (
            pred * target_std
        ) + target_mean

        target = batch.y

        mae = torch.mean(
            torch.abs(pred - target),
            dim=0
        )

        property_mae += mae
        num_batches += 1


property_mae /= num_batches


# ==========================
# SAVE RESULTS
# ==========================

results = pd.DataFrame({
    "Property": property_names,
    "MAE": property_mae.numpy()
})

results.to_csv(
    "results/multitask_metrics_20k.csv",
    index=False
)

print(results)
print("\nSaved to results/multitask_metrics_20k.csv")