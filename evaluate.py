import torch

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gin import MolecularGIN


# ==========================
# LOAD DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[20000:22000]

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False
)


# ==========================
# LOAD NORMALIZATION STATS
# ==========================

stats = torch.load(
    "models/normalization_stats.pth",
    map_location=torch.device("cpu")
)

target_mean = stats["mean"]
target_std = stats["std"]


# ==========================
# LOAD MODEL
# ==========================

model = MolecularGIN()

model.load_state_dict(
    torch.load(
        "models/molecular_gin_multitask_best.pth",
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
           batch.z,
           batch.pos,
           batch.batch
         )

        # Convert predictions back
        # to original QM9 scale

        pred = (
            pred * target_std
        ) + target_mean

        target = batch.y

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