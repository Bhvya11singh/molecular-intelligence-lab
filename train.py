import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from models.gin import MolecularGIN


# ==========================
# LOAD DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[:20000]


# ==========================
# TARGET NORMALIZATION
# ==========================

all_targets = dataset.data.y

target_mean = all_targets.mean(dim=0)
target_std = all_targets.std(dim=0)

print("Target Mean Shape:", target_mean.shape)
print("Target Std Shape:", target_std.shape)


# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================

train_dataset, val_dataset = random_split(
    subset,
    [16000, 4000],
    generator=torch.Generator().manual_seed(42)
)


# ==========================
# DATALOADERS
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================
# MODEL
# ==========================

model = MolecularGIN()

optimizer = Adam(
    model.parameters(),
    lr=0.001
)


# ==========================
# TRACK BEST MODEL
# ==========================

best_val_loss = float("inf")


# ==========================
# TRAINING LOOP
# ==========================

for epoch in range(50):

    # ------------------
    # TRAINING
    # ------------------

    model.train()

    train_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        target = (
            batch.y - target_mean
        ) / target_std

        loss = F.mse_loss(
            pred,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ------------------
    # VALIDATION
    # ------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            pred = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            target = (
                batch.y - target_mean
            ) / target_std

            loss = F.mse_loss(
                pred,
                target
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # ------------------
    # SAVE BEST MODEL
    # ------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/molecular_gin_multitask_best.pth"
        )

        print("Best model updated!")

    # ------------------
    # PRINT RESULTS
    # ------------------

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss = {train_loss:.4f} | "
        f"Val Loss = {val_loss:.4f}"
    )


# ==========================
# SAVE FINAL MODEL
# ==========================

torch.save(
    model.state_dict(),
    "models/molecular_gin_multitask.pth"
)

torch.save(
    {
        "mean": target_mean,
        "std": target_std
    },
    "models/normalization_stats.pth"
)

print("\nTraining completed!")
print(f"Best Validation Loss = {best_val_loss:.4f}")
print("Final model saved!")
print("Best model saved!")
print("Normalization stats saved!")