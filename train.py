import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from models.gin import MolecularGIN


# ==========================
# CREATE RESULTS FOLDER
# ==========================

os.makedirs("results", exist_ok=True)


# ==========================
# LOAD DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[:20000]


# ==========================
# TARGET NORMALIZATION
# ==========================

all_targets = dataset.y

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
    lr=0.001,
    weight_decay=1e-5
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ==========================
# TRACKING
# ==========================

best_val_loss = float("inf")

train_losses = []
val_losses = []


# ==========================
# TRAINING LOOP
# ==========================

for epoch in range(50):

    # ------------------
    # TRAINING
    # ------------------

    model.train()

    train_loss = 0.0

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

    val_loss = 0.0

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

    # Learning-rate scheduler
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # ------------------
    # SAVE BEST MODEL
    # ------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/molecular_gin_residual_best.pth"
        )

        print("Best model updated!")

    # ------------------
    # PRINT RESULTS
    # ------------------

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss = {train_loss:.4f} | "
        f"Val Loss = {val_loss:.4f} | "
        f"LR = {current_lr:.6f}"
    )


# ==========================
# SAVE FINAL MODEL
# ==========================

torch.save(
    model.state_dict(),
    "models/molecular_gin_residual.pth"
)

torch.save(
    {
        "mean": target_mean,
        "std": target_std
    },
    "models/normalization_stats.pth"
)


# ==========================
# SAVE TRAINING CURVE
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Residual GIN Training Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_curve.png"
)

plt.close()


# ==========================
# SAVE EXPERIMENT SUMMARY
# ==========================

summary = pd.DataFrame({
    "Model": ["ResidualGIN"],
    "Dataset_Size": [20000],
    "Targets": [19],
    "Best_Val_Loss": [best_val_loss]
})

summary.to_csv(
    "results/experiment_summary.csv",
    index=False
)

print("\nTraining completed!")
print(f"Best Validation Loss = {best_val_loss:.4f}")
print("Final model saved!")
print("Best model saved!")
print("Normalization stats saved!")
print("Training curve saved!")
print("Experiment summary saved!")