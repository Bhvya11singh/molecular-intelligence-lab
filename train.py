import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from models.gcn import MolecularGCN


# ==========================
# LOAD DATASET
# ==========================

dataset = QM9(root="data/QM9")

subset = dataset[:5000]


# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================

train_dataset, val_dataset = random_split(
    subset,
    [4000, 1000],
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

model = MolecularGCN()

optimizer = Adam(
    model.parameters(),
    lr=0.001
)


# ==========================
# TRACK BEST MODEL
# ==========================

best_val_loss = float("inf")
best_train_loss = None


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

        target = batch.y[:, 0].view(-1, 1)

        loss = F.mse_loss(
            pred,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

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

            target = batch.y[:, 0].view(-1, 1)

            loss = F.mse_loss(
                pred,
                target
            )

            val_loss += loss.item()

    # ------------------
    # SAVE BEST MODEL
    # ------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_train_loss = train_loss

        torch.save(
            model.state_dict(),
            "models/molecular_gcn_best.pth"
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
    "models/molecular_gcn.pth"
)

print("\nTraining completed!")
print(f"Best Validation Loss = {best_val_loss:.4f}")
print("Final model saved!")
print("Best model saved!")