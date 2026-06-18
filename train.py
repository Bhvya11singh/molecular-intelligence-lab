import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gcn import MolecularGCN


# Load dataset
dataset = QM9(root="data/QM9")

subset = dataset[:5000]

# Train / Validation split
from torch.utils.data import random_split

train_dataset, val_dataset = random_split(
    subset,
    [4000, 1000],
    generator=torch.Generator().manual_seed(42)
)
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

# Model
model = MolecularGCN()

optimizer = Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
for epoch in range(10):

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

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss = {train_loss:.4f} | "
        f"Val Loss = {val_loss:.4f}"
    )


# ------------------
# SAVE MODEL
# ------------------

torch.save(
    model.state_dict(),
    "models/molecular_gcn.pth"
)

print("Model saved successfully!")