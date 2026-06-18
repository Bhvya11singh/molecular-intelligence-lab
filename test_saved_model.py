import torch

from models.gcn import MolecularGCN

model = MolecularGCN()

model.load_state_dict(
    torch.load("models/molecular_gcn.pth")
)

model.eval()

print("Model loaded successfully!")