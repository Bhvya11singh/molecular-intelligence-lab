import torch
from torch.nn import Linear
from torch_geometric.nn import SchNet


class MolecularSchNet(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # SchNet backbone

        self.schnet = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0
        )

        # Final prediction head

        self.fc = Linear(128, 19)

    def forward(self, z, pos, batch):

        x = self.schnet(
            z=z,
            pos=pos,
            batch=batch
        )

        x = self.fc(x)

        return x