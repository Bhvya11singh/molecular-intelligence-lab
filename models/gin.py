import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


class MolecularGIN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        nn1 = Sequential(
            Linear(11, 64),
            ReLU(),
            Linear(64, 64)
        )

        nn2 = Sequential(
            Linear(64, 64),
            ReLU(),
            Linear(64, 64)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 1)

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x