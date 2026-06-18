import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class MolecularGCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(11, 64)
        self.conv2 = GCNConv(64, 64)

        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 1)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x