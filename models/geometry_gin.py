
import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


class MolecularGIN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # ==========================
        # GIN BLOCK 1
        # ==========================

        nn1 = Sequential(
            Linear(11, 128),
            ReLU(),
            Linear(128, 128)
        )

        # ==========================
        # GIN BLOCK 2
        # ==========================

        nn2 = Sequential(
            Linear(64, 64),
            ReLU(),
            Linear(64, 64)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        # ==========================
        # MLP HEAD
        # ==========================

        self.fc1 = Linear(64, 32)

        # 19 QM9 properties
        self.fc2 = Linear(32, 19)

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        # ==========================
        # GIN LAYER 1
        # ==========================

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        # ==========================
        # GIN LAYER 2
        # ==========================

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # ==========================
        # GRAPH POOLING
        # ==========================

        x = global_mean_pool(
            x,
            batch
        )

        # ==========================
        # MLP
        # ==========================

        x = F.relu(
            self.fc1(x)
        )

        # Dropout for regularization
        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.fc2(x)

        return x

