import torch
import torch.nn.functional as F

from torch.nn import (
    Linear,
    Sequential,
    ReLU,
    BatchNorm1d,
    Dropout
)

from torch_geometric.nn import (
    GINConv,
    global_mean_pool
)


class MolecularGIN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        hidden_dim = 128

        # ==========================
        # GIN BLOCK 1
        # ==========================

        nn1 = Sequential(
            Linear(11, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden_dim)

        # ==========================
        # GIN BLOCK 2
        # ==========================

        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(hidden_dim)

        # ==========================
        # GIN BLOCK 3
        # ==========================

        nn3 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # ==========================
        # DROPOUT
        # ==========================

        self.dropout = Dropout(p=0.3)

        # ==========================
        # MLP HEAD
        # ==========================

        self.fc1 = Linear(hidden_dim, 64)
        self.fc2 = Linear(64, 19)

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        # ==========================
        # LAYER 1
        # ==========================

        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        # ==========================
        # LAYER 2 + RESIDUAL
        # ==========================

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)

        # Residual connection
        x2 = x2 + x1

        x2 = F.relu(x2)

        # ==========================
        # LAYER 3 + RESIDUAL
        # ==========================

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)

        # Residual connection
        x3 = x3 + x2

        x3 = F.relu(x3)

        # ==========================
        # GLOBAL POOLING
        # ==========================

        x = global_mean_pool(
            x3,
            batch
        )

        # ==========================
        # MLP HEAD
        # ==========================

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x