import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch as T
import os
import numpy as np
from torch_geometric.nn import GATConv
from torch.nn import Linear


class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        # Building INPUT
        self.INPUT = INPUT
        # Defining base variables
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]  
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1],self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index):  # forward propagation includes defining layers         
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        return self.linear(out)

    def save_checkpoint(self):
        print(f'Saving {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'Loading {self.CHECKPOINT_FILE}...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))
