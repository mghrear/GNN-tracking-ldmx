import pandas as pd
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

# Adapted from https://github.com/GageDeZoort/interaction_network_paper
class INLayer(MessagePassing):
    def __init__(self, hidden_size=40, node_feat_in = 3, edge_feat_in = 3, node_feat_out=3, edge_feat_out=3):
        super(INLayer, self).__init__(aggr='add', 
                                                 flow='source_to_target')
        self.R1 = RelationalModel(2*node_feat_in+edge_feat_in, edge_feat_out, hidden_size)
        self.O = ObjectModel(node_feat_in+edge_feat_out, node_feat_out, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)
    
class INLayer_Edep(MessagePassing):
    def __init__(self, hidden_size=40, node_feat_in = 3, edge_feat_in = 3, node_feat_out=3, edge_feat_out=3):
        super(INLayer_Edep, self).__init__(aggr='add', 
                                                 flow='source_to_target')
        self.R1 = RelationalModel(2*node_feat_in+edge_feat_in, edge_feat_out, hidden_size)
        self.O = ObjectModel(node_feat_in+edge_feat_out+1, node_feat_out, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, Edep: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x_tilde = self.propagate(edge_index, x=x, Edep=Edep, edge_attr=edge_attr, size=None)
        return x_tilde, self.E

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x, Edep):

        c = torch.cat([x, Edep, aggr_out], dim=1)
        return self.O(c)
    
class MyIN(nn.Module):
    def __init__(self, hidden_size):
        super(MyIN, self).__init__()

        self.IN1 = INLayer(hidden_size=hidden_size, node_feat_in = 3, edge_feat_in = 3, node_feat_out=6, edge_feat_out=6)
        self.IN2 = INLayer(hidden_size=hidden_size, node_feat_in = 6, edge_feat_in = 6, node_feat_out=8, edge_feat_out=8)
        self.IN3 = INLayer(hidden_size=hidden_size, node_feat_in = 8, edge_feat_in = 8, node_feat_out=9, edge_feat_out=9)
        self.R2 = RelationalModel(2*9+9, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):

        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                        x[edge_index[0]],
                        edge_attr], dim=1)

        return torch.sigmoid(self.R2(edge_final))
    

class MyIN_Edep(nn.Module):
    def __init__(self, hidden_size):
        super(MyIN_Edep, self).__init__()

        self.IN1 = INLayer_Edep(hidden_size=hidden_size, node_feat_in = 3, edge_feat_in = 3, node_feat_out=6, edge_feat_out=6)
        self.IN2 = INLayer_Edep(hidden_size=hidden_size, node_feat_in = 6, edge_feat_in = 6, node_feat_out=8, edge_feat_out=8)
        self.IN3 = INLayer_Edep(hidden_size=hidden_size, node_feat_in = 8, edge_feat_in = 8, node_feat_out=9, edge_feat_out=9)
        self.R2 = RelationalModel(2*9+9, 1, hidden_size)

    def forward(self, x: Tensor, Edep: Tensor, edge_index: Tensor, edge_attr: Tensor):

        x, edge_attr = self.IN1(x, Edep, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, Edep, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, Edep, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                        x[edge_index[0]],
                        edge_attr], dim=1)

        return torch.sigmoid(self.R2(edge_final))
    