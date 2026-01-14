import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import scatter






class MyInMemoryDataset(InMemoryDataset):

    # In EC mode the dataset includes a feature tensor (x), edge_index where edges connect space points in adjacent layers, and labels (y) indicating whether the edge connects true space points from the same track.
    # For EC model and additional argument is needed to specify the number of primary electrons per event (n_primaries). This is only needed to build edge labels for training. 
    # In OC mode the data includes a feature tensor (x) and labels (y) indicating the true track ID for each space point.

    def __init__(self, df, n_primaries = None, mode = "EC"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.n_primaries = n_primaries

    def len(self):
        return len(self.df)

    def get(self, idx):

        # Build the feature tensor
        # Shape [num_nodes, 3]
        x = torch.Tensor([self.df.Digi_x[idx], self.df.Digi_y[idx], self.df.Digi_z[idx]]).T

        if self.mode == "EC":
            v_pos = torch.Tensor(self.df.iloc[idx].Digi_x)


            # Make edges connect space point in subsequent layers only
            Ta = torch.abs(v_pos[:, None] - v_pos[None, :]) < 95.0
            Tb = torch.abs(v_pos[:, None] - v_pos[None, :]) > 5.0
            T = Ta*Tb

            i_idx, j_idx = T.nonzero(as_tuple=True)
            edge_index = torch.stack((i_idx, j_idx), dim=0)    # shape (2, n_edges)

            # Build edge features as vector distance between connected nodes
            src = edge_index[0]
            dst = edge_index[1]
            edge_feat = x[dst] - x[src]  # [E, F]

            # Find edges that connect space points in subsequent layers from the same track
            vID = torch.Tensor(self.df.iloc[idx].Digi_ID)
            TID = torch.abs(vID[:, None] - vID[None, :]) == 0.0
            T_truth = TID * T

            # Remove edges connecting non-primaries
            NopT = vID > self.n_primaries
            T_truth[:,NopT] = False
            T_truth[NopT,:] = False

            iT, jT = T_truth.nonzero(as_tuple=True)
            edge_index_truth = torch.stack((iT, jT), dim=0)    # shape (2, n_edges)

            y = (edge_index.T[:, None, :] == edge_index_truth.T[None, :, :]).all(dim=2).any(dim=1).float()

            return Data(x=x, edge_index=edge_index, edge_attr=edge_feat, y=y)


        elif self.mode == "OC":
            y = torch.Tensor(self.df.iloc[idx].Digi_ID)

            return Data(x=x, y=y)
        
# from https://github.com/GageDeZoort/interaction_network_paper
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


# from https://github.com/GageDeZoort/interaction_network_paper
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

# from https://github.com/GageDeZoort/interaction_network_paper
class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_size):
        super(InteractionNetwork, self).__init__(aggr='add', 
                                                 flow='source_to_target')
        self.R1 = RelationalModel(9, 4, hidden_size)
        self.O = ObjectModel(7, 3, hidden_size)
        self.R2 = RelationalModel(10, 1, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing        

        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)
    

class INLayer(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, aggr='sum'):
        super().__init__()
        self.fR = MLP(2*node_dim + edge_dim, edge_dim, hidden)  # edge -> edge
        self.fO = MLP(node_dim + edge_dim, node_dim, hidden)    # node -> node
        self.aggr = aggr

    def forward(self, x, edge_index, e):
        src, dst = edge_index  # src=j, dst=i (messages go src->dst)

        # edge update: e'_{j->i}
        m_in = torch.cat([x[dst], x[src], e], dim=-1)
        e_new = self.fR(m_in)                      # [E, edge_dim]

        # aggregate to nodes: sum over incoming edges
        agg = scatter(e_new, dst, dim=0, dim_size=x.size(0), reduce='sum')  # [N, edge_dim]

        # node update
        x_new = self.fO(torch.cat([x, agg], dim=-1))  # [N, node_dim]

        return x_new, e_new
    


def plot_pyg_graph_3d(data, plot_truth=False, node_size=30, lw=0.8):
    # Node positions (N,3)
    pos = data.x[:, :3].detach().cpu()
    edge_index = data.edge_index.detach().cpu()


    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # nodes
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=node_size)

    # edges
    for e in range(edge_index.size(1)):
        i, j = edge_index[:, e]
        xs = [pos[i,0], pos[j,0]]
        ys = [pos[i,1], pos[j,1]]
        zs = [pos[i,2], pos[j,2]]


        ax.plot(xs, ys, zs, color="k", linewidth=lw, alpha=0.35)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

    if plot_truth:
        # Node positions (N,3)
        pos = data.x[:, :3].detach().cpu()
        edge_index = data.edge_index[:,data.y.bool()].detach().cpu()


        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # nodes
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=node_size)

        # edges
        for e in range(edge_index.size(1)):
            i, j = edge_index[:, e]
            xs = [pos[i,0], pos[j,0]]
            ys = [pos[i,1], pos[j,1]]
            zs = [pos[i,2], pos[j,2]]


            ax.plot(xs, ys, zs, color="k", linewidth=lw, alpha=0.35)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        plt.tight_layout()
        plt.show()


