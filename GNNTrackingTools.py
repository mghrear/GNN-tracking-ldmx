from time import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect


class MyInMemoryDataset(InMemoryDataset):

    # In EC mode the dataset includes a feature tensor (x), edge_index where edges connect space points in adjacent layers, and labels (y) indicating whether the edge connects true space points from the same track.
    # In OC mode the data includes a feature tensor (x) and labels (y) indicating the true track ID for each space point.

    def __init__(self, df, mode = "EC"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def len(self):
        return len(self.df)

    def get(self, idx):

        # Build the feature tensor
        # Shape [num_nodes, 3]
        x = torch.Tensor([self.df.Digi_x[idx], self.df.Digi_y[idx], self.df.Digi_z[idx]]).T

        if self.mode == "EC":

            edge_index = torch.stack([ torch.Tensor(self.df.edge_index_0[idx]),torch.Tensor(self.df.edge_index_1[idx])]).to(torch.int)
            edge_feat = torch.stack([torch.Tensor(self.df.edge_feat_0[idx]),torch.Tensor(self.df.edge_feat_1[idx]),torch.Tensor(self.df.edge_feat_2[idx])]).T
            y = torch.Tensor(self.df.edge_label[idx])

            return Data(x=x, edge_index=edge_index, edge_attr=edge_feat, truth_ID = torch.Tensor(self.df.iloc[idx].Digi_trackID), pdg_ID = torch.Tensor(self.df.iloc[idx].Digi_pdgID), Edep = torch.Tensor(self.df.iloc[idx].Digi_Edep).unsqueeze(1), y=y)


        elif self.mode == "OC":
            y = torch.Tensor(self.df.iloc[idx].Digi_trackID)

            return Data(x=x, y=y)
        

def plot_pyg_graph_3d(data, plot_truth=False, plot_pred=False, node_size=30, lw=0.8):
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

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
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

    if plot_pred:
        # Node positions (N,3)
        pos = data.x[:, :3].detach().cpu()
        edge_index = data.edge_index[:,data.pred.bool()].detach().cpu()


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


# Processes pandas dataframe to get graph information
# n_primaries is the number of primaries simulated, this is only needed for label building 
def GetGraphInfo(row, n_primaries, tracker = "Tagger"):

    x = np.stack([np.asarray(row[f'{tracker}_Digi_x']),
                np.asarray(row[f'{tracker}_Digi_y']),
                np.asarray(row[f'{tracker}_Digi_z'])], axis=1)

    v_pos = np.asarray(row[f'{tracker}_Digi_x'])   # shape (N,)

    # Make edges connect space point in subsequent layers only
    D = np.abs(v_pos[:, None] - v_pos[None, :])   # shape (N, N)
    T = (D < 110.0) & (D > 5.0)     

    # indices where T is True
    i_idx, j_idx = np.nonzero(T)                  # each shape (n_edges,)
    edge_index = np.stack((i_idx, j_idx), axis=0) # shape (2, n_edges)

    # Build edge features as vector distance between connected nodes
    src = edge_index[0]
    dst = edge_index[1]
    edge_feat = x[dst] - x[src]  # [E, F]

    # Find edges that connect space points in subsequent layers from the same track
    vID = np.asarray(row[f'{tracker}_Digi_trackID'])
    TID = np.abs(vID[:, None] - vID[None, :]) == 0.0
    T_truth = TID * T 

    # Remove edges connecting non-primaries
    NopT = vID > n_primaries
    T_truth[:,NopT] = False
    T_truth[NopT,:] = False

    iT, jT = np.nonzero(T_truth)
    edge_index_truth = np.stack((iT, jT), axis=0)    # shape (2, n_edges)

    edge_label = (edge_index.T[:, None, :] == edge_index_truth.T[None, :, :]).all(axis=2).any(axis=1).astype(np.float32)

    return edge_index[0], edge_index[1], edge_feat.T[0], edge_feat.T[1], edge_feat.T[2], edge_label 


def train(model, device, train_loader, optimizer, epoch):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        if Nparams == 4:
            output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
        else:
            output = model(data.x, data.edge_index, data.edge_attr)

        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx, len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        losses.append(loss.item())

    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)



def validate(model, device, val_loader):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)
    
    model.eval()
    losses = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        if Nparams == 4:
            output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
        else:
            output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        losses.append(loss)

    print("...val loss=", np.mean(losses))

    return np.mean(losses)


def test(model, device, test_loader, thld=0.5):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    model.eval()
    losses, accs, TPRs, TNRs = [], [], [], []
    labels, preds = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            if Nparams == 4:
                output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
            else:
                output = model(data.x, data.edge_index, data.edge_attr)
                
            TP = torch.sum((data.y==1).squeeze() & 
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() & 
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (output<thld).squeeze()).item()            
            acc = (TP+TN)/(TP+TN+FP+FN)
            TPR = TP / (TP+FN)
            TNR = TN / (TN+FP)

            loss = F.binary_cross_entropy(output.squeeze(1), data.y, 
                                          reduction='mean').item()
            accs.append(acc)
            TPRs.append(TPR)
            TNRs.append(TNR)
            losses.append(loss)
            
            labels += data.y.cpu().tolist()
            preds += output.squeeze().cpu().tolist()

    return np.mean(losses), np.mean(accs), np.mean(TPRs), np.mean(TNRs), labels, preds 


def PrimitiveTrackBuilder(model, device, test_loader, thld=0.5, min_nodes = 5, use_truth_labels = False):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    df_tracks = pd.DataFrame(columns=['x', 'y', 'z'])

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            # Move datat to device 
            data = data.to(device)

            # GNN inference
            if Nparams == 4:
                output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
            else:
                output = model(data.x, data.edge_index, data.edge_attr)

            if use_truth_labels:
                # This method cheats, using truth info
                y_pred = data.y.bool()
            else: 
                # Extract prediction from GNN output
                y_pred = output.squeeze()>thld 

            data.edge_index = data.edge_index[:,y_pred]

            for subgraph in data.connected_components():

                if subgraph.x.shape[0] >= min_nodes:

                    df_new = pd.DataFrame([{'x': np.asarray(subgraph.x[:,0].cpu()), 'y': np.asarray(subgraph.x[:,1].cpu()), 'z' : np.asarray(subgraph.x[:,2].cpu())}])
                    df_tracks = pd.concat([df_tracks, df_new], ignore_index=True)

    return df_tracks