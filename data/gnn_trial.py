import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datacleaner import *
from abc import ABC
import pandas as pd
import os
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
import torch
import networkx as nx
import numpy as np
import dgl
from tqdm import tqdm
import wandb

from torch.utils.data import WeightedRandomSampler
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.nn import MessagePassing, Sequential, GCNConv, global_add_pool, global_mean_pool
from sklearn.metrics import accuracy_score
import numpy as np
import time

from math import floor
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

class MIMICDataset(Dataset, ABC):
    def __init__(self, data_df):

        super().__init__()
        
        # Extract the features and labels
        features_tensor = torch.tensor([], dtype=torch.float)
        labels_tensor = torch.tensor([], dtype=torch.long)
        
        # Extract the labels
        self.labels = torch.tensor(data_df['HF'].tolist(), dtype=torch.long)
        
        # Extract the image data
        self.data = torch.FloatTensor(data_df.loc[:, data_df.columns != 'HF'].values.tolist())
        #print(self.data.shape)
        
        # Find the proportion of each digit in the set
        self.class_weights = 1 / np.unique(self.labels, return_counts=True)[1]
        #print(np.unique(self.labels, return_counts=True)[1])
        self.class_weights = self.class_weights[self.labels]
        #print(self.class_weights)
        
        self.nx_graph = nx.complete_graph(46) 
        
        self.num_samples = data_df.shape[0]    

    def get(self, idx):

        # Retrieve the sample
        sample_features = self.data[idx].view(46, -1).unsqueeze(0).unsqueeze(0) #TODO

        # Retrieve the label
        label = self.labels[idx]

        # Create the PyG data from the graph structure
        #g = from_networkx(self.nx_graph)
        #g = dgl.from_networkx(nx_g)
        # Create DGL graph from networkx
        g = from_networkx(self.nx_graph)

        # Add data and label to the PyG data
        g.y = label
        g.features = sample_features
        g.nx_graph = nx.complete_graph(46) 

        return g

    def len(self) -> int:
        return self.num_samples

    
class GCNClassifier(nn.Module):
    def __init__(self,
                 input_feature_dim,
                 dropout_p,
                 gnn_hidden_dims,
                 mlp_hidden_dim,
                 num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        # GNN layers
        for gnn_hidden_dim in gnn_hidden_dims:
            self.layers.append(Sequential('x, edge_index', [(GCNConv(in_channels=input_feature_dim,
                                                                     out_channels=gnn_hidden_dim), 'x, edge_index -> x'),
                                                            nn.BatchNorm1d(gnn_hidden_dim),
                                                            nn.Dropout(p=dropout_p),
                                                            nn.ReLU(inplace=True)]))
            input_feature_dim = gnn_hidden_dim

        # Output MLP layers
        self.output_mlp = nn.Sequential(nn.Linear(in_features=gnn_hidden_dims[-1],
                                                  out_features=mlp_hidden_dim),
                                        nn.BatchNorm1d(mlp_hidden_dim),
                                        nn.Dropout(p=dropout_p),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=mlp_hidden_dim,
                                                  out_features=num_classes),
                                        nn.Sigmoid())

    def forward(self, g):

        h = g.features.squeeze()
        h = torch.reshape(h, [h.shape[0]*h.shape[1], -1])
        edge_index = g.edge_index
        
        # GNN layers
        for gnn_layer in self.layers:
            h = gnn_layer(h, edge_index)

        # Pool node embeddings to create the graph embedding
        h = global_mean_pool(h, g.batch)

        # Output MLP
        h = self.output_mlp(h)
        h = h.squeeze()

        return h

def get_cleaned_df():
    df = get_dataframe()
    
    #Count the number of diagnosis risk factors
    icd9_cols = df.filter(regex='^icd9').columns

    # Find all columns with age of diagnoses
    age_cols = df.filter(regex='admit_age').columns
    age_cols = age_cols[:-1] #don't get the HF age cause we don't want to drop that one

    # Find all columns with icu stay of diagnoses
    icu_cols = df.filter(regex='icu_stay').columns

    # Find columns associated with echo data
    echo_cols = ['height', 'weight', 'bpsys', 'bpdias', 'hr', 'EF']
    
    #Create a new dataframe that has the groupings
    grouped_df = pd.DataFrame()

    # group the diagnoses
    for i in range(len(icd9_cols)):
        # stack the three columns into a single list column
        df['new_'+icd9_cols[i]] = df.apply(lambda x: [x[icd9_cols[i]], x[age_cols[i]], x[icu_cols[i]]], axis=1)
        grouped_df['new_'+icd9_cols[i]] = (df['new_'+icd9_cols[i]])
        df.drop(('new_'+icd9_cols[i]), axis=1)
        
    #group all echo variables with age and echo_icu_stay
    for i in range(len(echo_cols)):
        # stack the three columns into a single list column
        df['new_'+echo_cols[i]] = df.apply(lambda x: [x[echo_cols[i]], x['age'], x['echo_icu_stay']], axis=1)
        grouped_df['echo_'+echo_cols[i]] = (df['new_'+echo_cols[i]])
        df.drop('new_'+echo_cols[i], axis=1)
        
    #add gender - standalone
    # replace 'M' and 'F' with 1 and 2, respectively
    df['gender'] = df['gender'].replace({'M': 1, 'F': 2})
    grouped_df['gender'] = df.apply(lambda x: [x['gender'], 0, 0], axis=1)

    #add target HF
    grouped_df['HF'] = df['target_HF']
    
    return grouped_df

def train(trainloader, valloader, testloader, gnn_classifiers):
    # Create the loss function and the optimizer
    optimizer = Adam(list(gnn_classifiers.parameters()))
    loss_func = nn.BCELoss()
    
    epochs=50
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    wandb.init(project='ELEC571_Project', entity='vctoriawu')

    for epoch in range(epochs):

        train_time = time.time()
        train_loss = 0
        train_preds = []
        train_labels = []

        for i, data_batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()

            # Use GNN layers to propagate messages between embeddings #TODO - define data_batch.x and data_batch.y
            data_batch.x = data_batch.features
            x = gnn_classifiers(data_batch)
            y = data_batch.y
            y = y.type(torch.float)

            loss = loss_func(x, y)
            train_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()

            # Save label and prediction
            data_batch.y.detach().cpu().numpy()
            prediction = x.detach().cpu().numpy().round()

            train_preds.extend(prediction.tolist())
            train_labels.extend(data_batch.y.tolist())

        train_time = time.time() - train_time

        val_loss = 0
        val_preds = []
        val_labels = []

        test_time = time.time()

        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(valloader)):

                # Use GNN layers to propagate messages between pixel embeddings
                data_batch.x = data_batch.features
                x = gnn_classifiers(data_batch)
                y = data_batch.y
                y = y.type(torch.float)
                loss = loss_func(x, y)
                val_loss += loss.detach().cpu().item() * batch_size

                # Save label and prediction
                data_batch.y.detach().cpu().numpy()
                prediction = x.detach().cpu().numpy().round()

                val_preds.extend(prediction.tolist())
                val_labels.extend(data_batch.y.tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        train_acc = accuracy_score(train_labels, train_preds)


        test_time = time.time() - test_time

        # Append losses and accuracies to lists for plotting
        train_loss /= len(trainloader)*batch_size
        val_loss /= len(valloader)*batch_size
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch information
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")



if __name__ == "__main__":
    grouped_df = get_cleaned_df()
    val_size = 0.1
    test_size = 0.1
    batch_size = 32 
    
    # split data into training and test sets
    train_data, test_data = train_test_split(grouped_df, test_size=test_size)
    train_data, val_data = train_test_split(train_data, test_size=val_size/(1-test_size))
    
    train_dataset = MIMICDataset(train_data)
    val_dataset = MIMICDataset(val_data)
    test_dataset = MIMICDataset(test_data)
    
    print("The training set has {} samples.".format(len(train_dataset)))
    print("The training set has {} samples.".format(len(val_dataset)))
    print("The test set has {} samples.".format(len(test_dataset)))
    
    train_sampler = WeightedRandomSampler(weights=train_dataset.class_weights, num_samples=len(train_dataset), 
                                          replacement=False)
    val_sampler = WeightedRandomSampler(weights=val_dataset.class_weights, num_samples=len(val_dataset), 
                                          replacement=False)                                      
    test_sampler = WeightedRandomSampler(weights=test_dataset.class_weights, num_samples=len(test_dataset), 
                                         replacement=False)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler)
    valloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, sampler=val_sampler)
    testloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, sampler=test_sampler)

    gnn_classifiers = GCNClassifier(input_feature_dim=3, #TODO
                                dropout_p=0.3,
                                gnn_hidden_dims=[64, 16],
                                mlp_hidden_dim=16,
                                num_classes=1)
    
    train(trainloader, valloader, testloader)
    

