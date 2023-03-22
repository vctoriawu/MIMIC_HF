from typing import Dict

import numpy as np
import torch
from torch import nn

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs, option, calculate
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.gat import GraphAttentionLayer
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.gat.dataloader import *

class GAT(Module):
    """
    ## Graph Attention Network (GAT)

    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionLayer(n_hidden, 64, 1, is_concat=False, dropout=dropout)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(in_features=46, out_features=24)
        self.sigmoid = nn.Sigmoid()
        self.classification_mlp = nn.Sequential(nn.Linear(in_features=46,
                                                          out_features=12),
                                                #nn.BatchNorm1d(12),
                                                #nn.ELU(inplace=True),
                                                #nn.Dropout(p=0.5),
                                                nn.Linear(in_features=12,
                                                          out_features=1),
                                                nn.Sigmoid())

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        x = self.output(x, adj_mat)
        # (46, 46)
        x = torch.mean(x, axis=1)
        x = x[None, :]
        x = self.classification_mlp(x)
        x = torch.squeeze(x, 0)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output: torch.Tensor, labels: torch.Tensor):
    """
    A simple function to calculate the accuracy
    """
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)

class Configs(BaseConfigs):
    """
    ## Configurations
    """

    # Model
    model: GAT
    # Number of features per node in the input
    in_features: int = 3
    # Number of features in the first graph attention layer
    n_hidden: int = 64
    # Number of heads
    n_heads: int = 8
    # Number of classes for classification
    n_classes: int = 2
    # Dropout probability
    dropout: float = 0.6
    # Whether to include the citation network
    include_edges: bool = True
    # Dataset
    dataset: MIMICDataset 
    batch_size: int = 32
    val_size: float = 0.1
    test_size: float = 0.1
    # Number of training iterations
    epochs: int = 10
    # Loss function
    loss_func = nn.BCELoss()
    # Device to train on
    #
    # This creates configs for device, so that
    # we can change the device by passing a config value
    device: torch.device = DeviceConfigs()
    # Optimizer
    optimizer: torch.optim.Adam
    adj_mat: torch.Tensor

    def run(self):
        """
        ### Training loop
        We do full batch training since the dataset is small.
        If we were to sample and train we will have to sample a set of
        nodes for each training step along with the edges that span
        across those selected nodes.
        """
        # Move the adjacency matrix to the device
        edges_adj = self.adj_mat.to(self.device)
        # Add an empty third dimension for the heads
        edges_adj = edges_adj.unsqueeze(-1)

        # Training loop 
        e_train_losses = []
        e_train_acc = []
        e_val_losses = []
        e_val_acc = []

        for epoch in monit.loop(self.epochs):
            for batch_ndx, batch in enumerate(self.dataset[0]):
                for i,row in enumerate(batch['patient']):
                    train_features = batch['patient'][0].to(self.device)
                    train_labels = batch['label'][i].to(self.device)
                    print(f"training epoch: {epoch}, batch: {batch_ndx}, patient: {i+1}")
                    print(f"model parameters: {count_parameters(self.model)}")

                    # Set the model to training moxde
                    self.model.train()
                    # Make all the gradients zero
                    self.optimizer.zero_grad()
                    # Evaluate the model
                    output = self.model(train_features, edges_adj)
                    print(output)
                    # Get the loss for training nodes
                    loss = self.loss_func(output, train_labels)
                    e_train_losses.append(loss)
                    # Calculate gradients
                loss.backward()
                # Take optimization step
                self.optimizer.step()
                # Log the loss
                tracker.add('loss.train', loss)
                train_accuracy = accuracy(output, train_labels)
                e_train_acc.append(train_accuracy)
                # Log the accuracy
                tracker.add('accuracy.train', train_accuracy) 

            # Set mode to evaluation mode for validation
            self.model.eval()

            for batch_ndx, batch in enumerate(self.dataset[1]):
                for i,row in enumerate(batch['patient']):
                    val_features = batch['patient'][0].to(self.device)
                    val_labels = batch['label'][i].to(self.device)
                # No need to compute gradients
                with torch.no_grad():
                    # Evaluate the model again
                    output = self.model(val_features, edges_adj)
                    # Calculate the loss for validation nodes
                    loss = self.loss_func(output, val_labels) 
                    e_val_losses.append(loss)
                    # Log the loss
                    tracker.add('loss.valid', loss)
                    val_accuracy = accuracy(output, val_labels)
                    e_val_acc.append(val_accuracy)
                    # Log the accuracy
                    tracker.add('accuracy.valid', val_accuracy)
                    print(f"validation epoch: {epoch}, batch: {batch_ndx}, patient: {i+1}") 

            # Save logs
            tracker.save()

@option(Configs.dataset)
def get_dataset(c: Configs):
    """
    Create dataset
    """    
    return get_dataloader(c.batch_size, c.val_size, c.test_size)

@option(Configs.model)
def gat_model(c: Configs):
    """
    Create GAT model
    """
    return GAT(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout).to(c.device)

@option(Configs.adj_mat)
def get_adj_mat(c: Configs):
    """
    Get adjacency matrix for fully-connected graph
    """
    for i, sample in enumerate(c.dataset[0]):
        if i==0:
            num_features = sample['patient'].shape[1]

    adj_mat = torch.ones((num_features, num_features), dtype=torch.bool) 
    return adj_mat

@option(Configs.optimizer)
def _optimizer(c: Configs):
    """
    Create configurable optimizer
    """
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


def main():
    # Create configurations
    conf = Configs()
    # Create an experiment
    experiment.create(name='gat')
    # Calculate configurations.
    experiment.configs(conf, {
        # Adam optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 5e-3,
        'optimizer.weight_decay': 5e-4,
        }
    )

    # Start and watch the experiment
    with experiment.start():
        # Run the training
        conf.run()


#
if __name__ == '__main__':
    main()