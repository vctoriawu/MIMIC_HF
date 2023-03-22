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
from labml_nn.gat.dataloader_multigpu import *

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

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
    #device: torch.device = rank
    backend: str = "nccl"

    # Optimizer
    optimizer: torch.optim.Adam
    adj_mat: torch.Tensor

    def run(self, world_size: int, gpu_id: int):
        """
        ### Training loop
        We do full batch training since the dataset is small.
        If we were to sample and train we will have to sample a set of
        nodes for each training step along with the edges that span
        across those selected nodes.
        """
        # Move the adjacency matrix to the device
        edges_adj = self.adj_mat.to(gpu_id)
        # Add an empty third dimension for the heads
        edges_adj = edges_adj.unsqueeze(-1)
        #wrap model in ddp
        self.model = DDP(self.model, device_ids=[gpu_id])

        # Training loop 
        for epoch in monit.loop(self.epochs):
            for batch_ndx, batch in enumerate(self.dataset[0]):
                #set up ddp
                ddp_setup(rank, world_size)
                for i,row in enumerate(batch['patient']):
                    train_features = batch['patient'][0].to(gpu_id)
                    train_labels = batch['label'][i].to(gpu_id)
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
                    # Calculate gradients
                loss.backward()
                # Take optimization step
                self.optimizer.step()
                destroy_process_group()
                # Log the loss
                if self.gpu_id == 0:
                    tracker.add('loss.train', loss)
                    train_accuracy = accuracy(output, train_labels)
                    # Log the accuracy
                    tracker.add('accuracy.train', train_accuracy) 

            # Set mode to evaluation mode for validation
            self.model.eval()

            for batch_ndx, batch in enumerate(self.dataset[1]):
                for i,row in enumerate(batch['patient']):
                    val_features = batch['patient'][0].to(gpu_id)
                    val_labels = batch['label'][i].to(gpu_id)
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

def ddp_setup(rank, world_size, c:Configs):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = machine
    os.environ["MASTER_PORT"] = freeport 
    init_process_group(backend=c.backend, rank=rank, world_size=world_size)

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
    return GAT(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout)

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


def main(rank: int, world_size: int, freeport: str):
    # Create configurations
    rank = rank
    freeport = freeport
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
        conf.run(world_size = world_size, gpu_id = rank)


#
if __name__ == '__main__':
    import sys
    world_size = torch.cuda.device_count()
    gpu_ids = [0,1]
    machine: str = "purang28"
    freeport: str = "61247"
    mp.spawn(main, args=(world_size, freeport), nprocs=world_size)