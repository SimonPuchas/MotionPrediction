import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# preparing data
def preparing_data():
    #loading data
    path = '/home/simon/MotionPrediction/Datasets/lstm_dataset2.pt'
    tensor = torch.load(path, weights_only=True)
    X = tensor['X']
    y = tensor['y']
    sequence_lengths = tensor['sequence_lengths']

    # concatenating all the windows into one tensor to compute mean and std
    X_combined = torch.cat(X, dim=0)
    mean = X_combined.mean(dim=0)
    std = X_combined.std(dim=0)

    # standardizing the data
    std[std == 0] = 1 # to avoid division by zero
    X_standardized = [(x - mean) / std for x in X]
    y_standardized = [(y - mean) / std for y in y]

    # debugging
    #print(X_standardized[0].shape)
    #print(y_standardized[0].shape)
    #print(X_standardized[0])

    # splitting the data into training and testing sets
    # first we need to know which tensors belong to which movement
    # this ensures that we do not split movements across train/test/val sets
    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15

    movement_start_indices = [0] + list(np.cumsum(sequence_lengths)[:-1])
    num_movements = len(sequence_lengths)

    movement_indices = np.random.permutation(num_movements)

    train_split = int(num_movements * train_ratio)
    val_split = train_split + int(num_movements * val_ratio)

    train_indices = movement_indices[:train_split]
    val_indices = movement_indices[train_split:val_split]
    test_indices = movement_indices[val_split:]

    X_train = [X_standardized[start:end] for i in train_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]
    y_train = [y_standardized[start:end] for i in train_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]

    X_val = [X_standardized[start:end] for i in val_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]
    y_val = [y_standardized[start:end] for i in val_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]

    X_test = [X_standardized[start:end] for i in test_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]
    y_test = [y_standardized[start:end] for i in test_indices for start, end in [(movement_start_indices[i], movement_start_indices[i] + sequence_lengths[i])]]

    # debugging
    #print(len(X_train))
    #print(len(X_test))
    #print(len(X_val))
    #print(len(y_train))
    #print(len(y_test))
    #print(len(y_val))
    #print(X_test)
    
    batch_size = 8

    # creating dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # creating batches
    def collate_fn(batch):
        X = [x[0] for x in batch]
        y = [x[1] for x in batch]
        return pack_sequence(X, enforce_sorted=False), pack_sequence(y, enforce_sorted=False)
    









    



