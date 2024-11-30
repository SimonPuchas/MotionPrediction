import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

data_dir = '/home/simon/MotionPrediction/catkin_ws/src/datasetcreator/src/runs'

def load_and_combine_tensors(data_dir):
    tensor_files = [os.path.join(root, file) 
                    for root, dirs, files in os.walk(data_dir)
                    for file in files if file.endswith('tensor_data.npy')]
    
    print(f"Found {len(tensor_files)} tensor files")
    
    movement_sequences = []
    sequence_lengths = []
    
    for file_path in tensor_files:
        data = np.load(file_path, allow_pickle=True)
        movement_data = torch.tensor(data['data'], dtype=torch.float32)
        movement_data = torch.cat((movement_data[:, :-2], movement_data[:, -1:]), dim=1)  # Remove second-to-last column
        sequence_lengths.append(movement_data.shape[0])
        movement_sequences.append(movement_data)
        #print(movement_data.shape)
    
    print(f"Loaded {len(movement_sequences)} sequences with varying lengths.")
    return movement_sequences, sequence_lengths

def prepare_lstm_dataset(movement_sequences, sequence_lengths, window_size=10):
    X, y = [], []
    
    for seq_idx, sequence in enumerate(movement_sequences):
        seq_length = sequence_lengths[seq_idx]
        
        X_sequence = []
        y_sequence = []
        for i in range(seq_length - window_size):
            X_sequence.append(sequence[i:i+window_size])
            y_sequence.append(sequence[i+window_size])
        
        X.append(torch.stack(X_sequence))
        y.append(torch.stack(y_sequence))
    
    print(f"Prepared {len(X)} input sequences for training.")
    return X, y

'''
def standardize_data(X, y):
    X_all = torch.cat([x.flatten(0, 1) for x in X], dim=0)
    #y_all = torch.cat([yy.flatten(0, 1) for yy in y], dim=0)
    
    mean = X_all.mean(dim=0, keepdim=True)
    std = X_all.std(dim=0, keepdim=True)
    
    std[std == 0] = 1  # avoiding NaNs
    X_standardized = [(x - mean) / std for x in X]
    y_standardized = [(yy - mean) / std for yy in y]
    
    return X_standardized, y_standardized'''

def split_sequences(sequences, test_size=0.1, val_size=0.1):
    # Split sequences into train+val and test
    X_train_val, X_test = train_test_split(sequences, test_size=test_size, random_state=42)
    
    # Split the train+val into train and validation
    X_train, X_val = train_test_split(X_train_val, test_size=val_size / (1 - test_size), random_state=42)
    
    return X_train, X_val, X_test

test_sequ, test_lens = load_and_combine_tensors(data_dir)
X, y = prepare_lstm_dataset(test_sequ, test_lens)
#X_stand, y_stand = standardize_data(X, y)
X_train_seq, X_val_seq, X_test_seq = split_sequences(list(zip(X, y)))
    
# Unzip the sequences back into separate X and y lists
X_train, y_train = zip(*X_train_seq)
X_val, y_val = zip(*X_val_seq)
X_test, y_test = zip(*X_test_seq)

torch.save({
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }, '/home/simon/MotionPrediction/Datasets/lstm_dataset5_plain.pt')

#X_pad = pad_sequence(X_stand, batch_first=True, padding_value=0)
#print(X[0].shape)
#print(X_pad.shape)
#print(X_pad)

#print("X_train length", len(X_train))
#print("X_val length", len(X_val))
#print("X_test length", len(X_test))
#print(X_train[0].shape)
#print(y_train[0].shape)

#X_train_pad = pad_sequence(X_train, batch_first=True, padding_value=0)

#print(X_train_pad.shape)
