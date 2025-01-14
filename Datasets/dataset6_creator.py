import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    tensor_files = [os.path.join(root, file) for root, dirs, files in os.walk(filepath) for file in files if file.endswith('tensor_data.npy')]

    print(f"Found {len(tensor_files)} tensor files")

    sequences = []
    sequence_lengths = []

    for file_path in tensor_files:
        data = np.load(file_path, allow_pickle=True)
        movement_data = torch.tensor(data['data'], dtype=torch.float32)
        movement_data = torch.cat((movement_data[:, :-2], movement_data[:, -1:]), dim=1)  # Remove second-to-last column, this was a redundant feature
        sequence_lengths.append(movement_data.shape[0])
        sequences.append(movement_data)
        #print(movement_data.shape)
    
    print(f"Loaded {len(sequences)} sequences with varying lengths.")
    return sequences, sequence_lengths

# filtering out sequences with less than 50 timesteps
def filter_small_sequences(sequences, sequence_lengths, min_length=50):
    filtered_sequences = []
    filtered_sequence_lengths = []
    
    for seq_idx, sequence in enumerate(sequences):
        if sequence_lengths[seq_idx] >= min_length:
            filtered_sequences.append(sequence)
            filtered_sequence_lengths.append(sequence_lengths[seq_idx])
    
    print(f"Filtered out {len(sequences) - len(filtered_sequences)} sequences with less than {min_length} timesteps.")
    return filtered_sequences, filtered_sequence_lengths

# shortening all sequences, such that they are all of length 50
def shorten_sequences(filtered_sequences, filtered_sequence_lengths, max_length=50):
    cut_sequences = []
    cut_sequence_lengths = []

    for seq_idx, sequence in enumerate(filtered_sequences):
        if filtered_sequence_lengths[seq_idx] > max_length and filtered_sequence_lengths[seq_idx] < 2*max_length:
            cut_sequences.append(sequence[:max_length])
            cut_sequence_lengths.append(max_length)
        else:
            sequence = sequence[:2*max_length]
            t1 = sequence[:max_length]
            t2 = sequence[max_length:]
            cut_sequences.append(t1)
            cut_sequences.append(t2)
            cut_sequence_lengths.append(max_length)
            cut_sequence_lengths.append(max_length)

    return cut_sequences, cut_sequence_lengths

# apply sliding windows to create input-output pairs for the LSTM
def prepare_lstm_dataset(movement_sequences, sequence_lengths, window_size=10):
    X, y = [], []

    for seq_idx, sequence in enumerate(movement_sequences):
        seq_length = sequence_lengths[seq_idx]

        X_sequence = []
        y_sequence = []
        # added try-except, because one sequence gave an index out of bounds error, which didn't make sense
        try:
            for i in range(seq_length - window_size):
                X_sequence.append(sequence[i:i+window_size])
                y_sequence.append(sequence[i+window_size])
        
            X.append(torch.stack(X_sequence))
            y.append(torch.stack(y_sequence))
        except Exception as e:
            print(f"Error in sequence {seq_idx}: {e}")
            print(f"Sequence length: {seq_length}, Sequence shape: {sequence.shape}")


    print(f"Prepared {len(X)} input sequences for training.")
    return X, y

def standardize_data(X, y):
    X_all = torch.cat([x.flatten(0, 1) for x in X], dim=0)
    
    mean = X_all.mean(dim=0, keepdim=True)
    std = X_all.std(dim=0, keepdim=True)
    
    std[std == 0] = 1  # avoiding NaNs
    X_standardized = [(x - mean) / std for x in X]
    y_standardized = [(yy - mean) / std for yy in y]
    
    return X_standardized, y_standardized

def split_sequences(sequences, test_size=0.1, val_size=0.1):
    # Split sequences into train+val and test
    X_train_val, X_test = train_test_split(sequences, test_size=test_size, random_state=42)
    
    # Split the train+val into train and validation
    X_train, X_val = train_test_split(X_train_val, test_size=val_size / (1 - test_size), random_state=42)
    
    return X_train, X_val, X_test

def main():
    filepath = 'catkin_ws/src/datasetcreator/src/runs_new'
    sequences, sequence_lengths = load_data(filepath)
    #print(sequences[0].shape)
    #print(min(sequence_lengths))
    #print(max(sequence_lengths))

    filtered_sequences, filtered_sequence_lengths = filter_small_sequences(sequences, sequence_lengths, min_length=50)
    print("Filtered sequences: ", len(filtered_sequences))
    print("Minimum length after filtering: ", min(filtered_sequence_lengths))

    cut_sequences, cut_sequence_lengths = shorten_sequences(filtered_sequences, filtered_sequence_lengths, max_length=50)
    print("Shortened sequences: ", len(cut_sequences))
    print("Minimum length after shortening: ", min(cut_sequence_lengths))
    #print(max(cut_sequence_lengths))

    #print(cut_sequences)
    #print(cut_sequence_lengths)
    X, y = prepare_lstm_dataset(cut_sequences, cut_sequence_lengths, window_size=10)

    X_stand, y_stand = standardize_data(X, y)

    X_train_seq, X_val_seq, X_test_seq = split_sequences(list(zip(X_stand, y_stand)))

    X_train, y_train = zip(*X_train_seq)
    X_val, y_val = zip(*X_val_seq)
    X_test, y_test = zip(*X_test_seq)

    print(f"X_train: {len(X_train)}, X_val: {len(X_val)}, X_test: {len(X_test)}")
    print(f"y_train: {len(y_train)}, y_val: {len(y_val)}, y_test: {len(y_test)}")

    torch.save({
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }, 'Datasets/lstm_dataset6.pt')

if __name__ == '__main__':
    main()