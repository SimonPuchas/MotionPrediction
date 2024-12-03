import pandas as pd
import torch
import numpy as np
import os

'''This script was just used to test how to filter smaller sequences
    and how we can cut the sequences to same legnths'''

def load_data(filepath):
    tensor_files = [os.path.join(root, file) for root, dirs, files in os.walk(filepath) for file in files if file.endswith('tensor_data.npy')]

    print(f"Found {len(tensor_files)} tensor files")

    sequences = []
    sequence_lengths = []

    for file_path in tensor_files:
        data = np.load(file_path, allow_pickle=True)
        movement_data = torch.tensor(data['data'], dtype=torch.float32)
        movement_data = torch.cat((movement_data[:, :-2], movement_data[:, -1:]), dim=1)  # Remove second-to-last column
        sequence_lengths.append(movement_data.shape[0])
        sequences.append(movement_data)
        #print(movement_data.shape)
    
    print(f"Loaded {len(sequences)} sequences with varying lengths.")
    return sequences, sequence_lengths

def filter_small_sequences(sequences, sequence_lengths, min_length=50):
    filtered_sequences = []
    filtered_sequence_lengths = []
    
    for seq_idx, sequence in enumerate(sequences):
        if sequence_lengths[seq_idx] >= min_length:
            filtered_sequences.append(sequence)
            filtered_sequence_lengths.append(sequence_lengths[seq_idx])
    
    print(f"Filtered out {len(sequences) - len(filtered_sequences)} sequences with less than {min_length} timesteps.")
    return filtered_sequences, filtered_sequence_lengths

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

def main():
    filepath = '/home/simon/MotionPrediction/catkin_ws/src/datasetcreator/src/runs_new'
    sequences, sequence_lengths = load_data(filepath)
    #print(sequences[0].shape)
    #print(min(sequence_lengths))
    #print(max(sequence_lengths))

    filtered_sequences, filtered_sequence_lengths = filter_small_sequences(sequences, sequence_lengths, min_length=50)
    #print(len(filtered_sequences))
    #print(min(filtered_sequence_lengths))

    cut_sequences, cut_sequence_lengths = shorten_sequences(filtered_sequences, filtered_sequence_lengths, max_length=50)
    print(len(cut_sequences))
    print(min(cut_sequence_lengths))
    print(max(cut_sequence_lengths))
    print(cut_sequence_lengths)

if __name__ == '__main__':
    main()