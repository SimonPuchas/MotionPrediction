import torch
import numpy as np
import os

def load_and_combine_tensors(data_dir):
    
    # Find all tensor files
    tensor_files = [os.path.join(root, file) 
                    for root, dirs, files in os.walk(data_dir)
                    for file in files if file.endswith('tensor_data.npy')]
    
    print(f"Found {len(tensor_files)} tensor files")
    
    # Load and process each tensor
    movement_sequences = []
    sequence_lengths = []
    
    for file_path in tensor_files:
        data = np.load(file_path, allow_pickle=True)
        movement_data = torch.tensor(data['data'], dtype=torch.float32)
        sequence_lengths.append(movement_data.shape[0])
        movement_sequences.append(movement_data)
    
    print(f"Loaded {len(movement_sequences)} sequences with varying lengths.")
    return movement_sequences, sequence_lengths

def prepare_lstm_dataset(movement_sequences, sequence_lengths, window_size=10):
    
    X, y, new_lengths = [], [], []
    
    for seq_idx, sequence in enumerate(movement_sequences):
        seq_length = sequence_lengths[seq_idx]
        
        # Create sliding windows
        for i in range(seq_length - window_size):
            X.append(sequence[i:i+window_size])
            y.append(sequence[i+window_size])
        
        new_lengths.append(len(X) - sum(new_lengths))  # Count windows per sequence
    
    print(f"Prepared {len(X)} input sequences for training.")
    return X, y, new_lengths

if __name__ == "__main__":
    data_dir = "/home/simon/MotionPrediction/catkin_ws/src/datasetcreator/src/runs"  # Replace with actual path
    
    # Combine all tensors
    movement_sequences, sequence_lengths = load_and_combine_tensors(data_dir)
    
    # Prepare dataset for LSTM
    X, y, new_lengths = prepare_lstm_dataset(movement_sequences, sequence_lengths)
    
    # Save the prepared dataset
    torch.save({'X': X, 'y': y, 'sequence_lengths': new_lengths}, 'lstm_dataset.pt')
    
    print("Dataset saved successfully!")
