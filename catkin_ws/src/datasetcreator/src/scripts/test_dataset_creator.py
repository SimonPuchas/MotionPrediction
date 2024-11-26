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
    
    # Find max sequence length for padding
    max_seq_length = max(sequence_lengths)
    
    # Pad sequences to same length
    '''padded_sequences = [
        torch.cat([seq, torch.zeros((max_seq_length - seq.shape[0], seq.shape[1]), dtype=torch.float32)], dim=0) 
        if seq.shape[0] < max_seq_length else seq
        for seq in movement_sequences
    ]'''
    
    # Stack all sequences into a single tensor
    combined_tensor = torch.stack(movement_sequences)
    
    print(f"Combined tensor shape: {combined_tensor.shape}")
    return combined_tensor, sequence_lengths

def prepare_lstm_dataset(combined_tensor, sequence_lengths, window_size=10):
    
    X, y = [], []
    
    for seq_idx in range(combined_tensor.shape[0]):
        seq_length = sequence_lengths[seq_idx]
        sequence = combined_tensor[seq_idx, :seq_length, :]
        
        # Create sliding windows
        for i in range(seq_length - window_size):
            X.append(sequence[i:i+window_size])
            y.append(sequence[i+window_size])
    
    X = torch.stack(X)
    y = torch.stack(y)
    
    print(f"Training data shape (X): {X.shape}")
    print(f"Target data shape (y): {y.shape}")
    return X, y

if __name__ == "__main__":
    data_dir = "/home/simon/MotionPrediction/catkin_ws/src/datasetcreator/src/runs"  # Replace with your actual path
    
    # Combine all tensors
    combined_tensor, sequence_lengths = load_and_combine_tensors(data_dir)
    
    # Prepare dataset for LSTM
    X, y = prepare_lstm_dataset(combined_tensor, sequence_lengths)
    
    # Save the prepared dataset
    torch.save({'X': X, 'y': y, 'sequence_lengths': sequence_lengths}, 'lstm_dataset2.pt')
    
    print("Dataset saved successfully!")