import torch
import numpy as np
import os
from glob import glob

def load_and_combine_tensors(data_dir):
    """
    Load all tensor files from the runs directory and combine them into a single dataset.
    
    Args:
        data_dir (str): Path to the directory containing run folders
    
    Returns:
        combined_tensor: Combined tensor of shape (num_sequences, sequence_length, features)
        sequence_lengths: List of original sequence lengths
    """
    # Find all tensor files
    tensor_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('tensor_data.npy'):
                tensor_files.append(os.path.join(root, file))
    
    print(f"Found {len(tensor_files)} tensor files")
    
    # Load and process each tensor
    movement_sequences = []
    sequence_lengths = []
    
    for file_path in tensor_files:
        # Load the numpy file
        data = np.load(file_path, allow_pickle=True)
        
        # Extract the movement data
        movement_data = torch.tensor(data['data'], dtype=torch.float32)
        
        # Store sequence length for padding reference
        sequence_lengths.append(movement_data.shape[0])
        
        # Add to sequences list
        movement_sequences.append(movement_data)
    
    # Find max sequence length for padding
    max_seq_length = max(sequence_lengths)
    
    # Pad sequences to same length
    padded_sequences = []
    for seq in movement_sequences:
        # Calculate padding needed
        pad_length = max_seq_length - seq.shape[0]
        
        if pad_length > 0:
            # Create padding
            padding = torch.zeros((pad_length, seq.shape[1]), dtype=torch.float32)
            # Concatenate padding to sequence
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
            
        padded_sequences.append(padded_seq)
    
    # Stack all sequences into a single tensor
    combined_tensor = torch.stack(padded_sequences)
    
    print(f"Combined tensor shape: {combined_tensor.shape}")
    print(f"Number of sequences: {len(sequence_lengths)}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Number of features: {combined_tensor.shape[-1]}")
    
    return combined_tensor, sequence_lengths

def prepare_lstm_dataset(combined_tensor, sequence_lengths, window_size=10):
    """
    Prepare the combined tensor for LSTM training by creating sliding windows
    
    Args:
        combined_tensor: Tensor of shape (num_sequences, sequence_length, features)
        sequence_lengths: List of original sequence lengths
        window_size: Size of sliding window for input sequences
    
    Returns:
        X: Input sequences
        y: Target values
    """
    X = []
    y = []
    
    # Process each sequence
    for seq_idx in range(combined_tensor.shape[0]):
        seq_length = sequence_lengths[seq_idx]
        sequence = combined_tensor[seq_idx, :seq_length, :]
        
        # Create sliding windows
        for i in range(sequence.shape[0] - window_size):
            X.append(sequence[i:i+window_size])
            y.append(sequence[i+window_size])
    
    # Stack all windows into a single tensor
    X = torch.stack(X)
    y = torch.stack(y)
    
    print(f"Training data shape (X): {X.shape}")
    print(f"Target data shape (y): {y.shape}")
    
    return X, y

if __name__ == "__main__":
    # Example usage
    data_dir = "/home/simon/MotionPrediction/catkin_ws/src/datasetcreator/src/runs"  # Replace with your actual path
    
    # Combine all tensors
    combined_tensor, sequence_lengths = load_and_combine_tensors(data_dir)
    
    # Prepare dataset for LSTM
    X, y = prepare_lstm_dataset(combined_tensor, sequence_lengths)
    
    # Save the prepared dataset
    torch.save({
        'X': X,
        'y': y,
        'sequence_lengths': sequence_lengths
    }, 'lstm_dataset.pt')
    
    print("Dataset saved successfully!")