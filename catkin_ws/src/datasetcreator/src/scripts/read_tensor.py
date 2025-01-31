import numpy as np
import torch

# Load and check the .npy file
def load_npy_tensor(file_path):
    # Load data with NumPy
    np_data = np.load(file_path, allow_pickle=True)

    # Check if it's a structured array
    if np_data.dtype.names is not None:
        print("Data is a structured array. Fields:", np_data.dtype.names)
        # Extract each field and convert to PyTorch tensors
        tensors = {name: torch.tensor(np_data[name]) for name in np_data.dtype.names}
        for name, tensor in tensors.items():
            print(f"Tensor for field '{name}':\n", tensor)
            print(f"Shape of '{name}':", tensor.shape)
        return tensors
    else:
        # Convert directly if it's a standard ndarray
        tensor_data = torch.from_numpy(np_data)
        print("Tensor data:\n", tensor_data)
        print("Tensor shape:", tensor_data.shape)
        return tensor_data

# Example usage
if __name__ == "__main__":
    file_path = "catkin_ws/src/datasetcreator/src/runs_new/run_20241116_144558/tensor_data.npy"  # Replace with the actual path
    tensor = load_npy_tensor(file_path)

