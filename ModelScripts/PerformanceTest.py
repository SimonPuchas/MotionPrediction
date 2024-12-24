import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
import math
import os

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz

        # Projection layer to adjust input to hidden size
        self.input_projection = nn.Linear(input_sz, hidden_sz)

        self.W = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

        # Prediction head to map back to original input size
        self.fc = nn.Linear(hidden_sz, input_sz)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        bs, seq_sz, input_features = x.size()

        # Project input to hidden size
        x_projected = self.input_projection(x)

        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x_projected[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # Use the last hidden state for prediction
        prediction = self.fc(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return prediction, (h_t, c_t)

def test_model(model, test_loader, criterion, device, feature_count=8):
    """
    Test the model on the test dataset and print metrics and sample predictions.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        criterion: Loss function used during training (e.g., MSELoss).
        device: Device (CPU or GPU).
        feature_count: Number of features to compare in predictions.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_y_truth = []
    all_y_pred = []

    with torch.no_grad():  # Disable gradient calculations
        for X_batch, y_batch in test_loader:
            # Move data to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Reshape batches if necessary (depending on your training setup)
            bs, nw, seq_len, feature_dim = X_batch.shape
            X_batch = X_batch.view(bs * nw, seq_len, feature_dim)
            y_batch = y_batch.view(bs * nw, -1)

            # Forward pass
            y_pred, _ = model(X_batch)

            # Reshape predictions and ground truth for comparison
            y_pred = y_pred.view(bs, nw, -1)
            y_batch = y_batch.view(bs, nw, -1)

            # Calculate loss for the batch
            loss = criterion(y_pred[:, :, :feature_count], y_batch[:, :, :feature_count])
            test_loss += loss.item()

            # Store predictions and ground truth for further analysis
            all_y_truth.append(y_batch[:, :, :feature_count].cpu())
            all_y_pred.append(y_pred[:, :, :feature_count].cpu())

    # Compute average loss over the test set
    avg_test_loss = test_loss / len(test_loader)

    # Convert predictions and truth to a format suitable for analysis
    all_y_truth = torch.cat(all_y_truth, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()

    print(f"Test Loss: {avg_test_loss:.4f}")

    # Display a few examples
    num_samples = 3
    print("\nSample Predictions:")
    for i in range(min(num_samples, all_y_truth.shape[0])):
        print(f"Sample {i + 1}:")
        print("y_truth:", all_y_truth[i].tolist())
        print("y_pred :", all_y_pred[i].tolist())
        print("-" * 50)

def load_data(data_path):
    data = torch.load(data_path, weights_only=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    return X_train, y_train, X_val, y_val, X_test, y_test

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # List of tensors for inputs
        self.y = y  # List of tensors for targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLSTM(input_sz=8, hidden_sz=64).to(device)

    model_path = '/home/simon/MotionPrediction/Models/AMPM_2.ptm'
    model.load_state_dict(torch.load(model_path))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Load the test data
    data_path = '/home/simon/MotionPrediction/Datasets/lstm_dataset6.pt'
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)

    # Create a DataLoader for the test set
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test the model
    test_model(model, test_loader, criterion, device, feature_count=8)

if __name__ == '__main__':
    main()
