import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for DataFrame
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math

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
        """Assumes x is of shape (batch, sequence, feature)"""
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

file_path = 'M:\\PyCharm Community Edition 2024.2.4\\Projects\\Motion Prediction Model\\data\\lstm_dataset5.pt'

def load_data(data_path):
    data = torch.load(data_path, weights_only=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_data(file_path)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # List of tensors for inputs
        self.y = y  # List of tensors for targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    # Batch is a list of tuples: (X, y)
    X_batch, y_batch = zip(*batch)

    # Pad sequences to the maximum length in the batch
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=0)

    lengths = torch.tensor([torch.count_nonzero(x[:, 0]) for x in X_batch])

    return X_padded, y_padded, lengths

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Basic training loop
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    train_losses = []
    val_losses = []
    history = {'Epoch': [], 'Train Loss': [], 'Val Loss': []}  # Dictionary to store losses

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch, lengths in train_loader:  # Unpack lengths here
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Merge `num_windows` with `batch_size`
            bs, nw, seq_len, feature_dim = X_batch.shape
            X_batch = X_batch.view(bs * nw, seq_len, feature_dim)

            # Adjust y_batch accordingly
            y_batch = y_batch.view(bs * nw, -1)

            # Pass reshaped inputs to the model
            output, _ = model(X_batch)

            # Compute the loss
            loss = criterion(output, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, lengths in val_loader:  # Unpack lengths here as well
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Merge `num_windows` with `batch_size`
                bs, nw, seq_len, feature_dim = X_batch.shape
                X_batch = X_batch.view(bs * nw, seq_len, feature_dim)

                # Adjust y_batch accordingly
                y_batch = y_batch.view(bs * nw, -1)

                # Pass reshaped inputs to the model
                output, _ = model(X_batch)

                # Compute the loss
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Store losses in history dictionary
        history['Epoch'].append(epoch + 1)
        history['Train Loss'].append(train_loss)
        history['Val Loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses, history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomLSTM(input_sz=8, hidden_sz=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, history = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)

# DataFrame for easier reading
loss_df = pd.DataFrame(history)

# Important losses to keep note of
final_train_loss = loss_df['Train Loss'].iloc[-1]
final_val_loss = loss_df['Val Loss'].iloc[-1]
best_val_loss = loss_df['Val Loss'].min()
best_val_epoch = loss_df['Val Loss'].idxmin() + 1  # Epochs are 1-indexed

# Plotting the losses
fig, ax = plt.subplots(figsize=(10, 6))

# Plot train and validation losses
ax.plot(loss_df['Epoch'], loss_df['Train Loss'], label='Train Loss')
ax.plot(loss_df['Epoch'], loss_df['Val Loss'], label='Val Loss')

# Setting labels and legend
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()

stats_text = (
    f"Final Train Loss: {final_train_loss:.4f}\n"
    f"Final Val Loss: {final_val_loss:.4f}\n"
    f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})"
)
ax.text(
    1.02, 0.5, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='center',
    bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white")
)
# Adjust pos so it looks nice
plt.subplots_adjust(right= 0.7)
plt.show()