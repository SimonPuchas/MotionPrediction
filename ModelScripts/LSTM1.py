import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # List of tensors for inputs
        self.y = y  # List of tensors for targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Custom Collate Function to handle variable sequence lengths
def collate_fn(batch):
    # Batch is a list of tuples: (X, y)
    X_batch, y_batch = zip(*batch)

    # Pad sequences to the maximum length in the batch
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=0)

    lengths = torch.tensor([torch.count_nonzero(x[:, 0]) for x in X_batch])
    
    return X_padded, y_padded, lengths

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, lengths):
        # Pack the padded sequence
        packed_X = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass the packed sequence through the LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_X)

        # Unpack the output back into padded sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Pass through the fully connected layer
        output = self.fc(output)
        
        return output
    
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, lengths in train_loader:  # Unpack lengths here
            X_batch, y_batch, lengths = X_batch.to(device), y_batch.to(device), lengths.to(device)
            
            # Pass the batch through the model
            output = model(X_batch, lengths)
            
            # Calculate the loss
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
                X_batch, y_batch, lengths = X_batch.to(device), y_batch.to(device), lengths.to(device)
                
                output = model(X_batch, lengths)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def load_data(data_path):
    data = torch.load(data_path, weights_only=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(input_size=8, hidden_size=64, output_size=8).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_path = '/home/simon/MotionPrediction/Datasets/lstm_dataset5.pt'

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    batch_size = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
