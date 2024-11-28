import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class MotionPredictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create a DataLoader for batching
def create_dataloader(X, y, batch_size=8, shuffle=True):
    dataset = MotionPredictionDataset(X, y)
    
    def collate_fn(batch):
        X_batch, y_batch = zip(*batch)
    
        # Pad sequences
        X_batch = pad_sequence(X_batch, batch_first=True, padding_value=0)
        y_batch = torch.stack(y_batch, dim=0)
    
        # Get sequence lengths (the original lengths before padding)
        lengths = torch.tensor([len(seq) for seq in X_batch])
    
        return X_batch, y_batch, lengths
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # Sorting the sequences by length in descending order
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]

        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Packing the padded sequences
        x = pack_padded_sequence(x, sorted_lengths, batch_first=True)
        
        # Passing through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        # Unpacking the sequence output
        h_out, _ = pad_packed_sequence(ula, batch_first=True)
        
        # Taking the output from the last time step
        h_out = h_out[sorted_idx]  # Re-order the output to match original order
        h_out = h_out[:, -1, :]  # Get the output from the last timestep

        # Fully connected layer
        out = self.fc(h_out)

        return out
    
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels, lengths = data
            inputs, labels = inputs.to(device), labels.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss.append(running_loss / len(train_loader))
        
        model.eval()
        running_loss = 0.0
        for i, data in enumerate(val_loader):
            inputs, labels, lengths = data
            inputs, labels = inputs.to(device), labels.to(device), lengths.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
        val_loss.append(running_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}')
        
    return train_loss, val_loss

def load_data():
    data = torch.load('/home/simon/MotionPrediction/Datasets/lstm_dataset4.pt', weights_only=True)
    
    # Unpack the data into X_train, y_train, X_val, y_val, X_test, y_test
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    batch_size = 8
    # Create DataLoader for train, val, and test
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size)

    # defining hyperparameters
    num_classes = 1
    input_size = 8
    hidden_size = 64
    num_layers = 1
    num_epochs = 100
    learning_rate = 0.001

    # defining model
    model = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)

    # defining loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training the model
    train_loss, val_loss = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

    # plotting the training and validation loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
