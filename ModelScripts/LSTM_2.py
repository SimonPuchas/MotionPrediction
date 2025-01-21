import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import math
import wandb
import os

# Hyperparameter settings (centralized for easy tuning)
hyperparams = {
    "learning_rate": 0.001,     # 0.01, 0.005, 0.001
    "architecture": "LSTM",
    "dataset": "Self-collected, dataset6",
    "epochs": 110,
    "hidden_size": 64,
    "normalization": "Standardized",
    "batch_size": 32,
    "input_size": 8,
    "scheduler_factor": 0.5,  # Multiply learning rate by this factor
    "scheduler_patience": 5,  # Reduce learning rate after this many epochs with no improvement
    "scheduler_min_lr": 1e-10 # Minimum learning rate
}

# Name of the model file
model_name = "AMPM_001.ptm"

wandb.init(
    project="AMPM",
    config=hyperparams
)


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
            # Batch computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),        # input f(x) = 1 / 1+e^-x
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]), # f(x) = (e^x - e^-x) / (e^x + e^-x)
                torch.sigmoid(gates[:, HS * 3:])     # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # Use the last hidden state for prediction
        prediction = self.fc(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return prediction, (h_t, c_t)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
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
        wandb.log({"Train loss": train_loss})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                bs, nw, seq_len, feature_dim = X_batch.shape
                X_batch = X_batch.view(bs * nw, seq_len, feature_dim)
                y_batch = y_batch.view(bs * nw, -1)

                output, _ = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"Val loss": val_loss})

    return train_losses, val_losses

def load_data(data_path):
    data = torch.load(data_path, weights_only=True)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLSTM(input_sz=hyperparams['input_size'], hidden_sz=hyperparams['hidden_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=hyperparams['scheduler_factor'],
        patience=hyperparams['scheduler_patience'], min_lr=hyperparams['scheduler_min_lr'], verbose=True
    )

    data_path = 'Datasets/lstm_dataset6.pt'
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    train_losses, val_losses = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=hyperparams['epochs']
    )

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    OUTPUT_DIR = 'Models'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, model_name)
    torch.save(model.state_dict(), model_path)
    print('Model saved as:', model_path)


if __name__ == '__main__':
    main()
