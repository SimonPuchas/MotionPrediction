import torch
import torch.nn as nn
import math
import json
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    # Load model from:
    "model_path": "Models/AMPM_001.ptm",
    # Load dataset from:
    "data_path": "Datasets/lstm_dataset6.pt",
    # Save results to:
    "output_results_path": "EvaluationResults/results_001.json"
}

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
            # Batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:])  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # Use the last hidden state for prediction
        prediction = self.fc(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return prediction, (h_t, c_t)

def calculate_metrics(predictions, truths):
    abs_errors = torch.abs(predictions - truths)
    squared_errors = abs_errors ** 2

    mae = abs_errors.mean().item()
    mse = squared_errors.mean().item()

    return mae, mse

def test_model(model, test_loader, criterion, device, feature_count=8):
    model.eval() 
    test_loss = 0.0
    all_y_truth = []
    all_y_pred = []

    with torch.no_grad():  # Disable gradient calculations
        for X_batch, y_batch in test_loader:
            # Move data to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            bs, nw, seq_len, feature_dim = X_batch.shape
            X_batch = X_batch.view(bs * nw, seq_len, feature_dim)
            y_batch = y_batch.view(bs * nw, -1)

            # Forward pass
            y_pred, _ = model(X_batch)

            # Reshape predictions and ground truth for comparison
            y_pred = y_pred.view(bs, nw, -1)
            y_batch = y_batch.view(bs, nw, -1)

            loss = criterion(y_pred[:, :, :feature_count], y_batch[:, :, :feature_count])
            test_loss += loss.item()

            all_y_truth.append(y_batch[:, :, :feature_count].cpu())
            all_y_pred.append(y_pred[:, :, :feature_count].cpu())

    avg_test_loss = test_loss / len(test_loader)

    # Convert predictions and truth to a format suitable for analysis
    all_y_truth = torch.cat(all_y_truth, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()

    mae, mse = calculate_metrics(torch.tensor(all_y_pred), torch.tensor(all_y_truth))

    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # print(f"Mean Squared Error (MSE): {mse:.4f}")

    save_results_as_json(all_y_truth, all_y_pred, CONFIG["output_results_path"], mae, mse, avg_test_loss)

def save_results_as_json(truths, predictions, output_path, mae, mse, avg_test_loss):
    results = {
        "Metrics": {
            "Mean Absolute Error (MAE)": round(mae, 4),
            # "Mean Squared Error (MSE)": round(mse, 4),
            "Average Test Loss (MSE)": round(avg_test_loss, 4)
        },
        "Predictions": []
    }

    movement_id = 1
    for truth, pred in zip(truths, predictions):
        for t_row, p_row in zip(truth, pred):
            differences = [round(float(t) - float(p), 2) for t, p in zip(t_row.tolist(), p_row.tolist())]
            results["Predictions"].append({
                "Movement": movement_id,
                "Ground Truth": [f"{val:+.2f}" for val in t_row.tolist()],
                "Prediction": [f"{val:+.2f}" for val in p_row.tolist()],
                "Difference": [f"{val:+.2f}" for val in differences]
            })
            movement_id += 1

    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {output_path}")

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
    model = CustomLSTM(input_sz=8, hidden_sz=64).to(device)

    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=True))

    criterion = nn.MSELoss()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(CONFIG['data_path'])

    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_model(model, test_loader, criterion, device, feature_count=8)

if __name__ == '__main__':
    main()
