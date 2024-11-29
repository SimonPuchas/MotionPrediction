import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. Custom Dataset
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
    #lengths = torch.sum(X_padded != 0, dim=1)
    
    return X_padded, y_padded, lengths

data = torch.load('/home/simon/MotionPrediction/Datasets/lstm_dataset4.pt', weights_only=True)
    
# Unpack the data into X_train, y_train, X_val, y_val, X_test, y_test
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# 3. Create Datasets
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

# 4. Create DataLoader
batch_size = 4  # Adjust according to your needs

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 5. Now you can iterate over batches in your training loop
for X_batch, y_batch, lengths in train_loader:
    print(X_batch.shape)  # Check the shape of padded inputs
    print(y_batch.shape)  # Check the shape of padded targets
    print("lenghts", lengths.shape)  # Check the lengths of sequences

