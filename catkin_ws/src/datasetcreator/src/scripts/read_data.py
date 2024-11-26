import torch

# Replace with the path to your .pt file
file_path = '/home/simon/MotionPrediction/lstm_dataset.pt'

# To load a model (if it's a saved model):
# model = torch.load(file_path)
# model.eval()  # Set the model to evaluation mode if you're going to use it for inference

# To load a tensor (if it's a saved tensor):
tensor = torch.load(file_path)
X = tensor['X']
y = tensor['y']

#print(tensor)
#print(X[0].shape)   # shape of the first window in X
#print(y[0].shape)   # shape of the first window in y
#print(len(X))    # number of windows in X
#print(len(y))    # number of windows in y
print(X[1]) # prints the corresponding window/tensor
print(y[1]) # prints the corresponding window/tensor