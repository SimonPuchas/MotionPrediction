import torch

data = torch.load('/home/simon/MotionPrediction/Datasets/lstm_dataset4.pt', weights_only=True)
print(torch.is_tensor(data['X_test'][0]))
#print(data['X'][0].shape)
#print(data['y'][0].shape)
#print(data['X'][1].shape)
print(len(data['X_train']))
print(len(data['X_test']))
print(len(data['X_val']))
#print(data['X_test'][0])

