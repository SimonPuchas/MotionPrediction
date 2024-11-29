import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

'''
tensor = [10, 23, 12]
tensor2 = [10, 23, 12, 52, 2]
tensor3 = [10, 23, 12, 3]

max_length = max(len(tensor), len(tensor2), len(tensor3))

t_pad = pad_sequence([torch.tensor(tensor), torch.tensor(tensor2), torch.tensor(tensor3)], batch_first=True, padding_value=0)

print(t_pad)

t_pack = pack_padded_sequence(t_pad, lengths=[len(tensor), len(tensor2), len(tensor3)], batch_first=True, enforce_sorted=False)

print(t_pack)'''

t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t2 = torch.tensor([[10, 20, 42], [40, 50, 42]])
t3 = torch.tensor([[100, 4, 42], [400, 424, 42], [600, 11, 41], [900, 4, 6]])

t_pad = pad_sequence([t1, t2, t3], batch_first=True, padding_value=0)
print(t_pad)

t_pack = pack_padded_sequence(t_pad, lengths=[len(t1), len(t2), len(t3)], batch_first=True, enforce_sorted=False)
print(t_pack)