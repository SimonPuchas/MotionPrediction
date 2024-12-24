
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import random

# Create a list of variable length tensors
t1 = torch.randint(10, (220, 10, 8))
t2 = torch.randint(10, (2,10, 8))
t3 = torch.randint(10, (3,10, 8))

t_comb = pad_sequence([t1, t2, t3], batch_first=True, padding_value=0)
#print(t_comb)
print(t_comb.shape)

t_packed = pack_padded_sequence(t_comb, lengths=[len(t1), len(t2), len(t3)], batch_first=True, enforce_sorted=False)
#print(t_packed)
print(t_packed.data.shape)

