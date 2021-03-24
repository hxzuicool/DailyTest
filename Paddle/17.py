import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

rnn = nn.LSTM(4, 10, 2)  # input_size, hidden_size, num_layers
input = torch.rand(5, 3, 4)  # sqe_len, batch_size, input_size
# print(input)
h0 = torch.rand(2, 3, 10)
print(h0)
c0 = torch.rand(2, 3, 10)
output, (hn, cn) = rnn(input, (h0, c0))

# print(output)

print(783/26)
