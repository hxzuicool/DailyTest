import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt


lr = 0.01
batch_size = 32
epoch = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

