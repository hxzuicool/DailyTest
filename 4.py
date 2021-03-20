import numpy as np
import torch
from matplotlib import pyplot as plt

x = torch.rand(1, 100)
y = 2*x + 3

plt.scatter(x.numpy(), y.numpy())
plt.show()