import torch
import numpy as np

x = torch.Tensor([2])

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y = torch.mul(w, x)
z = torch.add(y, b)

print("x,w,b的require_grad属性分别为：{}, {}, {}".format(x.requires_grad, w.requires_grad, b.requires_grad))

z.backward()
print([x, w, b, z])

print(np.random.rand(2, 1))
