from __future__ import print_function
import torch


x = torch.randn((3), dtype=torch.float32, requires_grad=True)
y = torch.randn((3), dtype=torch.float32, requires_grad=True)
z = torch.randn((3), dtype=torch.float32, requires_grad=True)
t = x + y
loss = t.dot(z)  # 求向量的内积

loss.backward(retain_graph=True)
print(z, x.grad, y.grad)  # 预期打印出的结果都一样
print(t, z.grad)  # 预期打印出的结果都一样
print(t.grad)