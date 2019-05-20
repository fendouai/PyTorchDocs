from __future__ import print_function
import torch
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
# 重写了数据类型
print(x)
# 结果大小相同
#
print(x.size())
#
y = torch.rand(5, 3)
# #加法方式1
print(x + y)
# #加法方式2
# #print(torch.add(x, y))
#
print(x[:,1])