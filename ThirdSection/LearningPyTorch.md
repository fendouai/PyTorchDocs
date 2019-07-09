# PyTorch之小试牛刀
### 1 PyTorch的核心是两个主要特征：
* 一个n维张量，类似于numpy，但可以在GPU上运行
* 自动区分建立和训练神经网络

本章节将使用完全连接的ReLU网络作为运行示例。网络将具有单个隐藏层，并且将通过最小化网络输出与真实输出之间的欧几里德距离来训练梯度下降以适合随机数据。

### 2 Tensors
#### 2.1 Warm-up: numpy
在介绍PyTorch之前，本章节将首先使用numpy实现网络。
Numpy提供了一个n维数组对象，以及许多用于操作这些数组的
函数。Numpy是科学计算的通用框架; 它对计算图，深度学习
或渐变都一无所知。但是，通过使用numpy操作手动实现网络的
前向和后向传递，这里可以轻松地使用numpy将两层网络适
配到随机数据：
```
# -*- coding: utf-8 -*-
import numpy as np

# N是批量大小; D_in是输入维度;
# 49/5000 H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算和打印损失
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
Numpy是一个很棒的框架，但它不能利用GPU来加速其数值计算。
对于现代深度神经网络，GPU通常提供50倍或更高的加速，
因此不幸的是，numpy对于现代深度学习来说还不够。 

在这里，本章介绍最基本的PyTorch概念：
**Tensor**。PyTorch Tensor在概念上与numpy数组相同：
Tensor是一个n维数组，PyTorch提供了许多用于在这些
Tensors上运算的函数。在幕后，Tensors可以跟踪计算图和渐变，
但它们也可用作科学计算的通用工具。 

与numpy不同，PyTorch Tensors可以利用GPU加速其数值计算。要在GPU上运行PyTorch Tensor，只需将其转换为新的数据类型即可。  

在这里，本章使用PyTorch Tensors将双层网络与随机数据相匹配。
就像上面的numpy示例一样，这里需要手动实现网络中的前向和
后向传递：
```
# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

#创建随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
### 3 Autograd
在上面的例子中，需要手动实现神经网络的前向和后向
传递。手动实现反向传递对于小型双层网络来说并不是什么大问
题，但对于大型复杂网络来说很快就会变得非常繁琐。  

但是可以使用自动微分 来自动计算神经网络中的后向传递。
```PyTorch```中的 ```autograd ```包提供了这个功能。
使用```autograd ```时，网络的正向传递将定义 计算图形 ;
 图中的节点将是张量，边将是从输入张量产生输出```Tensor```的函数。
通过此图反向传播，您可以轻松计算渐变。

这听起来很复杂，在实践中使用起来非常简单。
每个```Tensor ```表示计算图中的节点。
如果x是T```ensor x.requires_grad=True```那么```x.grad```是
另一个```Tensor```持有```x```相对于某个标量值的梯度。

在这里，使用```PyTorch Tensors```和```autograd```
来实现双层网络;这样就不再需要手动实现通过网络
的反向传递：
```
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机Tensors以保持输入和输出。
# 设置requires_grad = False表示我们不需要计算渐变
# 在向后传球期间对于这些Tensors。
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 为权重创建随机Tensors。
# 设置requires_grad = True表示我们想要计算渐变
# 在向后传球期间尊重这些张贴。
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：使用Tensors上的操作计算预测y; 
    # 这些与我们使用Tensors计算正向传递的操作完全相同，但我们不需要保留对中间值的引用，因为我们没有手动实现反向传递。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors上的操作计算和打印丢失。
    # 现在损失是  a Tensor of shape (1,)
    # loss.item() 得到损失中的标量值。
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算向后传递。 此调用将使用requires_grad = True计算所有Tensors 的损失梯度。
    # 在此调用之后，w1.grad和w2.grad将分别持有相对于w1和w2的损失梯度的Tensors 。
    loss.backward()

    # 使用渐变下降手动更新权重.包裹在 torch.no_grad()
    # 因为权重有requires_grad = True，但我们不需要在autograd中跟踪它。
    # 另一种方法是对weight.data和weight.grad.data进行操作.
    # 回想一下，tensor.data给出了一个tensor，它与tensor共享存储，但不跟踪历史记录。
    # 您也可以使用torch.optim.SGD来实现这一目标。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 更新权重后手动将渐变归零
        w1.grad.zero_()
        w2.grad.zero_()
```