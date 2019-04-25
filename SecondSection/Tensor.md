# Tensor（张量）
Tensor（张量）是PyTorch最基本的操作对象，表示的是一个多维的矩阵，与python的numpy是对应的，而且PyTorch的Tensor可以和numpy
的ndarray相互转换，唯一不同的是PyTorch可以在GPU上运行，而numpy的ndarray只能在CPU上运行。
Tensor的基本数据类型包括
* 32位浮点型  torch.FloatTensor（默认）
* 64位浮点型 torch.DoubleTensor
* 16位整型  torch.ShortTensor
* 32位整型 torch.IntTensor
* 64位整型 torch.LongTensor

## 对Tensor的基本操作
首先我们必须先引入torch包：
```
from _future_ import print_function
import torch
```
### 1. 构造