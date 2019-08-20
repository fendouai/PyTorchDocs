可选择：数据并行处理（文末有完整代码下载）
作者：Sung Kim 和 Jenny Kang

在这个教程中，我们将学习如何用 DataParallel 来使用多 GPU。
通过 PyTorch 使用多个 GPU 非常简单。你可以将模型放在一个 GPU：
```python
 device = torch.device("cuda:0")
 model.to(device)
```
然后，你可以复制所有的张量到 GPU：
```python

mytensor = my_tensor.to(device)

```
请注意，只是调用 my_tensor.to(device) 返回一个 my_tensor 新的复制在GPU上，而不是重写 my_tensor。你需要分配给他一个新的张量并且在 GPU 上使用这个张量。

在多 GPU 中执行前馈，后馈操作是非常自然的。尽管如此，PyTorch 默认只会使用一个 GPU。通过使用 DataParallel 让你的模型并行运行，你可以很容易的在多 GPU 上运行你的操作。
```python
model = nn.DataParallel(model)

```
这是整个教程的核心，我们接下来将会详细讲解。
引用和参数

引入 PyTorch 模块和定义参数
```python
 import torch
 import torch.nn as nn
 from torch.utils.data import Dataset, DataLoader

```
# 参数
```python
 input_size = 5
 output_size = 2

 batch_size = 30
 data_size = 100
```
设备

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

```
实验（玩具）数据

生成一个玩具数据。你只需要实现 getitem.
```python

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),batch_size=batch_size, shuffle=True)
```
简单模型

为了做一个小 demo，我们的模型只是获得一个输入，执行一个线性操作，然后给一个输出。尽管如此，你可以使用 DataParallel   在任何模型(CNN, RNN, Capsule Net 等等.)

我们放置了一个输出声明在模型中来检测输出和输入张量的大小。请注意在 batch rank 0 中的输出。
```python

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

创建模型并且数据并行处理

这是整个教程的核心。首先我们需要一个模型的实例，然后验证我们是否有多个 GPU。如果我们有多个 GPU，我们可以用 nn.DataParallel 来   包裹 我们的模型。然后我们使用 model.to(device) 把模型放到多 GPU 中。
```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```
输出：

```python

Let's use 2 GPUs!

```
运行模型：
现在我们可以看到输入和输出张量的大小了。
```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```
输出：
<pre>In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])</pre>
结果：

如果你没有 GPU 或者只有一个 GPU，当我们获取 30 个输入和 30 个输出，模型将期望获得 30 个输入和 30 个输出。但是如果你有多个 GPU ，你会获得这样的结果。

多 GPU

如果你有 2 个GPU，你会看到：
<div id="gpus" class="section">
<pre><span class="c1"># on 2 GPUs</span>
<span class="n">Let</span><span class="s1">'s use 2 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
</div>
&nbsp;

如果你有 3个GPU，你会看到：
<pre><span class="n">Let</span><span class="s1">'s use 3 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
如果你有 8个GPU，你会看到：
<pre><span class="n">Let</span><span class="s1">'s use 8 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
<h2>总结</h2>
数据并行自动拆分了你的数据并且将任务单发送到多个 GPU 上。当每一个模型都完成自己的任务之后，DataParallel 收集并且合并这些结果，然后再返回给你。

更多信息，请访问：
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

下载 Python 版本完整代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/data_parallel_tutorial.py_.zip">data_parallel_tutorial.py</a>

下载 jupyter notebook 版本完整代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/data_parallel_tutorial.ipynb_.zip">data_parallel_tutorial.ipynb</a>

加入 PyTorch 交流 QQ 群：

<img class="alignnone wp-image-47 size-full" src="http://pytorchchina.com/wp-content/uploads/2018/12/WechatIMG1311.png" alt="" width="540" height="740" />

&nbsp;

&nbsp;

&nbsp;

&nbsp;
