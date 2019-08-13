# 使用字符级RNN生成名字
在上一个[教程](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)中，
中我们使用RNN网络对名字所属的语言进行分类。这一次我们会反过来根据语言生成名字。
```buildoutcfg
> python sample.py Russian RUS
Rovakov
Uantov
Shavakov

> python sample.py German GER
Gerren
Ereng
Rosher

> python sample.py Spanish SPA
Salla
Parer
Allan

> python sample.py Chinese CHI
Chan
Hang
Iun
```
我们仍使用只有几层线性层的小型RNN。最大的区别在于，这里不是在读取一个名字的所有字母后预测类别，而是输入一个类别之后在每一时刻
输出一个字母。循环预测字符以形成语言通常也被称为“语言模型”。（也可以将字符换成单词或更高级的结构进行这一过程）

* **阅读建议**

开始本教程前，你已经安装好了PyTorch，并熟悉Python语言，理解“张量”的概念：

* https://pytorch.org/ PyTorch 安装指南
* [Deep Learning with PyTorch：A 60 Minute Blitz ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html):PyTorch的基本入门教程
* [Learning PyTorch with Examples](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/LearningPyTorch.md):得到深层而广泛的概述
* [PyTorch for Former Torch Users Lua Torch](https://pytorch.org/tutorials/beginner/former_torchies_tutorial.html):如果你曾是一个Lua张量的使用者

事先学习并了解RNN的工作原理对理解这个例子十分有帮助:

* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)展示了很多实际的例子
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)是关于LSTM的，但也提供有关RNN的说明

## 1.准备数据
点击这里[下载数据](https://download.pytorch.org/tutorial/data.zip)并将其解压到当前文件夹。

有关此过程的更多详细信息，请参阅上一个教程。简而言之，有一些纯文本文件`data/names/[Language].txt`，它们的每行都有一个名字。
我们按行将文本按行分割得到一个数组，将Unicode编码转化为ASCII编码，最终得到`{language: [names ...]}`格式存储的字典变量。

```buildoutcfg
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# 读取文件并分成几行
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 构建category_lines字典，列表中的每行是一个类别
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))
```

* 输出结果
```buildoutcfg
# categories: 18 ['French', 'Czech', 'Dutch', 'Polish', 'Scottish', 'Chinese', 'English', 'Italian', 'Portuguese', 'Japanese', 'German', 'Russian', 'Korean', 'Arabic', 'Greek', 'Vietnamese', 'Spanish', 'Irish']
O'Neal
```

## 2.构造神经网络
这个神经网络比[上一个RNN教程](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/char_rnn_generation_tutorial.md#Creating-the-Network)
中的网络增加了额外的类别张量参数，该参数与其他输入连接在一起。类别可以像字母一样组成 one-hot 向量构成张量输入。

我们将输出作为下一个字母是什么的可能性。采样过程中，当前输出可能性最高的字母作为下一时刻输入字母。

在组合隐藏状态和输出之后我们增加了第二个linear层`o2o`，使模型的性能更好。当然还有一个dropout层，参考这篇论文[随机将输入部分替换为0](https://arxiv.org/abs/1207.0580)
给出的参数（dropout=0.1）来模糊处理输入防止过拟合。
我们将它添加到网络的末端，故意添加一些混乱使采样特征增加。
```buildoutcfg
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
```

## 3.训练
#### 3.1 训练准备
首先，构造一个可以随机获取成对训练数据(category, line)的函数。
```buildoutcfg
import random

# 列表中的随机项
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# 从该类别中获取随机类别和随机行
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line
```

对于每个时间步长（即，对于要训练单词中的每个字母），网络的输入将是“`（类别，当前字母，隐藏状态）`”，输出将是“`（下一个字母，
下一个隐藏状态）`”。因此，对于每个训练集，我们将需要类别、一组输入字母和一组输出/目标字母。

在每一个时间序列，我们使用当前字母预测下一个字母，所以训练用的字母对来自于一个单词。例如 对于 "`ABCD<EOS>`"，我们将创建
(“A”，“B”)，（“B”，“C”），（“C”，“D”），（“D”，“EOS”））。

类别张量是一个`<1 x n_categories>`尺寸的[one-hot张量](https://en.wikipedia.org/wiki/One-hot)。训练时，我们在每一个时间序
列都将其提供给神经网络。这是一种选择策略，也可选择将其作为初始隐藏状态的一部分，或者其他什么结构。
```buildoutcfg
# 类别的One-hot张量
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# 用于输入的从头到尾字母（不包括EOS）的one-hot矩阵
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# 用于目标的第二个结束字母（EOS）的LongTensor
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
```

为了方便训练，我们将创建一个`randomTrainingExample`函数，该函数随机获取（类别，行）的对并将它们转换为所需要的（类别，输入，
目标）格式张量。
```buildoutcfg
# 从随机(类别，行)对中创建类别，输入和目标张量
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor
```

#### 3.2 训练神经网络
和只使用最后一个时刻输出的分类任务相比，这次我们每一个时间序列都会进行一次预测，所以每一个时间序列我们都会计算损失。

autograd 的神奇之处在于您可以在每一步中简单地累加这些损失，并在最后反向传播。
```buildoutcfg
criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)
```

为了跟踪训练耗费的时间，我添加一个`timeSince（timestamp）`函数，它返回一个人类可读的字符串：

```buildoutcfg
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
```

训练过程和平时一样。多次运行训练，等待几分钟，每`print_every`次打印当前时间和损失。在`all_losses`中保留每`plot_every`次的平
均损失，以便稍后进行绘图。

```buildoutcfg
rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
```

* 输出结果：

```buildoutcfg
0m 23s (5000 5%) 3.1569
0m 43s (10000 10%) 2.3132
1m 3s (15000 15%) 2.5069
1m 24s (20000 20%) 1.3100
1m 44s (25000 25%) 3.6083
2m 4s (30000 30%) 3.5398
2m 24s (35000 35%) 2.4387
2m 44s (40000 40%) 2.2262
3m 4s (45000 45%) 2.6500
3m 24s (50000 50%) 2.4559
3m 44s (55000 55%) 2.5030
4m 4s (60000 60%) 2.9417
4m 24s (65000 65%) 2.1571
4m 44s (70000 70%) 1.7415
5m 4s (75000 75%) 2.3649
5m 24s (80000 80%) 3.0096
5m 44s (85000 85%) 1.9196
6m 4s (90000 90%) 1.9468
6m 25s (95000 95%) 2.1522
6m 45s (100000 100%) 2.0344
```

#### 3.3 损失数据作图
从`all_losses`得到历史损失记录，反映了神经网络的学习情况：

```buildoutcfg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
```

![](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_generation_tutorial_001.png)

## 4.网络采样
我们每次给网络提供一个字母并预测下一个字母是什么，将预测到的字母继续输入，直到得到EOS字符结束循环。

* 用输入类别、起始字母和空隐藏状态创建输入张量。

* 用起始字母构建一个字符串变量 output_name

* 得到最大输出长度，<br/>
&emsp; * 将当前字母传入神经网络<br/>
&emsp; * 从前一层得到下一个字母和下一个隐藏状态<br/>
&emsp; * 如果字母是EOS，在这里停止<br/>
&emsp; * 如果是一个普通的字母，添加到output_name变量并继续循环<br/>
* 返回最终得到的名字单词<br/>

另一种策略是，不必给网络一个起始字母，而是在训练中提供一个“字符串开始”的标记，并让网络自己选择起始的字母。

```buildoutcfg
max_length = 20

# 来自类别和首字母的样本
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# 从一个类别和多个起始字母中获取多个样本
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')
```

* 输出结果：

```buildoutcfg
Rovanik
Uakilovev
Shaveri
Garter
Eren
Romer
Santa
Parera
Artera
Chan
Ha
Iua
```

## 练习
* 尝试其它 （类别->行） 格式的数据集，比如:<br/>
&emsp; * 系列小说 -> 角色名称<br/>
&emsp; * 词性 -> 单词<br/>
&emsp; * 国家 -> 城市<br/>
* 尝试“start of sentence” 标记，使采样的开始过程不需要指定起始字母<br/>
* 通过更大和更复杂的网络获得更好的结果<br/>
&emsp; * 尝试 nn.LSTM 和 nn.GRU 层<br/>
&emsp; * 组合这些 RNN构造更复杂的神经网络<br/>