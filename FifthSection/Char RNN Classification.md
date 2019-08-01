# 使用字符级RNN进行名字分类
我们将构建和训练字符级RNN来对单词进行分类。字符级RNN将单词作为一系列字符读取，在每一步输出预测和“隐藏状态”，将其先前的隐藏
状态输入至下一时刻。我们将最终时刻输出作为预测结果，即表示该词属于哪个类。

具体来说，我们将在18种语言构成的几千个名字的数据集上训练模型，根据一个名字的拼写预测它是哪种语言的名字：

```buildoutcfg
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Schmidhuber
(-0.19) German
(-2.48) Czech
(-2.68) Dutch
```

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
点击[这里](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前文件夹。

在"`data/names`"文件夹下是名称为"[language].txt"的18个文本文件。每个文件的每一行都有一个名字，它们几乎都是罗马化的文本
（但是我们仍需要将其从Unicode转换为ASCII编码）

我们最终会得到一个语言对应名字列表的字典，`{language: [names ...]}`。通用变量“category”和“line”（例子中的语言和名字单词）
用于以后的可扩展性。

```buildoutcfg
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# 构建category_lines字典，每种语言的名字列表
category_lines = {}
all_categories = []

# 读取文件并分成几行
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
```

* 输出结果：

```buildoutcfg
['data/names/French.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/Polish.txt', 'data/names/Scottish.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Japanese.txt', 'data/names/German.txt', 'data/names/Russian.txt', 'data/names/Korean.txt', 'data/names/Arabic.txt', 'data/names/Greek.txt', 'data/names/Vietnamese.txt', 'data/names/Spanish.txt', 'data/names/Irish.txt']
Slusarski
```
现在我们有了`category_lines`，一个字典变量存储每一种语言及其对应的每一行文本(名字)列表的映射关系。变量`all_categories`是全部
语言种类的列表，变量`n_categories`是语言种类的数量，后续会使用。

```buildoutcfg
print(category_lines['Italian'][:5])
```

* 输出结果：

```buildoutcfg
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']
```

#### **单词转变为张量**
现在我们已经加载了所有的名字，我们需要将它们转换为张量来使用它们。

我们使用大小为`<1 x n_letters>`的“one-hot 向量”表示一个字母。一个one-hot向量所有位置都填充为0，并在其表示的字母的位置表示为1，
例如`"b" = <0 1 0 0 0 ...>`.（字母b的编号是2，第二个位置是1，其他位置是0）

我们使用一个`<line_length x 1 x n_letters>`的2D矩阵表示一个单词

额外的1维是batch的维度，PyTorch默认所有的数据都是成batch处理的。我们这里只设置了batch的大小为1。

```buildoutcfg
import torch

# 从all_letters中查找字母索引，例如 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# 仅用于演示，将字母转换为<1 x n_letters> 张量
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 将一行转换为<line_length x 1 x n_letters>，
# 或一个0ne-hot字母向量的数组
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())
```

* 输出结果：
```buildoutcfg
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
torch.Size([5, 1, 57])
```

## 2.构造神经网络
在autograd之前，要在Torch中构建一个可以复制之前时刻层参数的循环神经网络。layer的隐藏状态和梯度将交给计算图自己处理。这意味着
你可以像实现的常规的 feed-forward 层一样，以很纯粹的方式实现RNN。

这个RNN组件 (几乎是从这里复制的[the PyTorch for Torch users tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net))
仅使用两层 linear 层对输入和隐藏层做处理,在最后添加一层 LogSoftmax 层预测最终输出。

```buildoutcfg
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
```

要运行此网络的一个步骤，我们需要传递一个输入（在我们的例子中，是当前字母的Tensor）和一个先前隐藏的状态（我们首先将其初始化为零）。
我们将返回输出（每种语言的概率）和下一个隐藏状态（为我们下一步保留使用）。

```buildoutcfg
input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
```

为了提高效率，我们不希望为每一步都创建一个新的Tensor，因此我们将使用`lineToTensor`函数而不是`letterToTensor`函数，并使用切片
方法。这一步可以通过预先计算批量的张量进一步优化。

```buildoutcfg
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)
```

* 输出结果：

```buildoutcfg
tensor([[-2.8857, -2.9005, -2.8386, -2.9397, -2.8594, -2.8785, -2.9361, -2.8270,
         -2.9602, -2.8583, -2.9244, -2.9112, -2.8545, -2.8715, -2.8328, -2.8233,
         -2.9685, -2.9780]], grad_fn=<LogSoftmaxBackward>)
```

可以看到输出是一个`<1 x n_categories>`的张量，其中每一条代表这个单词属于某一类的可能性（越高可能性越大）。

## 2.训练
#### 2.1 训练前的准备
进行训练步骤之前我们需要构建一些辅助函数。
* 第一个是当我们知道输出结果对应每种类别的可能性时，解析神经网络的输出。我们可以使用
`Tensor.topk`函数得到最大值在结果中的位置索引：

```buildoutcfg
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))
```

* 输出结果：

```buildoutcfg
('Arabic', 13)
```

* 第二个是我们需要一种快速获取训练示例（得到一个名字及其所属的语言类别）的方法：

```buildoutcfg
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
```

* 输出结果：

```buildoutcfg
category = Dutch / line = Tholberg
category = Irish / line = Murphy
category = Vietnamese / line = An
category = German / line = Von essen
category = Polish / line = Kijek
category = Scottish / line = Bell
category = Czech / line = Marik
category = Korean / line = Jeong
category = Korean / line = Choe
category = Portuguese / line = Alves
```

#### 2.2 训练神经网络

现在，训练过程只需要向神经网络输入大量的数据，让它做出预测，并将对错反馈给它。

`nn.LogSoftmax`作为最后一层layer时，`nn.NLLLoss`作为损失函数是合适的。

```buildoutcfg
criterion = nn.NLLLoss()
```
训练过程的每次循环将会发生：

* 构建输入和目标张量<br/>
* 构建0初始化的隐藏状态<br/>
* 读入每一个字母<br/>
&emsp; * 将当前隐藏状态传递给下一字母<br/>
* 比较最终结果和目标<br/>
* 反向传播<br/>
* 返回结果和损失<br/>

```buildoutcfg
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数的梯度添加到其值中，乘以学习速率
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()
```

现在我们只需要准备一些例子来运行程序。由于train函数同时返回输出和损失，我们可以打印其输出结果并跟踪其损失画图。由于有1000个
示例，我们每`print_every`次打印样例，并求平均损失。

```buildoutcfg
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# 跟踪绘图的损失
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # 打印迭代的编号，损失，名字和猜测
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 将当前损失平均值添加到损失列表中
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
```

* 输出结果：

```buildoutcfg
5000 5% (0m 8s) 2.7792 Verdon / Scottish ✗ (English)
10000 10% (0m 16s) 2.0748 Campos / Greek ✗ (Portuguese)
15000 15% (0m 25s) 2.0458 Kuang / Vietnamese ✗ (Chinese)
20000 20% (0m 33s) 1.1703 Nghiem / Vietnamese ✓
25000 25% (0m 41s) 2.6035 Boyle / English ✗ (Scottish)
30000 30% (0m 50s) 2.2823 Mozdzierz / Dutch ✗ (Polish)
35000 35% (0m 58s) nan Lagana / Irish ✗ (Italian)
40000 40% (1m 6s) nan Simonis / Irish ✗ (Dutch)
45000 45% (1m 15s) nan Nobunaga / Irish ✗ (Japanese)
50000 50% (1m 23s) nan Ingermann / Irish ✗ (English)
55000 55% (1m 31s) nan Govorin / Irish ✗ (Russian)
60000 60% (1m 39s) nan Janson / Irish ✗ (German)
65000 65% (1m 48s) nan Tsangaris / Irish ✗ (Greek)
70000 70% (1m 56s) nan Vlasenkov / Irish ✗ (Russian)
75000 75% (2m 4s) nan Needham / Irish ✗ (English)
80000 80% (2m 12s) nan Matsoukis / Irish ✗ (Greek)
85000 85% (2m 21s) nan Koo / Irish ✗ (Chinese)
90000 90% (2m 29s) nan Novotny / Irish ✗ (Czech)
95000 95% (2m 37s) nan Dubois / Irish ✗ (French)
100000 100% (2m 45s) nan Padovano / Irish ✗ (Italian)
```

#### 2.3 绘画出结果
从`all_losses`得到历史损失记录，反映了神经网络的学习情况：

```buildoutcfg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
```

![](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_001.png)

## 3.评价结果
为了了解网络在不同类别上的表现，我们将创建一个混淆矩阵，显示每种语言（行）和神经网络将其预测为哪种语言（列）。为了计算混淆矩
阵，使用`evaluate()`函数处理了一批数据，`evaluate()`函数与去掉反向传播的`train()`函数大体相同。

```buildoutcfg
# 在混淆矩阵中跟踪正确的猜测
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# 只需返回给定一行的输出
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 查看一堆正确猜到的例子和记录
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 通过将每一行除以其总和来归一化
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 设置绘图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 设置轴
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 每个刻度线强制标签
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
```

![](https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/029a9d26725997aae97e9e3f6f10067f.jpg)

你可以从主轴线以外挑出亮的点，显示模型预测错了哪些语言，例如汉语预测为了韩语，西班牙预测为了意大利。看上去在希腊语上效果很好，
在英语上表现欠佳。（可能是因为英语与其他语言的重叠较多）。

#### **处理用户输入**
```buildoutcfg
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # 获得前N个类别
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
```

* 输出结果：
```buildoutcfg
> Dovesky
(-0.74) Russian
(-0.77) Czech
(-3.31) English

> Jackson
(-0.80) Scottish
(-1.69) English
(-1.84) Russian

> Satoshi
(-1.16) Japanese
(-1.89) Arabic
(-1.90) Polish
```

最终版的脚本[in the Practical PyTorch repo](https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification)
将上述代码拆分为几个文件：

* data.py (读取文件)<br/>
* model.py (构造RNN网络)<br/>
* train.py (运行训练过程)<br/>
* predict.py (在命令行中和参数一起运行predict()函数)<br/>
* server.py (使用bottle.py构建JSON API的预测服务)<br/>

运行`train.py`来训练和保存网络

将`predict.py`和一个名字的单词一起运行查看预测结果 :
```buildoutcfg
$ python predict.py Hazaki
(-0.42) Japanese
(-1.39) Polish
(-3.51) Czech
```
运行`server.py`并访问http://localhost:5533/Yourname 得到JSON格式的预测输出

## 4.练习
* 尝试其它 （类别->行） 格式的数据集，比如:

&emsp; * 任何单词 -> 语言<br/>
&emsp; * 姓名 -> 性别<br/>
&emsp; * 角色姓名 -> 作者<br/>
&emsp; * 页面标题 -> blog 或 subreddit<br/>

* 通过更大和更复杂的网络获得更好的结果<br/>
&emsp; * 增加更多linear层<br/>
&emsp; * 尝试 nn.LSTM 和 nn.GRU 层<br/>
&emsp; * 组合这些 RNN构造更复杂的神经网络<br/>
