# 使用Sequence2Sequence网络和注意力进行翻译
在这个项目中，我们将讲解使用神经网络将法语翻译成英语。

```buildoutcfg
[KEY: > input, = target, < output]

> il est en train de peindre un tableau .
= he is painting a picture .
< he is painting a picture .

> pourquoi ne pas essayer ce vin delicieux ?
= why not try that delicious wine ?
< why not try that delicious wine ?

> elle n est pas poete mais romanciere .
= she is not a poet but a novelist .
< she not not a poet but a novelist .

> vous etes trop maigre .
= you re too skinny .
< you re all alone .
```

...取得了不同程度的成功。

这可以通过序列到序列网络来实现，其中两个递归神经网络一起工作以将一个序列转换成另一个序列。编码器网络将输入序列压缩成向量，并
且解码器网络将该向量展开成新的序列。

![](https://pytorch.org/tutorials/_images/seq2seq.png)

* **阅读建议**

开始本教程前，你已经安装好了PyTorch，并熟悉Python语言，理解“张量”的概念：

* https://pytorch.org/ PyTorch 安装指南
* [Deep Learning with PyTorch：A 60 Minute Blitz ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html):PyTorch的基本入门教程
* [Learning PyTorch with Examples](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/LearningPyTorch.md):得到深层而广泛的概述
* [PyTorch for Former Torch Users Lua Torch](https://pytorch.org/tutorials/beginner/former_torchies_tutorial.html):如果你曾是一个Lua张量的使用者

事先学习并了解序列到序列网络的工作原理对理解这个例子十分有帮助:

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)

您还可以找到之前有关[Classifying Names with a Character-Level RNN](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Char%20RNN%20Classification.md)和 
[Generating Names with a Character-Level RNN](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Char%20RNN%20Generation.MD)
的教程，因为这些概念分别与编码器和解码器模型非常相似。

更多信息，请阅读介绍这些主题的论文：

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)

# 1.导入必须的包
```buildoutcfg
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

# 2.加载数据文件
该项目的数据是成千上万的英语到法语的翻译对的集合。

关于Open Data Stack Exchange的这个问题，开放式翻译网站 https://tatoeba.org/给出了指导，该网站的下载位于https://tatoeba.org/eng/downloads  
- 更好的是，有人做了额外的拆分工作，将语言对分成单独的文本文件：https：//www.manythings.org/anki/

英语到法语对因为太大而无法包含在repo中，因此下载到data / eng-fra.txt再继续进行后续步骤。该文件是以制表符分隔的翻译对列表：
```buildoutcfg
I am cold.    J'ai froid.
```

> 注意：从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据并将其解压缩到当前目录。

与字符级RNN教程中使用的字符编码类似，我们将语言中的每个单词表示为one-hot向量或零的巨向量，除了单个字符（在单词的索引处）。
与语言中可能存在的几十个字符相比，还有更多的字，因此编码向量很大。然而，我们投机取巧并修剪数据，每种语言只使用几千个单词。

![](https://pytorch.org/tutorials/_images/word-encoding.png)

我们将需要每个单词的唯一索引，以便稍后用作网络的输入和目标。为了跟踪所有这些，我们将使用一个名为`Lang`的辅助类，它具有
word→index（`word2index`）和index→word（`index2word`）的字典，以及用于稍后替换稀有单词的每个单词`word2count`的计数。

```buildoutcfg
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```

这些文件都是Unicode格式，为了简化我们将Unicode字符转换为ASCII，使所有内容都小写，并去掉大多数标点符号。

```buildoutcfg
# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写，修剪和删除非字母字符


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```

#### 2.1 读取数据文件
要读取数据文件，我们将文件拆分为行，然后将行拆分成对。 这些文件都是英语→其他语言，所以如果我们想翻译其他语言→英语，我添加
`reverse`标志来反转对。

```buildoutcfg
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # 读取文件并分成几行
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 将每一行拆分成对并进行标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 反向对，使Lang实例
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
```
由于有很多例句，我们想快速训练，我们会将数据集修剪成相对简短的句子。 这里最大长度是10个单词（包括结束标点符号），我们将过滤到
转换为“我是”或“他是”等形式的句子（考虑先前替换的撇号）。

```buildoutcfg
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
```

准备数据的完整过程是：

* 读取文本文件并拆分成行，将行拆分成对
* 规范化文本，按长度和内容进行过滤
* 从成对的句子中制作单词列表

```buildoutcfg
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
```

* 输出结果：
```buildoutcfg
Reading lines...
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4345
eng 2803
['nous nous deshabillons .', 'we re undressing .']
```

## 3.Seq2Seq模型
递归神经网络（RNN）是一种对序列进行操作的网络，它使用自己的输出作为后续步骤的输入。

[Sequence to Sequence network](https://arxiv.org/abs/1409.3215)(seq2seq网络)或[Encoder Decoder network(https://arxiv.org/pdf/1406.1078v3.pdf)
是由称为编码器和解码器的两个RNN组成的模型。编码器读取输入序列并输出单个向量，并且解码器读取该向量以产生输出序列。

![](https://pytorch.org/tutorials/_images/seq2seq.png)

与使用单个RNN的序列预测不同，其中每个输入对应于输出，seq2seq模型使我们从序列长度和顺序中解放出来，这使其成为两种语言之间转换
的理想选择。

考虑一句“Je ne suis pas le chat noir”→“我不是黑猫”。输入句子中的大多数单词在输出句子中都有直接翻译，但顺序略有不同，例
如 “聊天黑色”和“黑猫”。由于“ne / pas”结构，输入句中还有一个单词。直接从输入字序列产生正确的翻译将是困难的。

使用seq2seq模型，编码器创建单个向量，在理想情况下，将输入序列的“含义”编码为单个向量 - 句子的某些N维空间中的单个点。

#### 3.1 编码器
seq2seq网络的编码器是RNN，它为输入句子中的每个单词输出一些值。对于每个输入的词，编码器输出向量和隐藏状态，并将隐藏状态用于下
一个输入的单词。

![](https://pytorch.org/tutorials/_images/encoder-network.png)

```buildoutcfg
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

#### 3.2 解码器
码器是另一个RNN，它接收编码器输出向量并输出一系列字以创建转换。

简单的解码器
在最简单的seq2seq解码器中，我们仅使用编码器的最后一个输出。 最后一个输出有时称为上下文向量，因为它编码整个序列的上下文。 该上下文向量用作解码器的初始隐藏状态。

在解码的每个步骤中，给予解码器输入token和隐藏状态。初始输入token是开始字符串`<SOS>`标记，第一个隐藏状态是上下文向量（编码器
的最后隐藏状态）。

![](https://pytorch.org/tutorials/_images/decoder-network.png)

```buildoutcfg
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

我鼓励你训练和观察这个模型的结果，但为了节省空间，我们将直接进入主题并引入注意机制。

#### 3.3 注意力机制解码器
如果仅在编码器和解码器之间传递上下文向量，则该单个向量承担编码整个句子的信息。

注意力允许解码器网络针对解码器自身输出的每个步骤“聚焦”编码器输出的不同部分。首先，我们计算一组注意力权重。这些将乘以编码器
输出向量以创建加权组合。结果（在代码中称为`attn_applied`）应包含有关输入序列特定部分的信息，从而帮助解码器选择正确的输出单词。

使用解码器的输入和隐藏状态作为输入，使用另一个前馈层`attn`来计算注意力权重。因为训练数据中存在所有不同大小的句子，为了实际创建和
训练该层，我们必须选择它可以应用的最大句子长度（输入长度​​，对于编码器输出）。最大长度的句子将使用所有注意力权重，而较短的句子将
仅使用前几个。

![](https://pytorch.org/tutorials/_images/attention-decoder-network.png)

```buildoutcfg
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```
> 注意：<br/>
通过使用相对位置方法，还有其他形式的注意力可以解决长度限制问题。阅读[Effective Approaches to Attention-based Neural Machine Translation.]
(https://arxiv.org/abs/1508.04025)的“本地注意”。

## 4.训练
#### 4.1 准备训练数据
为了训练，对于每对翻译对，我们将需要输入张量（输入句子中的单词的索引）和目标张量（目标句子中的单词的索引）。在创建这些向量时，
我们会将EOS标记附加到两个序列。

```buildoutcfg
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
```

#### 4.2 训练模型
为了训练我们通过编码器运行的输入句子，并跟踪每个输出和最新的隐藏状态。然后，解码器被赋予<SOS>标记作为其第一输入，并且编码器的
最后隐藏状态作为其第一隐藏状态。

“Teacher Forcing”是将真实目标输出用作每个下一个输入的概念，而不是使用解码器的猜测作为下一个输入。使用teacher forcing使模型
更快地收敛，但是当利用受过训练的网络时，它可能表现出不稳定性。

您可以观察teacher forcing网络的输出，这些网络使用连贯的语法阅读，但远离正确的翻译 - 直觉上它已经学会表示输出语法，并且一旦
老师告诉它前几个单词就可以“提取”意义，但是它没有正确地学习如何从翻译中创建句子。

由于PyTorch的 autograd 为我们提供了自由，我们可以随意选择使用teacher forcing或不使用简单的if语句。将`teacher_forcing_ratio`
调高以使用更多。

```buildoutcfg
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
```

#### **辅助函数**
这是一个辅助函数，用于打印经过的时间和估计的剩余时间给定当前时间和进度％。

```buildoutcfg
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
```

整个训练过程如下：

* 启动计时器
* 初始化优化器和标准
* 创建一组训练对
* 启动空损数组进行绘图

然后我们调用`train`，偶尔打印进度（例子的百分比，到目前为止的时间，估计的时间）和平均损失。

```buildoutcfg
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
```

#### **结果绘图函数**
绘图使用 matplotlib 库完成，使用在训练时保存的`plot_losses`的损失值数组。

```buildoutcfg
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
```

#### **评价函数**
评估与训练大致相同，但没有目标，因此我们只需将解码器的预测反馈给每个步骤。每次它预测一个单词时我们都会将它添加到输出字符串中，
如果它预测了EOS标记，我们就会停在那里。我们还存储解码器的注意力输出以供稍后显示。

```buildoutcfg
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
```

我们可以从训练集中评估随机句子并打印输入、目标和输出以做出一些直观质量判断：

```buildoutcfg
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
```

## 5.训练和评价
有了所有这些辅助函数（它看起来像是额外的工作，但它使得运行多个实验更容易）我们实际上可以初始化网络并开始训练。

请记住，输入句子被严格过滤。对于这个小数据集，我们可以使用256个隐藏节点和单个GRU层的相对较小的网络。在MacBook CPU上大约40分
钟后，我们将得到一些合理的结果。

> 注意：<br/>
如果你运行这个笔记，你可以训练、中断内核、评估，并在以后继续训练。注释掉编码器和解码器初始化的行并再次运行trainIters。

```buildoutcfg
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
```

![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_002.png)

* 输出结果：
```buildoutcfg
1m 53s (- 26m 24s) (5000 6%) 2.8558
3m 42s (- 24m 3s) (10000 13%) 2.2832
5m 31s (- 22m 6s) (15000 20%) 1.9841
7m 19s (- 20m 8s) (20000 26%) 1.7271
9m 7s (- 18m 15s) (25000 33%) 1.5487
10m 54s (- 16m 21s) (30000 40%) 1.3461
12m 41s (- 14m 30s) (35000 46%) 1.2251
14m 30s (- 12m 41s) (40000 53%) 1.0956
16m 16s (- 10m 51s) (45000 60%) 1.0126
18m 5s (- 9m 2s) (50000 66%) 0.9212
19m 52s (- 7m 13s) (55000 73%) 0.7952
21m 41s (- 5m 25s) (60000 80%) 0.7481
23m 29s (- 3m 36s) (65000 86%) 0.6882
25m 17s (- 1m 48s) (70000 93%) 0.6190
27m 6s (- 0m 0s) (75000 100%) 0.5745
```

```buildoutcfg
evaluateRandomly(encoder1, attn_decoder1)
```

* 输出结果：
```buildoutcfg
> je pars en vacances pour quelques jours .
= i m taking a couple of days off .
< i m taking a couple of days off . <EOS>

> je ne me panique pas .
= i m not panicking .
< i m not panicking . <EOS>

> je recherche un assistant .
= i am looking for an assistant .
< i m looking a call . <EOS>

> je suis loin de chez moi .
= i m a long way from home .
< i m a little friend . <EOS>

> vous etes en retard .
= you re very late .
< you are late . <EOS>

> j ai soif .
= i am thirsty .
< i m thirsty . <EOS>

> je suis fou de vous .
= i m crazy about you .
< i m crazy about you . <EOS>

> vous etes vilain .
= you are naughty .
< you are naughty . <EOS>

> il est vieux et laid .
= he s old and ugly .
< he s old and ugly . <EOS>

> je suis terrifiee .
= i m terrified .
< i m touched . <EOS>
```

## 6.可视化注意力
注意力机制的一个有用特性是其高度可解释的输出。因为它用于对输入序列的特定编码器输出进行加权，所以我们可以想象在每个时间步长看
网络最关注的位置。

您可以简单地运行plt.matshow（attention）以将注意力输出显示为矩阵，其中列是输入步骤，行是输出步骤：

```
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())
```
![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_003.png)

为了获得更好的观看体验，我们将额外添加轴和标签：

```buildoutcfg
def showAttention(input_sentence, output_words, attentions):
    # 用colorbar设置图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # 设置坐标
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # 在每个刻度处显示标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")
```

![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_006.png)
![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_005.png)
![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_006.png)
![](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_007.png)

* 输出结果：
```
input = elle a cinq ans de moins que moi .
output = she is two years younger than me . <EOS>
input = elle est trop petit .
output = she s too trusting . <EOS>
input = je ne crains pas de mourir .
output = i m not afraid of dying . <EOS>
input = c est un jeune directeur plein de talent .
output = he s a fast person . <EOS>
```

## 7.练习
* 尝试使用其他数据集：<br/>
&emsp; * 另一种语言对<br/>
&emsp; * 人→机器（例如IOT命令）<br/>
&emsp; * 聊天→回复<br/>
&emsp; * 问题→答案<br/>
* 用预先训练过的字嵌入（例如word2vec或GloVe）替换嵌入<br/>
* 尝试使用更多图层，更多隐藏单元和更多句子。比较训练时间和结果。<br/>
* 如果你使用翻译文件，其中对有两个相同的短语（`I am test \t I am tes`），你可以使用它作为自动编码器。试试这个：<br/>
&emsp; * 训练为自动编码器<br/>
&emsp; * 仅保存编码器网络<br/>
&emsp; * 从那里训练一个新的解码器进行翻译<br/>