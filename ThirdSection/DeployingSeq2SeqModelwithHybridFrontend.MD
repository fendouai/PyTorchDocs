# 混合前端的seq2seq模型部署

本教程将介绍如何是seq2seq模型转换为PyTorch可用的前端混合Torch脚本。我们要转换的模型来自于聊天机器人教程[Chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)。
## 1.混合前端
在一个基于深度学习项目的研发阶段, 使用像PyTorch这样**即时**`eager`、命令式的界面进行交互能带来很大便利。 这使用户能够在使用Python数据结构、控制流操作、打印语句和调试实用程序时通过熟悉的、惯用的Python脚本编写。

尽管即时性界面对于研究和试验应用程序是一个有用的工具，但是对于生产环境中部署模型时，使用基于图形`graph-based`的模型表示将更加适用的。 一个延迟的图型展示意味着可以优化，比如无序执行操作，以及针对高度优化的硬件架构的能力。 此外，基于图形的表示支持框架无关的模型导出。PyTorch提供了将即时模式的代码增量转换为Torch脚本的机制，Torch脚本是一个在Python中的静态可分析和可优化的子集，Torch使用它来在Python运行时独立进行深度学习。

在Torch中的`torch.jit`模块可以找到将即时模式的PyTorch程序转换为Torch脚本的API。 这个模块有两个核心模式用于将即时模式模型转换为Torch脚本图形表示: **跟踪**`tracing` 以及 **脚本化**`scripting`。`torch.jit.trace` 函数接受一个模块或者一个函数和一组示例的输入，然后通过函数或模块运行输入示例，同时跟跟踪遇到的计算步骤，然后输出一个可以展示跟踪流程的基于图形的函数。**跟踪**`Tracing`对于不涉及依赖于数据的控制流的直接的模块和函数非常有用，就比如标准的卷积神经网络。

然而，如果一个有数据依赖的if语句和循环的函数被跟踪，则只记录示例输入沿执行路径调用的操作。换句话说，控制流本身并没有被捕获。要将带有数据依赖控制流的模块和函数进行转化，已提供了一个脚本化机制。脚本显式地将模块或函数代码转换为Torch脚本，包括所有可能的控制流路径。 如需使用脚本模式`script mode`， 要确定继承了 `torch.jit.ScriptModule`基本类 (取代`torch.nn.Module`) 并且增加 `torch.jit.script` 装饰器到你的Python函数或者 `torch.jit.script_method` 装饰器到你的模块方法。

使用脚本化的一个警告是，它只支持Python的一个受限子集。要获取与支持的特性相关的所有详细信息，请参考 Torch Script [language reference](https://pytorch.org/docs/master/jit.html)。为了达到最大的灵活性，可以组合Torch脚本的模式来表示整个程序，并且可以增量地应用这些技术。
![混合前端的流程图](https://upload-images.jianshu.io/upload_images/17387001-4ce74fdcee69415c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 2.预备环境

首先，导入所需的模块以及设置一些常量。如果想使用自己的模型，需要保证`MAX_LENGTH`常量设置正确。
提醒：这个常量定义了在训练过程中允许的最大句子长度以及模型能够产生的最大句子长度输出。

```
source-python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")

MAX_LENGTH = 10 # Maximum sentence length

# 默认的词向量
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token
```

### 3.模型概述
正如前文所言，我们使用的[sequence-to-sequence](https://arxiv.org/abs/1409.3215) (seq2seq) 模型。这种类型的模型用于输入是可变长度序列的情况，我们的输出也是一个可变长度序列它不一定是一对一输入映射。`seq2seq` 模型由两个递归神经网络(RNNs)组成：编码器 **encoder**和解码器**decoder**.
![模型概述](https://upload-images.jianshu.io/upload_images/17387001-84f6c1fb31ef2261.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### （1）编码器(Encoder)
编码器RNN在输入语句中每次迭代一个标记(例如单词)，每次步骤输出一个“输出”向量和一个“隐藏状态”向量。”隐藏状态“向量在之后则传递到下一个步骤，同时记录输出向量。编码器将序列中每个坐标代表的文本转换为高维空间中的一组坐标，解码器将使用这些坐标为给定的任务生成有意义的输出。

#### （2）解码器(Decoder)

解码器RNN以逐个令牌的方式生成响应语句。它使用来自于编码器的文本向量和内部隐藏状态来生成序列中的下一个单词。它继续生成单词，直到输出表示句子结束的EOS语句。我们在解码器中使用专注机制[attention mechanism](https://arxiv.org/abs/1409.0473)来帮助它在输入的某些部分生成输出时"保持专注"。对于我们的模型，我们实现了 [Luong et al](https://arxiv.org/abs/1508.04025)等人的“全局关注`Global attention`”模块，并将其作为解码模型中的子模块。

### 4.数据处理

尽管我们的模型在概念上处理标记序列，但在现实中，它们与所有机器学习模型一样处理数字。在这种情况下，在训练之前建立的模型词汇表中的每个单词都映射到一个整数索引。我们使用`Voc`对象来包含从单词到索引的映射，以及词汇表中的单词总数。我们将在运行模型之前加载对象。

此外，为了能够进行评估，我们必须提供一个处理字符串输入的工具。`normalizeString`函数将字符串中的所有字符转换为小写，并删除所有非字母字符。`indexesFromSentence`函数接受一个单词的句子并返回相应的单词索引序列。

```source-python
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计默认的令牌
        for word in keep_words:
            self.addWord(word)

# 小写并删除非字母字符
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 使用字符串句子，返回单词索引的句子
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
```
### 5.定义编码器
通过`torch.nn.GRU`模块实现编码器的RNN。本模块接受一批语句(嵌入单词的向量)的输入，它在内部遍历这些句子，每次一个标记，
计算隐藏状态。我们将这个模块初始化为双向的，这意味着我们有两个独立的GRUs:一个按时间顺序遍历序列，另一个按相反顺序遍历序列。
我们最终返回这两个GRUs输出的和。由于我们的模型是使用批处理进行训练的，所以我们的EncoderRNN模型的`forward`函数需要一个填充
的输入批处理。为了批量处理可变长度的句子，我们通过`MAX_LENGTH`令牌允许一个句子中支持的最大长度，并且批处理中所有小于`MAX_LENGTH`
令牌的句子都使用我们专用的`PAD_token`令牌填充在最后。要使用带有PyTorch RNN模块的批量填充，我们必须把转发`forward`密令在
调用`torch.nn.utils.rnn.pack_padded_sequence`和`torch.nn.utils.rnn.pad_packed_sequence`数据转换时进行打包。注意，`forward`
函数还接受一个`input_length`列表，其中包含批处理中每个句子的长度。该输入在填充时通过`torch.nn.utils.rnn.pack_padded_sequence`使用。

* 混合前端笔记
由于编码器的转发函数`forward`不包含任何依赖于数据的控制流，因此我们将使用跟踪`tracing`将其转换为脚本模式`script mode`。在跟踪模块时，
我们可以保持模块定义不变。在运行评估之前，我们将在本文末尾初始化所有模型。
```buildoutcfg
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU;input_size和hidden_size参数都设置为'hidden_size'
        # 因为我们输入的大小是一个有多个特征的词向量== hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 将单词索引转换为向量
        embedded = self.embedding(input_seq)
        # 为RNN模块填充批次序列
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 正向通过GRU
        outputs, hidden = self.gru(packed, hidden)
        # 打开填充
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 将双向GRU的输出结果总和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # 返回输出以及最终的隐藏状态
        return outputs, hidden
```
### 6.定义解码器的注意力模块
接下来，将定义注意力模块(Attn)。请注意，此模块将用作解码器模型中的子模块。Luong等人考虑了各种“分数函数”`score functions`，
它们取当前解码器RNN输出和整个编码器输出，并返回关注点“能值”`engergies`。这个关注能值张量`attension energies tensor`与
编码器输出的大小相同，两者最终相乘，得到一个加权张量，其最大值表示在特定时间步长解码的查询语句最重要的部分。
```buildoutcfg
# Luong的注意力层
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力权重（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 转置max_length和batch_size维度
        attn_energies = attn_energies.t()

        # 返回softmax归一化概率分数（增加维度）
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
```
### 7.定义解码器
类似于`EncoderRNN`，我们使用`torch.nn.GRU`模块作为我们的解码器RNN。然而，这一次我们使用单向GRU。需要注意的是，与编码器不同，我们将向解码器RNN每次提供一个单词。我们首先得到当前单词的嵌入并应用抛出功能`dropout`。接下来，我们将嵌入和最后的隐藏
状态转发给GRU，得到当前的GRU输出和隐藏状态。然后，我们使用Attn模块作为一个层来获得专注权重，我们将其乘以编码器的输出来获得我们的参与编码器输出。我们使用这个参与编码器输出作为文本`context`张量，它表示一个加权和，表示编码器输出的哪些部分需要注意。
在这里，我们使用线性层`linear layer`和`softmax normalization` 归一化来选择输出序列中的下一个单词。

* 混合前端笔记
与`EncoderRNN`类似，此模块不包含任何依赖于数据的控制流。因此，在初始化该模型并加载其参数之后，我们可以再次使用跟踪`tracing`将其转换为Torch脚本。
```buildoutcfg
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 保持参考
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义层
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 注意：我们这步只运行一次
        # 获取当前输入字对应的向量映射
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 通过单向GRU转发
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 通过当前GRU的输出计算注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 使用Luong的公式5来连接加权上下文向量和GRU输出
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # 使用Luong的公式6来预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # 返回输出和最终的隐藏状态
        return output, hidden
```
### 8.定义评估
#### 8.1 贪婪搜索解码器
在聊天机器人教程中，我们使用`GreedySearchDecoder`模块来简化实际的解码过程。该模块将训练好的编码器和解码器模型作为属性，
驱动输入语句(词索引向量)的编码过程，并一次一个词(词索引)迭代地解码输出响应序列。

对输入序列进行编码很简单:只需将整个序列张量及其对应的长度向量转发给编码器。需要注意的是，这个模块一次只处理一个输入序列，
而不是成批的序列。因此，当常数1用于声明张量大小时，它对应于批处理大小为1。要解码给定的解码器输出，我们必须通过解码器模型
迭代地向前运行，该解码器模型输出`softmax`分数，该分数对应于每个单词在解码序列中是正确的下一个单词的概率。我们将`decoder_input`
初始化为一个包含`SOS_token`的张量。在每次通过解码器之后，我们贪婪地将`softmax`概率最高的单词追加到`decoded_words`列表中。
我们还使用这个单词作为下一个迭代的`decoder_input`。如果`decoded_words`列表的长度达到`MAX_LENGTH`，或者预测的单词是`EOS_token`，
那么解码过程将终止。

* 混合前端笔记

该模块的`forward`方法涉及到在每次解码一个单词的输出序列时，遍历/([0,max/_length]/)的范围。因此，我们应该使用脚本将这个模
块转换为Torch脚本。与我们可以跟踪的编码器和解码器模型不同，我们必须对`GreedySearchDecoder`模块进行一些必要的更改，以便在
不出错的情况下初始化对象。换句话说，我们必须确保我们的模块遵守脚本机制的规则，并且不使用Torch脚本包含的Python子集之外的任何语言特性。

为了了解可能需要的一些操作，我们将回顾聊天机器人教程中的GreedySearchDecoder实现与下面单元中使用的实现之间的区别。请注意，
用红色突出显示的行是从原始实现中删除的行，而用绿色突出显示的行是新的。
![](./image/5.png)

##### 变更事项：
* `nn.Module` -> `torch.jit.ScriptModule`
    为了在模块上使用PyTorch的脚本化机制, 模型需要从`torch.jit.ScriptModule`继承。
* 将`decoder_n_layers`追加到结构参数
    这种变化源于这样一个事实，即我们传递给这个模块的编码器和解码器模型将是`TracedModule`(非模块)的子模块。因此，我们无法使用
`decoder.n_layers`访问解码器的层数。相反，我们对此进行计划，并在模块构建过程中传入此值。
* 将新属性作为常量保存
    在最初的实现中，我们可以在`GreedySearchDecoder`的`forward`方法中自由地使用来自周围(全局)范围的变量. 然而，现在我们正
    在使用脚本，我们没有这种自由，因为脚本处理的设想4是我们不一定要保留Python对象，尤其是在导出时。 一个简单的解决方案是
    将全局作用域中的这些值作为属性存储到构造函数中的模块中， 并将它们添加到一个名为`__constants__`的特殊列表中，以便在`forward`
    方法中构造图形时将它们用作文本值。这种用法的一个例子在第19行，取代使用`device`和`SOS_token`全局值，我们使用常量属性
    `self._device`和`self._SOS_token`。
* 将`torch.jit.script_method`装饰器添加到`forward`方法
    添加这个装饰器可以让JIT编译器知道它所装饰的函数应该是脚本化的。
* 强制`forward`方法的参数类型
    默认情况下，Torch脚本函数的所有参数都假定为张量。如果需要传递不同类型的参数，可以使用PEP 3107中引入的函数类型注释。 
    此外，还可以使用MyPy-style类型的注释声明不同类型的参数。
* 变更`decoder_input`的初始化
    在原有实现中，我们用`torch.LongTensor([[SOS_token]])`初始化了`decoder_input`的张量。当脚本编写时,我们不允许像这样以
    一种文字方式初始化张量。取而代之的是，我们可以用一个显式的torch函数，比如`torch.ones`来初始化我们的张量。这种情况下，
    我们可以很方便的复制标量`decoder_input`和通过将1乘以我们存在常量中的`SOS_token`的值`self._SOS_token`得到的张量。
 ```buildoutcfg
class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # 通过编码器模型转发输入
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 准备编码器的最终隐藏层作为解码器的第一个隐藏输入
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # 使用SOS_token初始化解码器输入
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # 初始化张量以将解码后的单词附加到
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # 一次迭代地解码一个词令牌
        for _ in range(max_length):
            # 正向通过解码器
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 获得最可能的单词标记及其softmax分数
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 记录令牌和分数
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 准备当前令牌作为下一个解码器输入（添加维度）
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回词令牌和分数的集合
        return all_tokens, all_scores
```
#### 8.2 输入评估
接下来，我们定义一些函数来计算输入。求值函数`evaluate`接受一个规范化字符串语句，将其处理为其对应的单词索引张量(批处理大小
为1)，并将该张量传递给一个名为`searcher`的`GreedySearchDecoder`实例，以处理编码/解码过程。检索器返回输出的单词索引向量和
一个分数张量，该张量对应于每个解码的单词标记的`softmax`分数。最后一步是使用`voc.index2word`将每个单词索引转换回其字符串表示形式。

我们还定义了两个函数来计算输入语句。`evaluateInput`函数提示用户输入，并计算输入。它持续请求另一次输入，直到用户输入“q”或“quit”。

`evaluateExample`函数只接受一个字符串输入语句作为参数，对其进行规范化、计算并输出响应。
```buildoutcfg
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 格式化输入句子作为批处理
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置批量的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用适当的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用earcher解码句子s
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# 评估来自用户输入的输入(stdin)
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 获取输入的句子
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# 规范化输入句子并调用evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # 规范化句子
    input_sentence = normalizeString(sentence)
    # 评估句子
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))
```

### 9.加载预训练参数
#### 9.1 使用托管模型
托管模型使用步骤:

1.下载模型[这里](https://download.pytorch.org/models/tutorials/4000_checkpoint.tar).
2.设置`loadFilename`变量作为下载的检查点文件的路径
3.将`checkpoint = torch.load(loadFilename)`行取消注释，表示托管模型在CPU上训练。

#### 9.2 使用自己的模型
加载自己的预训练模型设计步骤:

1.将`loadFilename`变量设置为希望加载的检查点文件的路径。注意，如果您遵循从chatbot tutorial中保存模型的协议，这会涉及更改
`model_name`、`encoder_n_layers`、`decoder_n_layers`、`hidden_size`和`checkpoint_iter`(因为这些值在模型路径中使用到)。
2.如果你在CPU上训练，确保你在`checkpoint = torch.load(loadFilename)`行打开了检查点。如果你在GPU上训练，并且在CPU运行这篇
教程，解除`checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))`的注释。

* 混合前端笔记

请注意，我们像往常一样初始化并将参数加载到编码器和解码器模型中。另外，在跟踪模型之前，我们必须调用`.to(device)`来设置模型
的设备选项，调用`.eval()`来设置抛出层`dropout layer`为test mode。`TracedModule`对象不继承`to`或`eval`方法。
```buildoutcfg
save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"

# 配置模型
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 如果你加载的是自己的模型
# 设置要加载的检查点
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

# 如果你加载的是托管模型
loadFilename = 'data/4000_checkpoint.tar'

# 加载模型
# 强制CPU设备选项（与本教程中的张量匹配）
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# 初始化词向量
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# 初始化编码器和解码器模型
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# 加载训练模型参数
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# 使用适当的设备
encoder = encoder.to(device)
decoder = decoder.to(device)
# 将dropout层设置为eval模式
encoder.eval()
decoder.eval()
print('Models built and ready to go!')
```
* 输出
```buildoutcfg
Building encoder and decoder ...
Models built and ready to go!
```

### 10.将模型转换为 Torch 脚本
#### 10.1 编码器
正如前文所述，要将编码器模型转换为Torch脚本，我们需要使用跟踪`Tracing`。跟踪任何需要通过模型的`forward`方法运行一个示例
输入，以及跟踪数据相遇时的图形计算。编码器模型接收一个输入序列和一个长度相关的张量。因此，我们创建一个输入序列`test_seq`，
配置合适的大小`(MAX_LENGTH,1)`包含适当范围内的数值\([0,voc.num\_words]\)以及搭配的类型(int64)。我们还创建了`test_seq_length`
标量，该标量实际包含与test_seq中单词数量对应的值。下一步是使用`torch.jit.trace`函数来跟踪模型。注意，我们传递的第一个参数
是要跟踪的模块，第二个参数是模块forward方法的参数元组。
#### 10.2 解码器
我们对解码器的跟踪过程与对编码器的跟踪过程相同。请注意，我们对`traced_encoder`的一组随机输入调用`forward`，以获得解码器所
需的输出。这不是必需的，因为我们也可以简单地生成一个形状、类型和值范围正确的张量。这种方法是可行的，因为在我们的例子中，
我们对张量的值没有任何约束，因为我们没有任何操作可能导致超出范围的输入出错。
#### 10.3 贪婪搜索解码器
回想一下，由于存在依赖于数据的控制流，我们为搜索器模块编写了脚本。在脚本化的情况下，我们通过添加修饰符并确保实现符合脚本
规则来预先完成转换工作。我们初始化脚本搜索器的方式与初始化未脚本化变量的方式相同。
```buildoutcfg
### 转换编码器模型
# 创建人工输入
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)
test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
# 跟踪模型
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

### 转换解码器模型
# 创建并生成人工输入
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# 跟踪模型
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

### 初始化searcher模块
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)
```

### 11.图形打印
现在我们的模型是Torch脚本形式的，我们可以打印每个模型的图形，以确保适当地捕获计算图形。因为`scripted_searcher`包含
`traced_encoder`和`traced_decoder`，所以这些图将以内联方式打印。

* 输出
```buildoutcfg
scripted_searcher graph:
 graph(%input_seq : Tensor,
      %input_length : Tensor,
      %max_length : int,
      %126 : Tensor,
      %127 : Tensor,
      %128 : Tensor,
      %129 : Tensor,
      %130 : Tensor,
      %131 : Tensor,
      %132 : Tensor,
      %133 : Tensor,
      %134 : Tensor,
      %135 : Tensor,
      %136 : Tensor,
      %137 : Tensor,
      %138 : Tensor,
      %139 : Tensor,
      %140 : Tensor,
      %141 : Tensor,
      %142 : Tensor,
      %143 : Tensor,
      %144 : Tensor,
      %145 : Tensor,
      %146 : Tensor,
      %147 : Tensor,
      %148 : Tensor,
      %149 : Tensor,
      %150 : Tensor,
      %151 : Tensor,
      %152 : Tensor,
      %153 : Tensor,
      %154 : Tensor,
      %155 : Tensor):
  %4 : bool? = prim::Constant()
  %5 : int? = prim::Constant()
  %6 : int = prim::Constant[value=9223372036854775807](), scope: EncoderRNN
  %7 : float = prim::Constant[value=0](), scope: EncoderRNN
  %8 : float = prim::Constant[value=0.1](), scope: EncoderRNN/GRU[gru]
  %9 : int = prim::Constant[value=2](), scope: EncoderRNN/GRU[gru]
  %10 : bool = prim::Constant[value=1](), scope: EncoderRNN/GRU[gru]
  %11 : int = prim::Constant[value=6](), scope: EncoderRNN/GRU[gru]
  %12 : int = prim::Constant[value=500](), scope: EncoderRNN/GRU[gru]
  %13 : int = prim::Constant[value=4](), scope: EncoderRNN
  %14 : Device = prim::Constant[value="cpu"](), scope: EncoderRNN
  %15 : bool = prim::Constant[value=0](), scope: EncoderRNN/Embedding[embedding]
  %16 : int = prim::Constant[value=-1](), scope: EncoderRNN/Embedding[embedding]
  %17 : int = prim::Constant[value=0]()
  %18 : int = prim::Constant[value=1]()
  %input.7 : Float(10, 1, 500) = aten::embedding(%155, %input_seq, %16, %15, %15), scope: EncoderRNN/Embedding[embedding]
  %lengths : Long(1) = aten::to(%input_length, %14, %13, %15, %15), scope: EncoderRNN
  %input.1 : Float(10, 500), %batch_sizes : Long(10) = aten::_pack_padded_sequence(%input.7, %lengths, %15), scope: EncoderRNN
  %43 : int[] = prim::ListConstruct(%13, %18, %12), scope: EncoderRNN/GRU[gru]
  %hx : Float(4, 1, 500) = aten::zeros(%43, %11, %17, %14, %15), scope: EncoderRNN/GRU[gru]
  %45 : Tensor[] = prim::ListConstruct(%154, %153, %152, %151, %150, %149, %148, %147, %146, %145, %144, %143, %142, %141, %140, %139), scope: EncoderRNN/GRU[gru]
  %46 : Float(10, 1000), %encoder_hidden : Float(4, 1, 500) = aten::gru(%input.1, %batch_sizes, %hx, %45, %10, %9, %8, %15, %10), scope: EncoderRNN/GRU[gru]
  %48 : int = aten::size(%batch_sizes, %17), scope: EncoderRNN
  %max_seq_length : Long() = prim::NumToTensor(%48), scope: EncoderRNN
  %50 : int = prim::Int(%max_seq_length), scope: EncoderRNN
  %outputs : Float(10, 1, 1000), %52 : Long(1) = aten::_pad_packed_sequence(%46, %batch_sizes, %15, %7, %50), scope: EncoderRNN
  %53 : Float(10, 1, 1000) = aten::slice(%outputs, %17, %17, %6, %18), scope: EncoderRNN
  %54 : Float(10, 1, 1000) = aten::slice(%53, %18, %17, %6, %18), scope: EncoderRNN
  %55 : Float(10, 1!, 500) = aten::slice(%54, %9, %17, %12, %18), scope: EncoderRNN
  %56 : Float(10, 1, 1000) = aten::slice(%outputs, %17, %17, %6, %18), scope: EncoderRNN
  %57 : Float(10, 1, 1000) = aten::slice(%56, %18, %17, %6, %18), scope: EncoderRNN
  %58 : Float(10, 1!, 500) = aten::slice(%57, %9, %12, %6, %18), scope: EncoderRNN
  %encoder_outputs : Float(10, 1, 500) = aten::add(%55, %58, %18), scope: EncoderRNN
  %decoder_hidden.1 : Tensor = aten::slice(%encoder_hidden, %17, %17, %9, %18)
  %61 : int[] = prim::ListConstruct(%18, %18)
  %62 : Tensor = aten::ones(%61, %13, %5, %14, %4)
  %decoder_input.1 : Tensor = aten::mul(%62, %18)
  %64 : int[] = prim::ListConstruct(%17)
  %all_tokens.1 : Tensor = aten::zeros(%64, %13, %5, %14, %4)
  %66 : int[] = prim::ListConstruct(%17)
  %all_scores.1 : Tensor = aten::zeros(%66, %5, %5, %14, %4)
  %all_scores : Tensor, %all_tokens : Tensor, %decoder_hidden : Tensor, %decoder_input : Tensor = prim::Loop(%max_length, %10, %all_scores.1, %all_tokens.1, %decoder_hidden.1, %decoder_input.1)
    block0(%72 : int, %73 : Tensor, %74 : Tensor, %75 : Tensor, %76 : Tensor):
      %input.2 : Float(1, 1, 500) = aten::embedding(%138, %76, %16, %15, %15), scope: LuongAttnDecoderRNN/Embedding[embedding]
      %input.3 : Float(1, 1, 500) = aten::dropout(%input.2, %8, %15), scope: LuongAttnDecoderRNN/Dropout[embedding_dropout]
      %97 : Tensor[] = prim::ListConstruct(%137, %136, %135, %134, %133, %132, %131, %130), scope: LuongAttnDecoderRNN/GRU[gru]
      %hidden : Float(1, 1, 500), %decoder_hidden.2 : Float(2, 1, 500) = aten::gru(%input.3, %75, %97, %10, %9, %8, %15, %15, %15), scope: LuongAttnDecoderRNN/GRU[gru]
      %100 : Float(10, 1, 500) = aten::mul(%hidden, %encoder_outputs), scope: LuongAttnDecoderRNN/Attn[attn]
      %101 : int[] = prim::ListConstruct(%9), scope: LuongAttnDecoderRNN/Attn[attn]
      %attn_energies : Float(10, 1) = aten::sum(%100, %101, %15), scope: LuongAttnDecoderRNN/Attn[attn]
      %input.4 : Float(1!, 10) = aten::t(%attn_energies), scope: LuongAttnDecoderRNN/Attn[attn]
      %104 : Float(1, 10) = aten::softmax(%input.4, %18), scope: LuongAttnDecoderRNN/Attn[attn]
      %attn_weights : Float(1, 1, 10) = aten::unsqueeze(%104, %18), scope: LuongAttnDecoderRNN/Attn[attn]
      %106 : Float(1!, 10, 500) = aten::transpose(%encoder_outputs, %17, %18), scope: LuongAttnDecoderRNN
      %context.1 : Float(1, 1, 500) = aten::bmm(%attn_weights, %106), scope: LuongAttnDecoderRNN
      %rnn_output : Float(1, 500) = aten::squeeze(%hidden, %17), scope: LuongAttnDecoderRNN
      %context : Float(1, 500) = aten::squeeze(%context.1, %18), scope: LuongAttnDecoderRNN
      %110 : Tensor[] = prim::ListConstruct(%rnn_output, %context), scope: LuongAttnDecoderRNN
      %input.5 : Float(1, 1000) = aten::cat(%110, %18), scope: LuongAttnDecoderRNN
      %112 : Float(1000!, 500!) = aten::t(%129), scope: LuongAttnDecoderRNN/Linear[concat]
      %113 : Float(1, 500) = aten::addmm(%128, %input.5, %112, %18, %18), scope: LuongAttnDecoderRNN/Linear[concat]
      %input.6 : Float(1, 500) = aten::tanh(%113), scope: LuongAttnDecoderRNN
      %115 : Float(500!, 7826!) = aten::t(%127), scope: LuongAttnDecoderRNN/Linear[out]
      %input : Float(1, 7826) = aten::addmm(%126, %input.6, %115, %18, %18), scope: LuongAttnDecoderRNN/Linear[out]
      %decoder_output : Float(1, 7826) = aten::softmax(%input, %18), scope: LuongAttnDecoderRNN
      %decoder_scores : Tensor, %decoder_input.2 : Tensor = aten::max(%decoder_output, %18, %15)
      %120 : Tensor[] = prim::ListConstruct(%74, %decoder_input.2)
      %all_tokens.2 : Tensor = aten::cat(%120, %17)
      %122 : Tensor[] = prim::ListConstruct(%73, %decoder_scores)
      %all_scores.2 : Tensor = aten::cat(%122, %17)
      %decoder_input.3 : Tensor = aten::unsqueeze(%decoder_input.2, %17)
      -> (%10, %all_scores.2, %all_tokens.2, %decoder_hidden.2, %decoder_input.3)
  %125 : (Tensor, Tensor) = prim::TupleConstruct(%all_tokens, %all_scores)
  return (%125)
```

### 12.运行结果评估
最后，我们将使用Torch脚本模型对聊天机器人模型进行评估。如果转换正确，模型的行为将与它们在即时模式表示中的行为完全相同。

默认情况下，我们计算一些常见的查询语句。如果您想自己与机器人聊天，取消对`evaluateInput`行的注释并让它旋转。
```buildoutcfg
# 评估例子
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)

# 评估你的输入
#evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)
```
* 输出
```buildoutcfg
> hello
Bot: hello .
> what's up?
Bot: i m going to get my car .
> who are you?
Bot: i m the owner .
> where am I?
Bot: in the house .
> where are you from?
Bot: south america .
```

### 13.保存模型
现在我们已经成功地将模型转换为Torch脚本，接下来将对其进行序列化，以便在非python部署环境中使用。为此，我们只需保存`scripted_searcher`
模块，因为这是用于对聊天机器人模型运行推理的面向用户的接口。保存脚本模块时，使用`script_module.save(PATH)`代替`torch.save(model, PATH)`。
