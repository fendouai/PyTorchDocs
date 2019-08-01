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

* https://pytorch.org/ PyTorch安装指南
* [Deep Learning with PyTorch：A 60 Minute Blitz ](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/beginner/deep_learning_60min_blitz.html):PyTorch的基本入门教程
* [Learning PyTorch with Examples](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/beginner/pytorch_with_examples.html):得到深层而广泛的概述
* [PyTorch for Former Torch Users Lua Torch](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/beginner/former_torchies_tutorial.html):如果你曾是一个Lua张量的使用者

事先学习并了解序列到序列网络的工作原理对理解这个例子十分有帮助:

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)

您还可以找到之前有关[Classifying Names with a Character-Level RNN]()和 
[Generating Names with a Character-Level RNN]()的教程，因为这些概念分别与编码器和解码器模型非常相似。

