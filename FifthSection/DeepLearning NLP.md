# 在深度学习和 NLP 中使用 Pytorch
本文带您进入pytorch框架进行深度学习编程的核心思想。Pytorch的很多概念(比如计算图抽象和自动求导)并非它所独有的,和其他深度学习
框架相关。

我写这篇教程是专门针对那些从未用任何深度学习框架(例如：Tensorflow, Theano, Keras, Dynet)编写代码而从事NLP领域的人。我假设你
已经知道NLP领域要解决的核心问题：词性标注、语言模型等等。我也认为你通过AI这本书中所讲的知识熟悉了神经网络达到了入门的级别。
通常这些课程都会介绍反向传播算法和前馈神经网络，并指出它们是线性组合和非线性组合构成的链。本文在假设你已经有了这些知识的情况下，
教你如何开始写深度学习代码。

注意这篇文章主要关于`_models_`，而不是数据。对于所有的模型，我只创建一些数据维度较小的测试示例以便你可以看到权重在训练过程中
如何变化。如果你想要尝试一些真实数据，您有能力删除本示例中的模型并重新训练他们。

* PyTorch简介：
[PyTorch简介](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/PyTorch_Introuctiion.md)

* 使用PyTorch进行深度学习：
[使用PyTorch进行深度学习](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/DL_with_PyTorch.md)

* 词嵌入：编码形式的词汇语义：
[词嵌入：编码形式的词汇语义](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Word_Embedding.md)

* 序列模型和长短句记忆（LSTM）模型：
[序列模型和长短句记忆（LSTM）模型](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Sequence_and_LSTM_Network.md)

* 高级：制定动态决策和BI-LSTM CRF：
[制定动态决策和BI-LSTM CRF](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Dynamic_Desicion_Bi-LSTM.md)