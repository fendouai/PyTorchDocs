# 简介

目前研究人员正在使用的深度学习框架不尽相同，有 TensorFlow 、PyTorch、Keras等。这些深度学习框架被应用于计算机视觉、语音识别、自然语言处理与生物信息学等领域，并获取了极好的效果。其中，PyTorch是当前难得的简洁优雅且高效快速的框架，当前开源的框架中，没有哪一个框架能够在灵活性、易用性、速度这三个方面有两个能同时超过PyTorch。

本文档的定位是 PyTorch 入门教程，主要针对想要学习PyTorch的学生群体或者深度学习爱好者。通过教程的学习，能够实现零基础想要了解和学习深度学习，降低自学的难度，快速学习PyTorch。

官方教程包含了 PyTorch 介绍，安装教程；60分钟快速入门教程，可以迅速从小白阶段完成一个分类器模型；计算机视觉常用模型，方便基于自己的数据进行调整，不再需要从头开始写；自然语言处理模型，聊天机器人，文本生成等生动有趣的项目。

总而言之：
* 如果你想了解一下 PyTorch，可以看介绍部分。
* 如果你想快速入门 PyTorch，可以看60分钟快速入门。
* 如果你想解决计算机视觉问题，可以看计算机视觉部分。
* 如果你想解决自然语言处理问题，可以看NLP 部分。
* 还有强化学习和生成对抗网络部分内容。

作者：[磐创AI](http://www.panchuangai.com/) [PyTorch](http://pytorch123.com/) 翻译小组: News & PanChuang

原文：[https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

# 目录
## 第一章：PyTorch之简介与下载
### 1.[PyTorch简介](https://github.com/fendouai/PyTorchDocs/blob/master/FirstSection/PyTorchIntro.md)
### 2.[PyTorch环境搭建](https://github.com/fendouai/PyTorchDocs/blob/master/FirstSection/InstallIutorial.md)

## 第二章：PyTorch之60min入门
### 1.[PyTorch 入门](https://github.com/fendouai/PyTorchDocs/blob/master/SecondSection/what_is_pytorch.md)
### 2.[PyTorch 自动微分](https://github.com/fendouai/PyTorchDocs/blob/master/SecondSection/autograd_automatic_differentiation.md)
### 3.[PyTorch 神经网络](https://github.com/fendouai/PyTorchDocs/blob/master/SecondSection/neural_networks.md)
### 4.[PyTorch 图像分类器](https://github.com/fendouai/PyTorchDocs/blob/master/SecondSection/training_a_classifier.md)
### 5.[PyTorch 数据并行处理](https://github.com/fendouai/PyTorchDocs/blob/master/SecondSection/optional_data_parallelism.md)

## 第三章：PyTorch之入门强化
### 1.[数据加载和处理](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/DataLoding.md)
### 2.[PyTorch小试牛刀](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/LearningPyTorch.md)
### 3.[迁移学习](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/TransferLearning.md)
### 4.[混合前端的seq2seq模型部署](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/DeployingSeq2SeqModelwithHybridFrontend.MD)
### 5.[保存和加载模型](https://github.com/fendouai/PyTorchDocs/blob/master/ThirdSection/SaveModel.md)

## 第四章：PyTorch之图像篇
### 1.[微调基于torchvision 0.3的目标检测模型](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/ObjectDetectionFinetuning.md)
### 2.[微调TorchVision模型](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/FinetuningTorchVisionModel.md)
### 3.[空间变换器网络](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/SpatialTranNet.md)
### 4.[使用PyTorch进行Neural-Transfer](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/NeuralTransfer.md)
### 5.[生成对抗示例](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/AdversarialExampleGene.md)
### 6.[使用ONNX将模型转移至Caffe2和移动端](https://github.com/fendouai/PyTorchDocs/blob/master/fourSection/ONNX.md)

## 第五章：PyTorch之文本篇
### 1.[聊天机器人教程](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Chatbot.md)
### 2.[使用字符级RNN生成名字](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Char%20RNN%20Generation.MD)
### 3.[使用字符级RNN进行名字分类](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Char%20RNN%20Classification.md)
### 4.[在深度学习和NLP中使用Pytorch](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/DeepLearning%20NLP.md)
### 5.[使用Sequence2Sequence网络和注意力进行翻译](https://github.com/fendouai/PyTorchDocs/blob/master/FifthSection/Translation_S2S%20Network.md)

## 第六章：PyTorch之生成对抗网络

### 1.[生成对抗网络（Generative Adversarial Networks）](https://github.com/fendouai/PyTorchDocs/blob/master/SixthSection/Dcgan.md)

## 第七章：PyTorch之强化学习

### 1.[强化学习（DQN）](https://github.com/fendouai/PyTorchDocs/blob/master/SeventhSection/ReinforcementLearning.md)

# 教程推荐

* PyTorch 入门教程

[http://pytorchchina.com](http://pytorchchina.com)

* 磐创AI 聊天机器人，智能客服：

[http://www.panchuangai.com/](http://www.panchuangai.com/)

* 磐创教程网站，TensorFlow，Pytorch，Keras：

[http://panchuang.net/](http://panchuang.net/)

* 魔图互联 知识图谱推荐系统：

[http://motuhulian.com](http://motuhulian.com)

由于译者水平有限，如有疏漏，欢迎提交 PR。
