你已经了解了如何定义神经网络，计算损失值和网络里权重的更新。
<h3>现在你也许会想应该怎么处理数据？</h3>
通常来说，当你处理图像，文本，语音或者视频数据时，你可以使用标准 python 包将数据加载成 numpy 数组格式，然后将这个数组转换成 torch.*Tensor
<ul>
 	<li>对于图像，可以用 Pillow，OpenCV</li>
 	<li>对于语音，可以用 scipy，librosa</li>
 	<li>对于文本，可以直接用 Python 或 Cython 基础数据加载模块，或者用 NLTK 和 SpaCy</li>
</ul>
特别是对于视觉，我们已经创建了一个叫做 totchvision 的包，该包含有支持加载类似Imagenet，CIFAR10，MNIST 等公共数据集的数据加载模块 torchvision.datasets 和支持加载图像数据数据转换模块 torch.utils.data.DataLoader。

这提供了极大的便利，并且避免了编写“样板代码”。

对于本教程，我们将使用CIFAR10数据集，它包含十个类别：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。CIFAR-10 中的图像尺寸为3*32*32，也就是RGB的3层颜色通道，每层通道内的尺寸为32*32。

<img class="alignnone size-full wp-image-116" src="http://pytorchchina.com/wp-content/uploads/2018/12/cifar10.png" alt="" width="472" height="369" />
<h3>训练一个图像分类器</h3>
我们将按次序的做如下几步：
<ol>
 	<li>使用torchvision加载并且归一化CIFAR10的训练和测试数据集</li>
 	<li>定义一个卷积神经网络</li>
 	<li>定义一个损失函数</li>
 	<li>在训练样本数据上训练网络</li>
 	<li>在测试样本数据上测试网络</li>
</ol>
加载并归一化 CIFAR10
使用 torchvision ,用它来加载 CIFAR10 数据非常简单。

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

torchvision 数据集的输出是范围在[0,1]之间的 PILImage，我们将他们转换成归一化范围为[-1,1]之间的张量 Tensors。
```python

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

```

输出：
```python

Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Files already downloaded and verified
```

让我们来展示其中的一些训练图片。
```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

```
输出：
```python
cat plane  ship  frog

```
<div>定义一个卷积神经网络
在这之前先 从神经网络章节 复制神经网络，并修改它为3通道的图片(在此之前它被定义为1通道)</div>

```python

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```


<div>定义一个损失函数和优化器
让我们使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器。</div>

```python

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

<div>训练网络
这里事情开始变得有趣，我们只需要在数据迭代器上循环传给网络和优化器 输入就可以。</div>
<div></div>

```python

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```


输出：

```
[1,  2000] loss: 2.187
[1,  4000] loss: 1.852
[1,  6000] loss: 1.672
[1,  8000] loss: 1.566
[1, 10000] loss: 1.490
[1, 12000] loss: 1.461
[2,  2000] loss: 1.389
[2,  4000] loss: 1.364
[2,  6000] loss: 1.343
[2,  8000] loss: 1.318
[2, 10000] loss: 1.282
[2, 12000] loss: 1.286
Finished Training

```


在测试集上测试网络
我们已经通过训练数据集对网络进行了2次训练，但是我们需要检查网络是否已经学到了东西。

我们将用神经网络的输出作为预测的类标来检查网络的预测性能，用样本的真实类标来校对。如果预测是正确的，我们将样本添加到正确预测的列表里。

好的，第一步，让我们从测试集中显示一张图像来熟悉它。<img class="alignnone size-full wp-image-118" src="http://pytorchchina.com/wp-content/uploads/2018/12/sphx_glr_cifar10_tutorial_002.png" alt="" width="640" height="480" />

输出：

```python
GroundTruth:    cat  ship  ship plane
```

现在让我们看看 神经网络认为这些样本应该预测成什么：
```python
outputs = net(images)

```
输出是预测与十个类的近似程度，与某一个类的近似程度越高，网络就越认为图像是属于这一类别。所以让我们打印其中最相似类别类标：

```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

```
输出：
```python

Predicted:    cat  ship   car  ship

```
结果看起开非常好，让我们看看网络在整个数据集上的表现。
```python

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

输出：
```python
Accuracy of the network on the 10000 test images: 54 %


```
这看起来比随机预测要好，随机预测的准确率为10%（随机预测出为10类中的哪一类）。看来网络学到了东西。

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

```
输出：
<pre>Accuracy of plane : 57 %
Accuracy of   car : 73 %
Accuracy of  bird : 49 %
Accuracy of   cat : 54 %
Accuracy of  deer : 18 %
Accuracy of   dog : 20 %
Accuracy of  frog : 58 %
Accuracy of horse : 74 %
Accuracy of  ship : 70 %
Accuracy of truck : 66 %</pre>
所以接下来呢？

我们怎么在GPU上跑这些神经网络？

在GPU上训练
就像你怎么把一个张量转移到GPU上一样，你要将神经网络转到GPU上。
如果CUDA可以用，让我们首先定义下我们的设备为第一个可见的cuda设备。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)
```
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>cuda:0
</pre>
</div>
</div>
本节剩余部分都会假定设备就是台CUDA设备。

接着这些方法会递归地遍历所有模块，并将它们的参数和缓冲器转换为CUDA张量。
<div class="code python highlight-default notranslate">
<div class="highlight">
<pre><span class="n">net</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
</pre>
</div>
</div>
记住你也必须在每一个步骤向GPU发送输入和目标：
<div class="code python highlight-default notranslate">
<div class="highlight">
<pre><span class="n">inputs</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="n">inputs</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">),</span> <span class="n">labels</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
</pre>
</div>
</div>
为什么没有注意到与CPU相比巨大的加速？因为你的网络非常小。

<strong> </strong>

<strong>练习：</strong>尝试增加你的网络宽度（首个 nn.Conv2d 参数设定为 2，第二个nn.Conv2d参数设定为1--它们需要有相同的个数），看看会得到怎么的速度提升。

<strong>目标：</strong>
<ul>
 	<li>深度理解了PyTorch的张量和神经网络</li>
 	<li>训练了一个小的神经网络来分类图像</li>
</ul>
在多个GPU上训练

如果你想要来看到大规模加速，使用你的所有GPU，请查看：数据并行性（https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html）。PyTorch 60 分钟入门教程：数据并行处理

http://pytorchchina.com/2018/12/11/optional-data-parallelism/

下载 Python 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/cifar10_tutorial.py_.zip">cifar10_tutorial.py</a>

下载 Jupyter 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/cifar10_tutorial.ipynb_.zip">cifar10_tutorial.ipynb</a>
