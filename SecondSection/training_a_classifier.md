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
<pre><span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torchvision</span>
<span class="kn">import</span> <span class="nn">torchvision.transforms</span> <span class="kn">as</span> <span class="nn">transforms</span></pre>
torchvision 数据集的输出是范围在[0,1]之间的 PILImage，我们将他们转换成归一化范围为[-1,1]之间的张量 Tensors。
<pre><span class="n">transform</span> <span class="o">=</span> <span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">(</span>
    <span class="p">[</span><span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
     <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">((</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">),</span> <span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">))])</span>

<span class="n">trainset</span> <span class="o">=</span> <span class="n">torchvision</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">CIFAR10</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">'./data'</span><span class="p">,</span> <span class="n">train</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span>
                                        <span class="n">download</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">transform</span><span class="o">=</span><span class="n">transform</span><span class="p">)</span>
<span class="n">trainloader</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">trainset</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                                          <span class="n">shuffle</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">num_workers</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

<span class="n">testset</span> <span class="o">=</span> <span class="n">torchvision</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">CIFAR10</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">'./data'</span><span class="p">,</span> <span class="n">train</span><span class="o">=</span><span class="bp">False</span><span class="p">,</span>
                                       <span class="n">download</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">transform</span><span class="o">=</span><span class="n">transform</span><span class="p">)</span>
<span class="n">testloader</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">testset</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                                         <span class="n">shuffle</span><span class="o">=</span><span class="bp">False</span><span class="p">,</span> <span class="n">num_workers</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

<span class="n">classes</span> <span class="o">=</span> <span class="p">(</span><span class="s1">'plane'</span><span class="p">,</span> <span class="s1">'car'</span><span class="p">,</span> <span class="s1">'bird'</span><span class="p">,</span> <span class="s1">'cat'</span><span class="p">,</span>
           <span class="s1">'deer'</span><span class="p">,</span> <span class="s1">'dog'</span><span class="p">,</span> <span class="s1">'frog'</span><span class="p">,</span> <span class="s1">'horse'</span><span class="p">,</span> <span class="s1">'ship'</span><span class="p">,</span> <span class="s1">'truck'</span><span class="p">)</span></pre>
输出：
<pre>Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Files already downloaded and verified</pre>
让我们来展示其中的一些训练图片。
<pre><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="kn">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>

<span class="c1"># functions to show an image</span>


<span class="k">def</span> <span class="nf">imshow</span><span class="p">(</span><span class="n">img</span><span class="p">):</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">img</span> <span class="o">/</span> <span class="mi">2</span> <span class="o">+</span> <span class="mf">0.5</span>     <span class="c1"># unnormalize</span>
    <span class="n">npimg</span> <span class="o">=</span> <span class="n">img</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">npimg</span><span class="p">,</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">)))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>


<span class="c1"># get some random training images</span>
<span class="n">dataiter</span> <span class="o">=</span> <span class="nb">iter</span><span class="p">(</span><span class="n">trainloader</span><span class="p">)</span>
<span class="n">images</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="n">dataiter</span><span class="o">.</span><span class="n">next</span><span class="p">()</span>

<span class="c1"># show images</span>
<span class="n">imshow</span><span class="p">(</span><span class="n">torchvision</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">make_grid</span><span class="p">(</span><span class="n">images</span><span class="p">))</span>
<span class="c1"># print labels</span>
<span class="k">print</span><span class="p">(</span><span class="s1">' '</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="s1">'</span><span class="si">%5s</span><span class="s1">'</span> <span class="o">%</span> <span class="n">classes</span><span class="p">[</span><span class="n">labels</span><span class="p">[</span><span class="n">j</span><span class="p">]]</span> <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">4</span><span class="p">)))</span></pre>
&nbsp;

<img class="alignnone size-full wp-image-117" src="http://pytorchchina.com/wp-content/uploads/2018/12/sphx_glr_cifar10_tutorial_001.png" alt="" width="640" height="480" />

输出：
<div id="loading-and-normalizing-cifar10" class="section">
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>cat plane  ship  frog
</pre>
</div>
</div>
</div>
<div id="define-a-convolutional-neural-network" class="section"></div>
<div></div>
<div>定义一个卷积神经网络
在这之前先 从神经网络章节 复制神经网络，并修改它为3通道的图片(在此之前它被定义为1通道)</div>
<div></div>
<div>
<pre><span class="kn">import</span> <span class="nn">torch.nn</span> <span class="kn">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="kn">as</span> <span class="nn">F</span>


<span class="k">class</span> <span class="nc">Net</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">Net</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conv1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">5</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">pool</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">MaxPool2d</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conv2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">5</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">16</span> <span class="o">*</span> <span class="mi">5</span> <span class="o">*</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">120</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">120</span><span class="p">,</span> <span class="mi">84</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc3</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">84</span><span class="p">,</span> <span class="mi">10</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">pool</span><span class="p">(</span><span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">conv1</span><span class="p">(</span><span class="n">x</span><span class="p">)))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">pool</span><span class="p">(</span><span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">conv2</span><span class="p">(</span><span class="n">x</span><span class="p">)))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">16</span> <span class="o">*</span> <span class="mi">5</span> <span class="o">*</span> <span class="mi">5</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">fc3</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">x</span>


<span class="n">net</span> <span class="o">=</span> <span class="n">Net</span><span class="p">()</span></pre>
</div>
<div></div>
<div>

&nbsp;

</div>
<div>定义一个损失函数和优化器
让我们使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器。</div>
<div></div>
<div>
<pre><span class="kn">import</span> <span class="nn">torch.optim</span> <span class="kn">as</span> <span class="nn">optim</span>

<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">CrossEntropyLoss</span><span class="p">()</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">SGD</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">0.001</span><span class="p">,</span> <span class="n">momentum</span><span class="o">=</span><span class="mf">0.9</span><span class="p">)</span></pre>
</div>
<div></div>
<div>训练网络
这里事情开始变得有趣，我们只需要在数据迭代器上循环传给网络和优化器 输入就可以。</div>
<div></div>
<div>
<pre><span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="p">):</span>  <span class="c1"># loop over the dataset multiple times</span>

    <span class="n">running_loss</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">trainloader</span><span class="p">,</span> <span class="mi">0</span><span class="p">):</span>
        <span class="c1"># get the inputs</span>
        <span class="n">inputs</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="n">data</span>

        <span class="c1"># zero the parameter gradients</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>

        <span class="c1"># forward + backward + optimize</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="n">inputs</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="n">labels</span><span class="p">)</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

        <span class="c1"># print statistics</span>
        <span class="n">running_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">%</span> <span class="mi">2000</span> <span class="o">==</span> <span class="mi">1999</span><span class="p">:</span>    <span class="c1"># print every 2000 mini-batches</span>
            <span class="k">print</span><span class="p">(</span><span class="s1">'[</span><span class="si">%d</span><span class="s1">, </span><span class="si">%5d</span><span class="s1">] loss: </span><span class="si">%.3f</span><span class="s1">'</span> <span class="o">%</span>
                  <span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">running_loss</span> <span class="o">/</span> <span class="mi">2000</span><span class="p">))</span>
            <span class="n">running_loss</span> <span class="o">=</span> <span class="mf">0.0</span>

<span class="k">print</span><span class="p">(</span><span class="s1">'Finished Training'</span><span class="p">)</span></pre>
</div>
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight"> 输出：</div>
<div>
<pre>[1,  2000] loss: 2.187
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
Finished Training</pre>
</div>
</div>
在测试集上测试网络
我们已经通过训练数据集对网络进行了2次训练，但是我们需要检查网络是否已经学到了东西。

我们将用神经网络的输出作为预测的类标来检查网络的预测性能，用样本的真实类标来校对。如果预测是正确的，我们将样本添加到正确预测的列表里。

好的，第一步，让我们从测试集中显示一张图像来熟悉它。<img class="alignnone size-full wp-image-118" src="http://pytorchchina.com/wp-content/uploads/2018/12/sphx_glr_cifar10_tutorial_002.png" alt="" width="640" height="480" />

输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>GroundTruth:    cat  ship  ship plane
</pre>
</div>
</div>
现在让我们看看 神经网络认为这些样本应该预测成什么：
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">outputs</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="n">images</span><span class="p">)</span>
</pre>
</div>
</div>
输出是预测与十个类的近似程度，与某一个类的近似程度越高，网络就越认为图像是属于这一类别。所以让我们打印其中最相似类别类标：
<pre><span class="n">_</span><span class="p">,</span> <span class="n">predicted</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

<span class="k">print</span><span class="p">(</span><span class="s1">'Predicted: '</span><span class="p">,</span> <span class="s1">' '</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="s1">'</span><span class="si">%5s</span><span class="s1">'</span> <span class="o">%</span> <span class="n">classes</span><span class="p">[</span><span class="n">predicted</span><span class="p">[</span><span class="n">j</span><span class="p">]]</span>
                              <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">4</span><span class="p">)))</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>Predicted:    cat  ship   car  ship
</pre>
</div>
</div>
结果看起开非常好，让我们看看网络在整个数据集上的表现。
<pre><span class="n">correct</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">testloader</span><span class="p">:</span>
        <span class="n">images</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="n">data</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="n">images</span><span class="p">)</span>
        <span class="n">_</span><span class="p">,</span> <span class="n">predicted</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">total</span> <span class="o">+=</span> <span class="n">labels</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">correct</span> <span class="o">+=</span> <span class="p">(</span><span class="n">predicted</span> <span class="o">==</span> <span class="n">labels</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>

<span class="k">print</span><span class="p">(</span><span class="s1">'Accuracy of the network on the 10000 test images: </span><span class="si">%d</span> <span class="si">%%</span><span class="s1">'</span> <span class="o">%</span> <span class="p">(</span>
    <span class="mi">100</span> <span class="o">*</span> <span class="n">correct</span> <span class="o">/</span> <span class="n">total</span><span class="p">))</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>Accuracy of the network on the 10000 test images: 54 %
</pre>
</div>
</div>
这看起来比随机预测要好，随机预测的准确率为10%（随机预测出为10类中的哪一类）。看来网络学到了东西。
<pre><span class="n">class_correct</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="mf">0.</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">10</span><span class="p">))</span>
<span class="n">class_total</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="mf">0.</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">10</span><span class="p">))</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">testloader</span><span class="p">:</span>
        <span class="n">images</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="n">data</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="n">images</span><span class="p">)</span>
        <span class="n">_</span><span class="p">,</span> <span class="n">predicted</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">c</span> <span class="o">=</span> <span class="p">(</span><span class="n">predicted</span> <span class="o">==</span> <span class="n">labels</span><span class="p">)</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">4</span><span class="p">):</span>
            <span class="n">label</span> <span class="o">=</span> <span class="n">labels</span><span class="p">[</span><span class="n">i</span><span class="p">]</span>
            <span class="n">class_correct</span><span class="p">[</span><span class="n">label</span><span class="p">]</span> <span class="o">+=</span> <span class="n">c</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">item</span><span class="p">()</span>
            <span class="n">class_total</span><span class="p">[</span><span class="n">label</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>


<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">10</span><span class="p">):</span>
    <span class="k">print</span><span class="p">(</span><span class="s1">'Accuracy of </span><span class="si">%5s</span><span class="s1"> : </span><span class="si">%2d</span> <span class="si">%%</span><span class="s1">'</span> <span class="o">%</span> <span class="p">(</span>
        <span class="n">classes</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="mi">100</span> <span class="o">*</span> <span class="n">class_correct</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">/</span> <span class="n">class_total</span><span class="p">[</span><span class="n">i</span><span class="p">]))</span></pre>
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
<pre><span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="s2">"cuda:0"</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s2">"cpu"</span><span class="p">)</span>

<span class="c1"># Assume that we are on a CUDA machine, then this should print a CUDA device:</span>

<span class="k">print</span><span class="p">(</span><span class="n">device</span><span class="p">)</span></pre>
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
