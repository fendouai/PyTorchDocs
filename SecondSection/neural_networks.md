神经网络

神经网络可以通过 torch.nn 包来构建。

现在对于自动梯度(autograd)有一些了解，神经网络是基于自动梯度 (autograd)来定义一些模型。一个 nn.Module 包括层和一个方法 forward(input) 它会返回输出(output)。

例如，看一下数字图片识别的网络：

<img class="alignnone size-full wp-image-126" src="http://pytorchchina.com/wp-content/uploads/2018/12/mnist.png" alt="" width="759" height="209" />

这是一个简单的前馈神经网络，它接收输入，让输入一个接着一个的通过一些层，最后给出输出。

一个典型的神经网络训练过程包括以下几点：

1.定义一个包含可训练参数的神经网络

2.迭代整个输入

3.通过神经网络处理输入

4.计算损失(loss)

5.反向传播梯度到神经网络的参数

6.更新网络的参数，典型的用一个简单的更新方法：<span class="pre">weight</span> <span class="pre">=</span> <span class="pre">weight</span> <span class="pre">-</span> <span class="pre">learning_rate</span> <span class="pre">*</span><span class="pre">gradient</span>

定义神经网络
<pre><span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="kn">as</span> <span class="nn">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="kn">as</span> <span class="nn">F</span>


<span class="k">class</span> <span class="nc">Net</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>

    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">Net</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="c1"># 1 input image channel, 6 output channels, 5x5 square convolution</span>
        <span class="c1"># kernel</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conv1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">5</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conv2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Conv2d</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">5</span><span class="p">)</span>
        <span class="c1"># an affine operation: y = Wx + b</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">16</span> <span class="o">*</span> <span class="mi">5</span> <span class="o">*</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">120</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">120</span><span class="p">,</span> <span class="mi">84</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc3</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">84</span><span class="p">,</span> <span class="mi">10</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="c1"># Max pooling over a (2, 2) window</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">max_pool2d</span><span class="p">(</span><span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">conv1</span><span class="p">(</span><span class="n">x</span><span class="p">)),</span> <span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">))</span>
        <span class="c1"># If the size is a square you can only specify a single number</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">max_pool2d</span><span class="p">(</span><span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">conv2</span><span class="p">(</span><span class="n">x</span><span class="p">)),</span> <span class="mi">2</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">num_flat_features</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">fc3</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">x</span>

    <span class="k">def</span> <span class="nf">num_flat_features</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">size</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">size</span><span class="p">()[</span><span class="mi">1</span><span class="p">:]</span>  <span class="c1"># all dimensions except the batch dimension</span>
        <span class="n">num_features</span> <span class="o">=</span> <span class="mi">1</span>
        <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">size</span><span class="p">:</span>
            <span class="n">num_features</span> <span class="o">*=</span> <span class="n">s</span>
        <span class="k">return</span> <span class="n">num_features</span>


<span class="n">net</span> <span class="o">=</span> <span class="n">Net</span><span class="p">()</span>
<span class="k">print</span><span class="p">(</span><span class="n">net</span><span class="p">)</span></pre>
输出：
<pre>Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)</pre>
你刚定义了一个前馈函数，然后反向传播函数被自动通过 autograd 定义了。你可以使用任何张量操作在前馈函数上。

一个模型可训练的参数可以通过调用 net.parameters() 返回：
<pre><span class="n">params</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">parameters</span><span class="p">())</span>
<span class="k">print</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">params</span><span class="p">))</span>
<span class="k">print</span><span class="p">(</span><span class="n">params</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>  <span class="c1"># conv1's .weight</span></pre>
输出：
<pre>10
torch.Size([6, 1, 5, 5])</pre>
让我们尝试随机生成一个 32x32 的输入。注意：期望的输入维度是 32x32 。为了使用这个网络在 MNIST 数据及上，你需要把数据集中的图片维度修改为 32x32。
<pre><span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">out</span><span class="p">)</span></pre>
输出：
<pre>tensor([[-0.0233,  0.0159, -0.0249,  0.1413,  0.0663,  0.0297, -0.0940, -0.0135,
          0.1003, -0.0559]], grad_fn=&lt;AddmmBackward&gt;)</pre>
把所有参数梯度缓存器置零，用随机的梯度来反向传播
<pre><span class="n">net</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
<span class="n">out</span><span class="o">.</span><span class="n">backward</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span></pre>
在继续之前，让我们复习一下所有见过的类。

torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation, creates at least a single Function node, that connects to functions that created a Tensor and encodes its history.

在此，我们完成了：

1.定义一个神经网络

2.处理输入以及调用反向传播

还剩下：

1.计算损失值

2.更新网络中的权重

损失函数

一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。

有一些不同的损失函数在 nn 包中。一个简单的损失函数就是 nn.MSELoss ，这计算了均方误差。

例如：
<pre><span class="n">output</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>
<span class="n">target</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>  <span class="c1"># a dummy target, for example</span>
<span class="n">target</span> <span class="o">=</span> <span class="n">target</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># make it the same shape as output</span>
<span class="n">criterion</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">MSELoss</span><span class="p">()</span>

<span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">output</span><span class="p">,</span> <span class="n">target</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">loss</span><span class="p">)</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>tensor(1.3389, grad_fn=&lt;MseLossBackward&gt;)</pre>
</div>
</div>
现在，如果你跟随损失到反向传播路径，可以使用它的 .grad_fn 属性，你将会看到一个这样的计算图：
<pre><span class="nb">input</span> <span class="o">-&gt;</span> <span class="n">conv2d</span> <span class="o">-&gt;</span> <span class="n">relu</span> <span class="o">-&gt;</span> <span class="n">maxpool2d</span> <span class="o">-&gt;</span> <span class="n">conv2d</span> <span class="o">-&gt;</span> <span class="n">relu</span> <span class="o">-&gt;</span> <span class="n">maxpool2d</span>
      <span class="o">-&gt;</span> <span class="n">view</span> <span class="o">-&gt;</span> <span class="n">linear</span> <span class="o">-&gt;</span> <span class="n">relu</span> <span class="o">-&gt;</span> <span class="n">linear</span> <span class="o">-&gt;</span> <span class="n">relu</span> <span class="o">-&gt;</span> <span class="n">linear</span>
      <span class="o">-&gt;</span> <span class="n">MSELoss</span>
      <span class="o">-&gt;</span> <span class="n">loss</span></pre>
所以，当我们调用 loss.backward()，整个图都会微分，而且所有的在图中的requires_grad=True 的张量将会让他们的 grad 张量累计梯度。

为了演示，我们将跟随以下步骤来反向传播。
<pre><span class="k">print</span><span class="p">(</span><span class="n">loss</span><span class="o">.</span><span class="n">grad_fn</span><span class="p">)</span>  <span class="c1"># MSELoss</span>
<span class="k">print</span><span class="p">(</span><span class="n">loss</span><span class="o">.</span><span class="n">grad_fn</span><span class="o">.</span><span class="n">next_functions</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span>  <span class="c1"># Linear</span>
<span class="k">print</span><span class="p">(</span><span class="n">loss</span><span class="o">.</span><span class="n">grad_fn</span><span class="o">.</span><span class="n">next_functions</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">next_functions</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span>  <span class="c1"># ReLU</span></pre>
输出：
<pre>&lt;MseLossBackward object at 0x7fab77615278&gt;
&lt;AddmmBackward object at 0x7fab77615940&gt;
&lt;AccumulateGrad object at 0x7fab77615940&gt;</pre>
反向传播

为了实现反向传播损失，我们所有需要做的事情仅仅是使用 loss.backward()。你需要清空现存的梯度，要不然帝都将会和现存的梯度累计到一起。

现在我们调用 loss.backward() ，然后看一下 con1 的偏置项在反向传播之前和之后的变化。
<pre><span class="n">net</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>     <span class="c1"># zeroes the gradient buffers of all parameters</span>

<span class="k">print</span><span class="p">(</span><span class="s1">'conv1.bias.grad before backward'</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">conv1</span><span class="o">.</span><span class="n">bias</span><span class="o">.</span><span class="n">grad</span><span class="p">)</span>

<span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>

<span class="k">print</span><span class="p">(</span><span class="s1">'conv1.bias.grad after backward'</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">conv1</span><span class="o">.</span><span class="n">bias</span><span class="o">.</span><span class="n">grad</span><span class="p">)</span></pre>
输出：
<pre>conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0054,  0.0011,  0.0012,  0.0148, -0.0186,  0.0087])</pre>
现在我们看到了，如何使用损失函数。

唯一剩下的事情就是更新神经网络的参数。

更新神经网络参数：

最简单的更新规则就是随机梯度下降。
<blockquote>
<div><code class="docutils literal notranslate"><span class="pre">weight</span> <span class="pre">=</span> <span class="pre">weight</span> <span class="pre">-</span> <span class="pre">learning_rate</span> <span class="pre">*</span> <span class="pre">gradient</span></code></div></blockquote>
我们可以使用 python 来实现这个规则：
<pre><span class="n">learning_rate</span> <span class="o">=</span> <span class="mf">0.01</span>
<span class="k">for</span> <span class="n">f</span> <span class="ow">in</span> <span class="n">net</span><span class="o">.</span><span class="n">parameters</span><span class="p">():</span>
    <span class="n">f</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">sub_</span><span class="p">(</span><span class="n">f</span><span class="o">.</span><span class="n">grad</span><span class="o">.</span><span class="n">data</span> <span class="o">*</span> <span class="n">learning_rate</span><span class="p">)</span></pre>
尽管如此，如果你是用神经网络，你想使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp, 等。为了让这可行，我们建立了一个小包：torch.optim 实现了所有的方法。使用它非常的简单。
<pre><span class="kn">import</span> <span class="nn">torch.optim</span> <span class="kn">as</span> <span class="nn">optim</span>

<span class="c1"># create your optimizer</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">SGD</span><span class="p">(</span><span class="n">net</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">0.01</span><span class="p">)</span>

<span class="c1"># in your training loop:</span>
<span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>   <span class="c1"># zero the gradient buffers</span>
<span class="n">output</span> <span class="o">=</span> <span class="n">net</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">criterion</span><span class="p">(</span><span class="n">output</span><span class="p">,</span> <span class="n">target</span><span class="p">)</span>
<span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
<span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>    <span class="c1"># Does the update</span></pre>
下载 Python 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/neural_networks_tutorial.py_.zip">neural_networks_tutorial.py</a>

下载 Jupyter 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/neural_networks_tutorial.ipynb_.zip">neural_networks_tutorial.ipynb</a>
