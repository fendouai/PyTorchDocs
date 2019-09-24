autograd 包是 PyTorch 中所有神经网络的核心。首先让我们简要地介绍它，然后我们将会去训练我们的第一个神经网络。该 autograd 软件包为 Tensors 上的所有操作提供自动微分。它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。我们从 tensor 和 gradients 来举一些例子。

1、TENSOR

torch.Tensor 是包的核心类。如果将其属性 .requires_grad 设置为 True，则会开始跟踪针对 tensor 的所有操作。完成计算后，您可以调用 .backward() 来自动计算所有梯度。该张量的梯度将累积到 .grad 属性中。

要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。

要停止跟踪历史记录（和使用内存），您还可以将代码块使用 with torch.no_grad(): 包装起来。在评估模型时，这是特别有用，因为模型在训练阶段具有 requires_grad = True 的可训练参数有利于调参，但在评估阶段我们不需要梯度。

还有一个类对于 autograd 实现非常重要那就是 Function。Tensor 和 Function 互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则g rad_fn 是 None ）。

如果你想计算导数，你可以调用 Tensor.backward()。如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。
<pre>import torch</pre>
创建一个张量，设置 requires_grad=True 来跟踪与它相关的计算
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">requires_grad</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></pre>
输出：
<pre>tensor([[1., 1.],
        [1., 1.]], requires_grad=True)</pre>
针对张量做一个操作
<pre><span class="n">y</span> <span class="o">=</span> <span class="n">x</span> <span class="o">+</span> <span class="mi">2</span>
<span class="k">print</span><span class="p">(</span><span class="n">y</span><span class="p">)</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>tensor([[3., 3.],
        [3., 3.]], grad_fn=&lt;AddBackward0&gt;)
</pre>
</div>
</div>
y 作为操作的结果被创建，所以它有 grad_fn
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="k">print</span><span class="p">(</span><span class="n">y</span><span class="o">.</span><span class="n">grad_fn</span><span class="p">)</span></pre>
</div>
</div>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>&lt;AddBackward0 object at 0x7fe1db427470&gt;
</pre>
</div>
</div>
针对 y 做更多的操作：
<pre><span class="n">z</span> <span class="o">=</span> <span class="n">y</span> <span class="o">*</span> <span class="n">y</span> <span class="o">*</span> <span class="mi">3</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">z</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>

<span class="k">print</span><span class="p">(</span><span class="n">z</span><span class="p">,</span> <span class="n">out</span><span class="p">)</span></pre>
输出：
<pre>tensor([[27., 27.],
        [27., 27.]], grad_fn=&lt;MulBackward0&gt;) tensor(27., grad_fn=&lt;MeanBackward0&gt;)</pre>
<code class="docutils literal notranslate"><span class="pre">.requires_grad_(</span> <span class="pre">...</span> <span class="pre">)</span></code> 会改变张量的 <code class="docutils literal notranslate"><span class="pre">requires_grad</span></code> 标记。输入的标记默认为  <code class="docutils literal notranslate"><span class="pre">False</span></code> ，如果没有提供相应的参数。
<pre><span class="n">a</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>
<span class="n">a</span> <span class="o">=</span> <span class="p">((</span><span class="n">a</span> <span class="o">*</span> <span class="mi">3</span><span class="p">)</span> <span class="o">/</span> <span class="p">(</span><span class="n">a</span> <span class="o">-</span> <span class="mi">1</span><span class="p">))</span>
<span class="k">print</span><span class="p">(</span><span class="n">a</span><span class="o">.</span><span class="n">requires_grad</span><span class="p">)</span>
<span class="n">a</span><span class="o">.</span><span class="n">requires_grad_</span><span class="p">(</span><span class="bp">True</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">a</span><span class="o">.</span><span class="n">requires_grad</span><span class="p">)</span>
<span class="n">b</span> <span class="o">=</span> <span class="p">(</span><span class="n">a</span> <span class="o">*</span> <span class="n">a</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="k">print</span><span class="p">(</span><span class="n">b</span><span class="o">.</span><span class="n">grad_fn</span><span class="p">)</span></pre>
输出：
<pre>False
True
&lt;SumBackward0 object at 0x7fe1db427dd8&gt;</pre>
梯度：

我们现在后向传播，因为输出包含了一个标量，<code class="docutils literal notranslate"><span class="pre">out.backward()</span></code> 等同于<code class="docutils literal notranslate"><span class="pre">out.backward(torch.tensor(1.))</span></code>。
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">out</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span></pre>
</div>
</div>
打印梯度  d(out)/dx
<pre><span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">grad</span><span class="p">)</span></pre>
<div class="highlight-python notranslate">
<div class="highlight"> 输出：</div>
<div>
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])</pre>
</div>
</div>
</div>
</div>
&nbsp;

原理解释：

<img class="alignnone size-full wp-image-106" src="http://pytorchchina.com/wp-content/uploads/2018/12/WechatIMG1376.jpeg" alt="" width="3544" height="1952" />
<div class="trans-left">
<div class="trans-input-wrap">
<div class="input-wrap" dir="ltr">
<div class="input-operate">
<div class="op-favor-container"></div>
</div>
</div>
</div>
</div>
<div class="trans-right">
<div class="output-wrap small-font">
<div class="output-mod ordinary-wrap">
<div class="output-bd" dir="ltr">
<p class="ordinary-output target-output clearfix"><span class="">现在让我们看一个雅可比向量积的例子：</span></p>

</div>
</div>
</div>
</div>
<div class="highlight-python notranslate">
<div class="highlight"></div>
</div>
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">requires_grad</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span>

<span class="n">y</span> <span class="o">=</span> <span class="n">x</span> <span class="o">*</span> <span class="mi">2</span>
<span class="k">while</span> <span class="n">y</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">norm</span><span class="p">()</span> <span class="o">&lt;</span> <span class="mi">1000</span><span class="p">:</span>
    <span class="n">y</span> <span class="o">=</span> <span class="n">y</span> <span class="o">*</span> <span class="mi">2</span>

<span class="k">print</span><span class="p">(</span><span class="n">y</span><span class="p">)</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>tensor([ -444.6791,   762.9810, -1690.0941], grad_fn=&lt;MulBackward0&gt;)</pre>
</div>
</div>
&nbsp;

现在在这种情况下，y 不再是一个标量。torch.autograd 不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。
<pre><span class="n">v</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">([</span><span class="mf">0.1</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="mf">0.0001</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float</span><span class="p">)</span>
<span class="n">y</span><span class="o">.</span><span class="n">backward</span><span class="p">(</span><span class="n">v</span><span class="p">)</span>

<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">grad</span><span class="p">)</span></pre>
输出：
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])</pre>
</div>
</div>
&nbsp;

你可以通过将代码包裹在 <span class="pre">with</span> <span class="pre">torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。</span>
<pre><span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">requires_grad</span><span class="p">)</span>
<span class="k">print</span><span class="p">((</span><span class="n">x</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">requires_grad</span><span class="p">)</span>

<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">print</span><span class="p">((</span><span class="n">x</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">requires_grad</span><span class="p">)</span></pre>
输出：
<pre>True
True
False</pre>
稍后可以阅读：

<code class="docutils literal notranslate"><span class="pre">autograd</span></code> 和 <code class="docutils literal notranslate"><span class="pre">Function</span></code> 的文档在： <a class="reference external" href="https://pytorch.org/docs/autograd">https://pytorch.org/docs/autograd</a>

下载 Python 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/autograd_tutorial.py_.zip">autograd_tutorial.py</a>

下载 Jupyter 源代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/autograd_tutorial.ipynb_.zip">autograd_tutorial.ipynb</a>
