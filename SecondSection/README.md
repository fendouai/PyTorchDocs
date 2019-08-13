## pytorch1.0-cn
pytorch1.0官方文档 中文版


PyTorch 入门教程【1】
https://github.com/fendouai/pytorch1.0-cn/blob/master/what-is-pytorch.md

PyTorch 自动微分【2】
https://github.com/fendouai/pytorch1.0-cn/blob/master/autograd-automatic-differentiation.md

PyTorch 神经网络【3】
https://github.com/fendouai/pytorch1.0-cn/blob/master/neural-networks.md

PyTorch 图像分类器【4】
https://github.com/fendouai/pytorch1.0-cn/blob/master/training-a-classifier.md

PyTorch 数据并行处理【5】
https://github.com/fendouai/pytorch1.0-cn/blob/master/optional-data-parallelism.md


## PytorchChina:
http://pytorchchina.com

## PyTorch 入门教程【1】
<h1>什么是 PyTorch?</h1>
PyTorch 是一个基于 Python 的科学计算包，主要定位两类人群：
<ul>
 	<li>NumPy 的替代品，可以利用 GPU 的性能进行计算。</li>
 	<li>深度学习研究平台拥有足够的灵活性和速度</li>
</ul>
<div id="getting-started" class="section">
<h2>开始学习</h2>
<div id="tensors" class="section">
<h3>Tensors (张量)</h3>
Tensors 类似于 NumPy 的 ndarrays ，同时  Tensors 可以使用 GPU 进行计算。
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="kn">from</span> <span class="nn">__future__</span> <span class="kn">import</span> <span class="n">print_function</span>
<span class="kn">import</span> <span class="nn">torch</span></pre>
</div>
</div>
</div>
</div>
构造一个5x3矩阵，不初始化。
<div id="getting-started" class="section">
<div id="tensors" class="section">
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">empty</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">(</span><span class="mf">1.00000e-04</span> <span class="o">*</span>
       <span class="p">[[</span><span class="o">-</span><span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">1.5135</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">,</span>  <span class="mf">0.0000</span><span class="p">]])</span></pre>
</div>
</div>
&nbsp;

构造一个随机初始化的矩阵：
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">rand</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span> <span class="mf">0.6291</span><span class="p">,</span>  <span class="mf">0.2581</span><span class="p">,</span>  <span class="mf">0.6414</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.9739</span><span class="p">,</span>  <span class="mf">0.8243</span><span class="p">,</span>  <span class="mf">0.2276</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.4184</span><span class="p">,</span>  <span class="mf">0.1815</span><span class="p">,</span>  <span class="mf">0.5131</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.5533</span><span class="p">,</span>  <span class="mf">0.5440</span><span class="p">,</span>  <span class="mf">0.0718</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">0.2908</span><span class="p">,</span>  <span class="mf">0.1850</span><span class="p">,</span>  <span class="mf">0.5297</span><span class="p">]])</span></pre>
</div>
</div>
&nbsp;

构造一个矩阵全为 0，而且数据类型是 long.

Construct a matrix filled zeros and of dtype long:
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">long</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">],</span>
        <span class="p">[</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">],</span>
        <span class="p">[</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">],</span>
        <span class="p">[</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">],</span>
        <span class="p">[</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">0</span><span class="p">]])</span></pre>
</div>
</div>
构造一个张量，直接使用数据：
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">([</span><span class="mf">5.5</span><span class="p">,</span> <span class="mi">3</span><span class="p">])</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([</span> <span class="mf">5.5000</span><span class="p">,</span>  <span class="mf">3.0000</span><span class="p">])</span></pre>
</div>
</div>
创建一个 tensor 基于已经存在的 tensor。
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">new_ones</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">double</span><span class="p">)</span>      
<span class="c1"># new_* methods take in sizes</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

<span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn_like</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float</span><span class="p">)</span>    
<span class="c1"># override dtype!</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>                                      
<span class="c1"># result has the same size</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">,</span>  <span class="mf">1.</span><span class="p">]],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float64</span><span class="p">)</span>
<span class="n">tensor</span><span class="p">([[</span><span class="o">-</span><span class="mf">0.2183</span><span class="p">,</span>  <span class="mf">0.4477</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.4053</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">1.7353</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.0048</span><span class="p">,</span>  <span class="mf">1.2177</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">1.1111</span><span class="p">,</span>  <span class="mf">1.0878</span><span class="p">,</span>  <span class="mf">0.9722</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.7771</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.2174</span><span class="p">,</span>  <span class="mf">0.0412</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">2.1750</span><span class="p">,</span>  <span class="mf">1.3609</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.3322</span><span class="p">]])</span></pre>
</div>
</div>
获取它的维度信息:
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">size</span><span class="p">())</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">输出:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">])</span></pre>
</div>
</div>
<div class="admonition note">
<p class="first admonition-title">注意</p>
<p class="last"><code class="docutils literal notranslate"><span class="pre">torch.Size</span></code>  是一个元组，所以它支持左右的元组操作。</p>
<p class="last"></p>

</div>
</div>
<div id="operations" class="section">
<h3>操作</h3>
在接下来的例子中，我们将会看到加法操作。

加法: 方式 1
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">y</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">rand</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span> <span class="o">+</span> <span class="n">y</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span><span class="o">-</span><span class="mf">0.1859</span><span class="p">,</span>  <span class="mf">1.3970</span><span class="p">,</span>  <span class="mf">0.5236</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">2.3854</span><span class="p">,</span>  <span class="mf">0.0707</span><span class="p">,</span>  <span class="mf">2.1970</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.3587</span><span class="p">,</span>  <span class="mf">1.2359</span><span class="p">,</span>  <span class="mf">1.8951</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.1189</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.1376</span><span class="p">,</span>  <span class="mf">0.4647</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">1.8968</span><span class="p">,</span>  <span class="mf">2.0164</span><span class="p">,</span>  <span class="mf">0.1092</span><span class="p">]])</span></pre>
</div>
</div>
加法: 方式2
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="k">print</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">))</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span><span class="o">-</span><span class="mf">0.1859</span><span class="p">,</span>  <span class="mf">1.3970</span><span class="p">,</span>  <span class="mf">0.5236</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">2.3854</span><span class="p">,</span>  <span class="mf">0.0707</span><span class="p">,</span>  <span class="mf">2.1970</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.3587</span><span class="p">,</span>  <span class="mf">1.2359</span><span class="p">,</span>  <span class="mf">1.8951</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.1189</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.1376</span><span class="p">,</span>  <span class="mf">0.4647</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">1.8968</span><span class="p">,</span>  <span class="mf">2.0164</span><span class="p">,</span>  <span class="mf">0.1092</span><span class="p">]])</span></pre>
</div>
</div>
加法: 提供一个输出 tensor 作为参数
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">result</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">empty</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
<span class="n">torch</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">out</span><span class="o">=</span><span class="n">result</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">result</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span><span class="o">-</span><span class="mf">0.1859</span><span class="p">,</span>  <span class="mf">1.3970</span><span class="p">,</span>  <span class="mf">0.5236</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">2.3854</span><span class="p">,</span>  <span class="mf">0.0707</span><span class="p">,</span>  <span class="mf">2.1970</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.3587</span><span class="p">,</span>  <span class="mf">1.2359</span><span class="p">,</span>  <span class="mf">1.8951</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.1189</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.1376</span><span class="p">,</span>  <span class="mf">0.4647</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">1.8968</span><span class="p">,</span>  <span class="mf">2.0164</span><span class="p">,</span>  <span class="mf">0.1092</span><span class="p">]])</span></pre>
</div>
</div>
加法: in-place
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="c1"># adds x to y</span>
<span class="n">y</span><span class="o">.</span><span class="n">add_</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">y</span><span class="p">)</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([[</span><span class="o">-</span><span class="mf">0.1859</span><span class="p">,</span>  <span class="mf">1.3970</span><span class="p">,</span>  <span class="mf">0.5236</span><span class="p">],</span>
        <span class="p">[</span> <span class="mf">2.3854</span><span class="p">,</span>  <span class="mf">0.0707</span><span class="p">,</span>  <span class="mf">2.1970</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.3587</span><span class="p">,</span>  <span class="mf">1.2359</span><span class="p">,</span>  <span class="mf">1.8951</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">0.1189</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.1376</span><span class="p">,</span>  <span class="mf">0.4647</span><span class="p">],</span>
        <span class="p">[</span><span class="o">-</span><span class="mf">1.8968</span><span class="p">,</span>  <span class="mf">2.0164</span><span class="p">,</span>  <span class="mf">0.1092</span><span class="p">]])</span></pre>
</div>
</div>
<div class="admonition note">
<p class="first admonition-title">Note</p>
注意

任何使张量会发生变化的操作都有一个前缀 '_'。例如：<code class="docutils literal notranslate"><span class="pre">x.copy_(y)</span></code>, <code class="docutils literal notranslate"><span class="pre">x.t_()</span></code>, 将会改变 <code class="docutils literal notranslate"><span class="pre">x</span></code>.
<p class="last">你可以使用标准的  NumPy 类似的索引操作</p>

</div>
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">])</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([</span> <span class="mf">0.4477</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.0048</span><span class="p">,</span>  <span class="mf">1.0878</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.2174</span><span class="p">,</span>  <span class="mf">1.3609</span><span class="p">])</span></pre>
</div>
</div>
改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 <code class="docutils literal notranslate"><span class="pre">torch.view</span></code>:
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">4</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="mi">16</span><span class="p">)</span>
<span class="n">z</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">8</span><span class="p">)</span>  <span class="c1"># the size -1 is inferred from other dimensions</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">size</span><span class="p">(),</span> <span class="n">y</span><span class="o">.</span><span class="n">size</span><span class="p">(),</span> <span class="n">z</span><span class="o">.</span><span class="n">size</span><span class="p">())</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">4</span><span class="p">])</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">16</span><span class="p">])</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">8</span><span class="p">])</span></pre>
</div>
</div>
如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。
<div class="highlight-python notranslate">
<div class="highlight">
<pre><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">item</span><span class="p">())</span></pre>
</div>
</div>
<p class="sphx-glr-script-out">Out:</p>

<div class="sphx-glr-script-out highlight-default notranslate">
<div class="highlight">
<pre><span class="n">tensor</span><span class="p">([</span> <span class="mf">0.9422</span><span class="p">])</span>
<span class="mf">0.9422121644020081</span></pre>
</div>
</div>
</div>
</div>
<div id="cuda-tensors" class="section">
 
 
<div class="sphx-glr-footer docutils container">
<div>PyTorch windows 安装教程：两行代码搞定 PyTorch 安装
http://pytorchchina.com/2018/12/11/pytorch-windows-install-1/
 
PyTorch Mac 安装教程
http://pytorchchina.com/2018/12/11/pytorch-mac-install/

PyTorch Linux 安装教程
http://pytorchchina.com/2018/12/11/pytorch-linux-install/

</div>
<div></div>


<div class="sphx-glr-download docutils container">PyTorch QQ群</div>
<div></div>
<div class="sphx-glr-download docutils container"><img class="alignnone wp-image-47 size-full" src="http://pytorchchina.com/wp-content/uploads/2018/12/WechatIMG1311.png" alt="" width="540" height="740" /></div>
</div>
</div>


## PyTorch 自动微分【2】
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


## PyTorch 神经网络【3】


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


## PyTorch 图像分类器【4】
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


## PyTorch 数据并行处理【5】
可选择：数据并行处理（文末有完整代码下载）
作者：Sung Kim 和 Jenny Kang

在这个教程中，我们将学习如何用 DataParallel 来使用多 GPU。
通过 PyTorch 使用多个 GPU 非常简单。你可以将模型放在一个 GPU：
<pre> device = torch.device("cuda:0")
 model.to(device)</pre>
然后，你可以复制所有的张量到 GPU：
<pre> mytensor = my_tensor.to(device)</pre>
请注意，只是调用 my_tensor.to(device) 返回一个 my_tensor 新的复制在GPU上，而不是重写 my_tensor。你需要分配给他一个新的张量并且在 GPU 上使用这个张量。

在多 GPU 中执行前馈，后馈操作是非常自然的。尽管如此，PyTorch 默认只会使用一个 GPU。通过使用 DataParallel 让你的模型并行运行，你可以很容易的在多 GPU 上运行你的操作。
<pre> model = nn.DataParallel(model)</pre>
这是整个教程的核心，我们接下来将会详细讲解。
引用和参数

引入 PyTorch 模块和定义参数
<pre> import torch
 import torch.nn as nn
 from torch.utils.data import Dataset, DataLoader</pre>
### 参数
<pre> input_size = 5
 output_size = 2

 batch_size = 30
 data_size = 100</pre>
设备
<pre>device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")</pre>
实验（玩具）数据

生成一个玩具数据。你只需要实现 getitem.
<pre><span class="k">class</span> <span class="nc">RandomDataset</span><span class="p">(</span><span class="n">Dataset</span><span class="p">):</span>

    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">size</span><span class="p">,</span> <span class="n">length</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">len</span> <span class="o">=</span> <span class="n">length</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">data</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">length</span><span class="p">,</span> <span class="n">size</span><span class="p">)</span>

    <span class="k">def</span> <span class="fm">__getitem__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">index</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="n">index</span><span class="p">]</span>

    <span class="k">def</span> <span class="fm">__len__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">len</span>

<span class="n">rand_loader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">RandomDataset</span><span class="p">(</span><span class="n">input_size</span><span class="p">,</span> <span class="n">data_size</span><span class="p">),</span><span class="n">batch_size</span><span class="o">=</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span></pre>
简单模型

为了做一个小 demo，我们的模型只是获得一个输入，执行一个线性操作，然后给一个输出。尽管如此，你可以使用 DataParallel   在任何模型(CNN, RNN, Capsule Net 等等.)

我们放置了一个输出声明在模型中来检测输出和输入张量的大小。请注意在 batch rank 0 中的输出。
<pre><span class="k">class</span> <span class="nc">Model</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="c1"># Our model</span>

    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">input_size</span><span class="p">,</span> <span class="n">output_size</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">Model</span><span class="p">,</span> <span class="bp">self</span><span class="p">)</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">input_size</span><span class="p">,</span> <span class="n">output_size</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="nb">input</span><span class="p">):</span>
        <span class="n">output</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">fc</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>
        <span class="k">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\t</span><span class="s2">In Model: input size"</span><span class="p">,</span> <span class="nb">input</span><span class="o">.</span><span class="n">size</span><span class="p">(),</span>
              <span class="s2">"output size"</span><span class="p">,</span> <span class="n">output</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>

        <span class="k">return</span> <span class="n">output</span></pre>
&nbsp;

创建模型并且数据并行处理

这是整个教程的核心。首先我们需要一个模型的实例，然后验证我们是否有多个 GPU。如果我们有多个 GPU，我们可以用 nn.DataParallel 来   包裹 我们的模型。然后我们使用 model.to(device) 把模型放到多 GPU 中。

&nbsp;
<pre><span class="n">model</span> <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">input_size</span><span class="p">,</span> <span class="n">output_size</span><span class="p">)</span>
<span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">device_count</span><span class="p">()</span> <span class="o">&gt;</span> <span class="mi">1</span><span class="p">:</span>
  <span class="k">print</span><span class="p">(</span><span class="s2">"Let's use"</span><span class="p">,</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">device_count</span><span class="p">(),</span> <span class="s2">"GPUs!"</span><span class="p">)</span>
  <span class="c1"># dim = 0 [30, xxx] -&gt; [10, ...], [10, ...], [10, ...] on 3 GPUs</span>
  <span class="n">model</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">DataParallel</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>

<span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span></pre>
输出：
<div id="create-model-and-dataparallel" class="section">
<div class="sphx-glr-script-out highlight-none notranslate">
<div class="highlight">
<pre>Let's use 2 GPUs!
</pre>
</div>
</div>
</div>
<div id="run-the-model" class="section"> 运行模型：</div>
<div>现在我们可以看到输入和输出张量的大小了。</div>
<div></div>
<div>
<pre><span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">rand_loader</span><span class="p">:</span>
    <span class="nb">input</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
    <span class="n">output</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>
    <span class="k">print</span><span class="p">(</span><span class="s2">"Outside: input size"</span><span class="p">,</span> <span class="nb">input</span><span class="o">.</span><span class="n">size</span><span class="p">(),</span>
          <span class="s2">"output_size"</span><span class="p">,</span> <span class="n">output</span><span class="o">.</span><span class="n">size</span><span class="p">())</span></pre>
</div>
输出：
<pre>In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])</pre>
结果：

如果你没有 GPU 或者只有一个 GPU，当我们获取 30 个输入和 30 个输出，模型将期望获得 30 个输入和 30 个输出。但是如果你有多个 GPU ，你会获得这样的结果。

多 GPU

如果你有 2 个GPU，你会看到：
<div id="gpus" class="section">
<pre><span class="c1"># on 2 GPUs</span>
<span class="n">Let</span><span class="s1">'s use 2 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">15</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">5</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
</div>
&nbsp;

如果你有 3个GPU，你会看到：
<pre><span class="n">Let</span><span class="s1">'s use 3 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
如果你有 8个GPU，你会看到：
<pre><span class="n">Let</span><span class="s1">'s use 8 GPUs!</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">30</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
    <span class="n">In</span> <span class="n">Model</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
<span class="n">Outside</span><span class="p">:</span> <span class="nb">input</span> <span class="n">size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="n">output_size</span> <span class="n">torch</span><span class="o">.</span><span class="n">Size</span><span class="p">([</span><span class="mi">10</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span></pre>
<h2>总结</h2>
数据并行自动拆分了你的数据并且将任务单发送到多个 GPU 上。当每一个模型都完成自己的任务之后，DataParallel 收集并且合并这些结果，然后再返回给你。

更多信息，请访问：
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

下载 Python 版本完整代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/data_parallel_tutorial.py_.zip">data_parallel_tutorial.py</a>

下载 jupyter notebook 版本完整代码：

<a href="http://pytorchchina.com/wp-content/uploads/2018/12/data_parallel_tutorial.ipynb_.zip">data_parallel_tutorial.ipynb</a>

加入 PyTorch 交流 QQ 群：

<img class="alignnone wp-image-47 size-full" src="http://pytorchchina.com/wp-content/uploads/2018/12/WechatIMG1311.png" alt="" width="540" height="740" />





# PytorchChina:
http://pytorchchina.com
