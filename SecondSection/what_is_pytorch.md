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
