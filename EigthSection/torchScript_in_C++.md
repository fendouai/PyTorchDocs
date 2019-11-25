# 在C++中加载TorchScript模型
本教程已更新为可与PyTorch 1.2一起使用

顾名思义，PyTorch的主要接口是Python编程语言。尽管Python是合适于许多需要动态性和易于迭代的场景，并且是首选的语言，但同样的，在
许多情况下，Python的这些属性恰恰是不利的。后者通常适用的一种环境是要求生产-低延迟和严格部署。对于生产场景，即使只将C ++绑定到Java，
Rust或Go之类的另一种语言中，它也是经常选择的语言。以下各段将概述PyTorch提供的从现有Python模型到可以完全从C ++加载和执行的序
列化表示形式的路径，而无需依赖Python。

# 步骤1：将PyTorch模型转换为Torch脚本
PyTorch模型从Python到C ++的旅程由Torch Script启动，Torch Script是PyTorch模型的一种表示形式，可以由Torch Script编译器理解，
编译和序列化。如果您是从使用vanilla“eager” API编写的现有PyTorch模型开始的，则必须首先将模型转换为Torch脚本。在最常见的情况
下（如下所述），这只需要花费很少的功夫。如果您已经有了Torch脚本模块，则可以跳到本教程的下一部分。

有两种将PyTorch模型转换为Torch脚本的方法。第一种称为跟踪，一种机制，其中通过使用示例输入对模型的结构进行一次评估，并记录这些
输入在模型中的流量，从而捕获模型的结构。这适用于有限使用控制流的模型。第二种方法是在模型中添加显式批注，以告知Torch Script编
译器可以根据Torch Script语言施加的约束直接解析和编译模型代码。

> 提示：您可以在官方[Torch脚本参考](https://pytorch.org/docs/master/jit.html)中找到有关这两种方法的完整文档，以及使用方法的进一步指导。

## 方法1：通过跟踪转换为Torch脚本
要将PyTorch模型通过跟踪转换为Torch脚本，必须将模型的实例以及示例输入传递给`torch.jit.trace`函数。这将产生一个`torch.jit.ScriptModule`
对象，该对象的模型评估痕迹将嵌入模块的`forward`方法中：

```buildoutcfg
import torch
import torchvision

# 你模型的一个实例.
model = torchvision.models.resnet18()

# 您通常会提供给模型的forward()方法的示例输入。
example = torch.rand(1, 3, 224, 224)

# 使用`torch.jit.trace `来通过跟踪生成`torch.jit.ScriptModule`
traced_script_module = torch.jit.trace(model, example)
```

现在可以对跟踪的`ScriptModule`进行评估，使其与常规PyTorch模块相同：

```buildoutcfg
In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
In[2]: output[0, :5]
Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

## 方法2：通过注释转换为Torch脚本
在某些情况下，例如，如果模型采用特定形式的控制流，则可能需要直接在Torch脚本中编写模型并相应地注释模型。例如，假设您具有以下
vanilla Pytorch模型：

```buildoutcfg
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output
```

因为此模块的前向方法使用取决于输入的控制流，所以它不适合跟踪。相反，我们可以将其转换为`ScriptModule`。为了将模块转换为
`ScriptModule`，需要使用`torch.jit.script`编译模块，如下所示：

```buildoutcfg
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
```

如果您需要在`nn.Module`中排除某些方法，因为它们使用了`TorchScript`尚不支持的Python功能，则可以使用`@torch.jit.ignore`对其进行注释

`my_module`是`ScriptModule`的实例，可以序列化。

# 步骤2：将脚本模块序列化为文件
一旦有了ScriptModule（通过跟踪或注释PyTorch模型），您就可以将其序列化为文件了。稍后，您将可以使用C ++从此文件加载模块并执行它，
而无需依赖Python。假设我们要序列化先前在跟踪示例中显示的`ResNet18`模型。要执行此序列化，只需在模块上调用[save](https://pytorch.org/docs/master/jit.html#torch.jit.ScriptModule.save)并传递一个文件名即可：

```buildoutcfg
traced_script_module.save("traced_resnet_model.pt")
```

这将在您的工作目录中生成`traced_resnet_model.pt`文件。如果您还想序列化`my_module`，请调用`my_module.save("my_module_model.pt")`
我们现在已经正式离开Python领域，并准备跨入C ++领域。

# 步骤3：在C ++中加载脚本模块
要在C ++中加载序列化的PyTorch模型，您的应用程序必须依赖于PyTorch C ++ API（也称为LibTorch）。LibTorch发行版包含共享库，头文件
和CMake构建配置文件的集合。虽然CMake不是依赖LibTorch的要求，但它是推荐的方法，并且将来会得到很好的支持。 对于本教程，我们将
使用CMake和LibTorch构建一个最小的C ++应用程序，该应用程序简单地加载并执行序列化的PyTorch模型。

### 最小的C ++应用程序
让我们从讨论加载模块的代码开始。以下将已经做：

```buildoutcfg
include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // 使用以下命令从文件中反序列化脚本模块: torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}
```

`<torch/script.h>`标头包含运行示例所需的LibTorch库中的所有相关包含。我们的应用程序接受序列化的PyTorch ScriptModule的文件路径
作为其唯一的命令行参数，然后使用`torch::jit::load（）`函数继续对该模块进行反序列化，该函数将此文件路径作为输入。作为返回，我们
收到一个`Torch::jit::script::Module`对象。我们将稍后讨论如何执行它。

### 取决于LibTorch和构建应用程序
假设我们将以上代码存储在名为`example-app.cpp`的文件中。最小的`CMakeLists.txt`可能看起来很简单：

```buildoutcfg
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
```

建立示例应用程序的最后一件事是LibTorch发行版。您可以随时从PyTorch网站的[下载页面](https://pytorch.org/)上获取最新的稳定版本。
如果下载并解压缩最新的归档文件，则应收到具有以下目录结构的文件夹：
```buildoutcfg
libtorch/
  bin/
  include/
  lib/
  share/
```

* lib/ 文件夹包含您必须链接的共享库，

* include/ 文件夹包含程序需要包含的头文件，

* share/ 文件夹包含必要的CMake配置，以启用上面的简单`find_package(Torch)`命令。

> 提示;在Windows上，调试和发行版本不兼容ABI。 如果您打算以调试模式构建项目，请尝试使用LibTorch的调试版本。

最后一步是构建应用程序。为此，假定示例目录的布局如下：
```buildoutcfg
example-app/
  CMakeLists.txt
  example-app.cpp
```

现在，我们可以运行以下命令从`example-app/`文件夹中构建应用程序：
```buildoutcfg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

`/path/to/libtorch`应该是解压缩的LibTorch发行版的完整路径。如果一切顺利，它将看起来像这样：

```buildoutcfg
root@4b5a67132e81:/example-app# mkdir build
root@4b5a67132e81:/example-app# cd build
root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring done
-- Generating done
-- Build files have been written to: /example-app/build
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
```

如果我们提供了我们之前创建的到示例应用程序二进制文件的跟踪ResNet18模型`traced_resnet_model.pt`的路径，则应该以友好的“ ok”
作为奖励。 请注意，如果尝试使用`my_module_model.pt`运行此示例，则会收到一条错误消息，提示您输入的形状不兼容。`my_module_model.pt`
需要1D而不是4D。

```buildoutcfg
root@4b5a67132e81:/example-app/build# ./example-app <path_to_model>/traced_resnet_model.pt
ok
```

# 步骤4：在C ++中执行脚本模块
成功用C ++加载了序列化的ResNet18之后，我们现在只需执行几行代码即可！让我们将这些行添加到C ++应用程序的`main()`函数中：

```buildoutcfg
// 创建输入向量
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// 执行模型并将输出转化为张量
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
```

前两行设置了我们模型的输入。我们创建一个`torch::jit::IValue`的向量（类型为type-erased的值`Script::Module`方法接受并返回），
并添加单个输入。要创建输入张量，我们使用`torch::ones()`，等效于C ++ API中的`torch.ones`。然后，我们运行`script::Module`的
`forward`方法，并向其传递我们创建的输入向量。作为回报，我们得到一个新的IValue，通过调用`toTensor()`将其转换为张量。

> 提示：要总体上了解有关torch::ones和PyTorch C ++ API之类的功能的更多信息，请参阅其文档，网址为https://pytorch.org/cppdocs。
PyTorch C ++ API提供了与Python API几乎相同的功能奇偶校验，使您可以像在Python中一样进一步操纵和处理张量。

在最后一行中，我们打印输出的前五个条目。由于在本教程前面的部分中，我们向Python中的模型提供了相同的输入，因此理想情况下，我们
应该看到相同的输出。让我们通过重新编译我们的应用程序并以相同的序列化模型运行它来进行尝试：

```buildoutcfg
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
root@4b5a67132e81:/example-app/build# ./example-app traced_resnet_model.pt
-0.2698 -0.0381  0.4023 -0.3010 -0.0448
[ Variable[CPUFloatType]{1,5} ]
```

作为参考，Python以前的输出为：

```buildoutcfg
tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

看来匹配得很好！

>提示：要将模型移至GPU内存，可以编写model.to（at::kCUDA）;。通过调用tensor.to（at::kCUDA），确保模型的输入也位于CUDA内存中，
这将在CUDA内存中返回新的张量。

#步骤5：获取帮助并探索API
本教程有望使您对PyTorch模型从Python到C ++的路径有一个大致的了解。使用本教程中描述的概念，您应该能够从vanilla, “eager” PyTorch模型，
到Python中的已编译`ScriptModule`，再到磁盘上的序列化文件，以及–结束循环–到可执行脚本： C ++中的模块。

当然，有许多我们没有介绍的概念。例如，您可能会发现自己想要使用以C ++或CUDA实现的自定义运算符扩展`ScriptModule`，并在加载到纯
C ++生产环境中的ScriptModule中执行此自定义运算符。好消息是：这是可能的，并且得到了很好的支持！现在，您可以浏览[此文件夹](https://github.com/pytorch/pytorch/tree/master/test/custom_operator)中的示例，
我们将很快提供一个教程。 目前，以下链接通常可能会有所帮助：

* Torch Script参考：https://pytorch.org/docs/master/jit.html

* PyTorch C ++ API文档：https://pytorch.org/cppdocs/

* PyTorch Python API文档：https://pytorch.org/docs/

与往常一样，如果您遇到任何问题或疑问，可以使用我们的[论坛](https://discuss.pytorch.org/)或[GitHub issues](https://github.com/pytorch/pytorch/issues)进行联系。