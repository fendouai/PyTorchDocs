# 通过带Flask的REST API在Python中部署PyTorch

在本教程中，我们将使用Flask来部署PyTorch模型，并用讲解用于模型推断的 REST API。特别是，我们将部署一个预训练的DenseNet 121模
型来检测图像。

> 备注：
    可在[GitHub](https://github.com/avinassh/pytorch-flask-api)上获取本文用到的完整代码
    
这是在生产中部署PyTorch模型的系列教程中的第一篇。到目前为止，以这种方式使用Flask是开始为PyTorch模型提供服务的最简单方法，
但不适用于具有高性能要求的用例。因此：
 * 如果您已经熟悉TorchScript，则可以直接进入我们的[Loading a TorchScript Model in C++](https://github.com/fendouai/PyTorchDocs/blob/master/EigthSection/torchScript_in_C%2B%2B.md)教程。
 * 如果您首先需要复习TorchScript，请查看我们的[Intro a TorchScript](https://github.com/fendouai/PyTorchDocs/blob/master/EigthSection/torchScript.md)教程。
 
 ## 1.定义API
 我们将首先定义API端点、请求和响应类型。我们的API端点将位于`/ predict`，它接受带有包含图像的`file`参数的HTTP POST请求。响应
 将是包含预测的JSON响应：
 ```buildoutcfg
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

## 2.依赖（包）
运行下面的命令来下载我们需要的依赖：
```buildoutcfg
$ pip install Flask==1.0.3 torchvision-0.3.0
```
## 3.简单的Web服务器
以下是一个简单的Web服务器，摘自Flask文档
```buildoutcfg
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'
```
将以上代码段保存在名为`app.py`的文件中，您现在可以通过输入以下内容来运行Flask开发服务器：
```buildoutcfg
$ FLASK_ENV=development FLASK_APP=app.py flask run
```
当您在web浏览器中访问`http://localhost:5000/`时，您会收到文本`Hello World`的问候！

我们将对以上代码片段进行一些更改，以使其适合我们的API定义。首先，我们将重命名`predict`方法。我们将端点路径更新为`/predict`。 
由于图像文件将通过HTTP POST请求发送，因此我们将对其进行更新，使其也仅接受POST请求：
```buildoutcfg
@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'
```
我们还将更改响应类型，以使其返回包含ImageNet类的id和name的JSON响应。更新后的`app.py`文件现在为：
```buildoutcfg
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
```

## 4.推理
在下一部分中，我们将重点介绍编写推理代码。这将涉及两部分，第一部分是准备图像，以便可以将其馈送到DenseNet；第二部分，我们将编
写代码以从模型中获取实际的预测。

### 4.1 准备图像
DenseNet模型要求图像为尺寸为224 x 224的 3 通道RGB图像。我们还将使用所需的均值和标准偏差值对图像张量进行归一化。你可以点击
[这里](https://pytorch.org/docs/stable/torchvision/models.html)来了解更多关于它的内容。

我们将使用来自`torchvision`库的`transforms`来建立转换管道，该转换管道可根据需要转换图像。您可以在[此处](https://pytorch.org/docs/stable/torchvision/transforms.html)
阅读有关转换的更多信息。
```buildoutcfg
import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
```
上面的方法以字节为单位获取图像数据，应用一系列变换并返回张量。要测试上述方法，请以字节模式读取图像文件（首先将../_static/img/
sample_file.jpeg替换为计算机上文件的实际路径），然后查看是否获得了张量：
```buildoutcfg
with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)
```

* 输出结果：
```buildoutcfg
tensor([[[[ 0.4508,  0.4166,  0.3994,  ..., -1.3473, -1.3302, -1.3473],
          [ 0.5364,  0.4851,  0.4508,  ..., -1.2959, -1.3130, -1.3302],
          [ 0.7077,  0.6392,  0.6049,  ..., -1.2959, -1.3302, -1.3644],
          ...,
          [ 1.3755,  1.3927,  1.4098,  ...,  1.1700,  1.3584,  1.6667],
          [ 1.8893,  1.7694,  1.4440,  ...,  1.2899,  1.4783,  1.5468],
          [ 1.6324,  1.8379,  1.8379,  ...,  1.4783,  1.7352,  1.4612]],

         [[ 0.5728,  0.5378,  0.5203,  ..., -1.3704, -1.3529, -1.3529],
          [ 0.6604,  0.6078,  0.5728,  ..., -1.3004, -1.3179, -1.3354],
          [ 0.8529,  0.7654,  0.7304,  ..., -1.3004, -1.3354, -1.3704],
          ...,
          [ 1.4657,  1.4657,  1.4832,  ...,  1.3256,  1.5357,  1.8508],
          [ 2.0084,  1.8683,  1.5182,  ...,  1.4657,  1.6583,  1.7283],
          [ 1.7458,  1.9384,  1.9209,  ...,  1.6583,  1.9209,  1.6408]],

         [[ 0.7228,  0.6879,  0.6531,  ..., -1.6476, -1.6302, -1.6476],
          [ 0.8099,  0.7576,  0.7228,  ..., -1.6476, -1.6476, -1.6650],
          [ 1.0017,  0.9145,  0.8797,  ..., -1.6476, -1.6650, -1.6999],
          ...,
          [ 1.6291,  1.6291,  1.6465,  ...,  1.6291,  1.8208,  2.1346],
          [ 2.1868,  2.0300,  1.6814,  ...,  1.7685,  1.9428,  2.0125],
          [ 1.9254,  2.0997,  2.0823,  ...,  1.9428,  2.2043,  1.9080]]]])
```

### 4.2 预测
现在将使用预训练的DenseNet 121模型来预测图像的类别。我们将使用`torchvision`库中的一个库，加载模型并进行推断。在此示例中，我们
将使用预训练的模型，但您可以对自己的模型使用相同的方法。在这个[教程](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
中了解有关加载模型的更多信息。
```buildoutcfg
from torchvision import models

# 确保使用`pretrained`作为`True`来使用预训练的权重：
model = models.densenet121(pretrained=True)
# 由于我们仅将模型用于推理，因此请切换到“eval”模式：
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat
```
张量`y_hat`将包含预测的类的id的索引。但是，我们需要一个易于阅读的类名。为此，我们需要一个类id来命名映射。将[该文件](https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json)
下载为`imagenet_class_index.json`并记住它的保存位置（或者，如果您按照本教程中的确切步骤操作，请将其保存在`tutorials/_static`中）。
此文件包含ImageNet类的id到ImageNet类的name的映射。我们将加载此JSON文件并获取预测索引的类的name。
```buildoutcfg
import json

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
```
在使用字典`imagenet_class_index`之前，首先我们将张量值转换为字符串值，因为字典`imagenet_class_index`中的keys是字符串。我们将
测试上述方法：
```buildoutcfg
with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
```
* 输出结果：
```buildoutcfg
['n02124075', 'Egyptian_cat']
```
你会得到这样的一个响应：
```buildoutcfg
['n02124075', 'Egyptian_cat']
```
数组中的第一项是ImageNet类的id，第二项是人类可读的name。

> 注意：您是否注意到模型变量不是`get_prediction`方法的一部分？或者为什么模型是全局变量？就内存和计算而言，加载模型可能是
一项昂贵的操作。如果将模型加载到`get_prediction`方法中，则每次调用该方法时都会不必要地加载该模型。由于我们正在构建Web服务
器，因此每秒可能有成千上万的请求，因此我们不应该浪费时间为每个推断重复加载模型。因此，我们仅将模型加载到内存中一次。在生
产系统中，必须有效利用计算以能够大规模处理请求，因此通常应在处理请求之前加载模型。

## 5.将模型集成到我们的API服务器中
在最后一部分中，我们将模型添加到Flask API服务器中。由于我们的API服务器应该获取图像文件，因此我们将更新`predict`方法以从请求中
读取文件：
```buildoutcfg
from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 从请求中获得文件
        file = request.files['file']
        # 转化为字节
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
```
`app.py`文件现已完成。以下是完整版本；将路径替换为保存文件的路径，它的运行应是如下：
```buildoutcfg
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
```
让我们测试一下我们的web服务器，运行：
```buildoutcfg
$ FLASK_ENV=development FLASK_APP=app.py flask run
```
我们可以使用[requests](https://pypi.org/project/requests/)库来发送一个POST请求到我们的app：
```buildoutcfg
import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})
```
打印`resp.json()`会显示下面的结果：
```buildoutcfg
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

## 6.下一步工作
我们编写的服务器非常琐碎，可能无法完成生产应用程序所需的一切。因此，您可以采取一些措施来改善它：

* 端点`/predict`假定请求中总会有一个图像文件。这可能不适用于所有请求。我们的用户可能发送带有其他参数的图像，或者根本不发送任何图像。
* 用户也可以发送非图像类型的文件。由于我们没有处理错误，因此这将破坏我们的服务器。添加显式的错误处理路径来引发异常，这将使我们
能够更好地处理错误的输入
* 即使模型可以识别大量类别的图像，也可能无法识别所有图像。增强实现以处理模型无法识别图像中的任何情况的情况。
* 我们在开发模式下运行Flask服务器，该服务器不适合在生产中进行部署。您可以查看[教程](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/)
以在生产环境中部署Flask服务器。
* 您还可以通过创建一个带有表单的页面来添加UI，该表单可以拍摄图像并显示预测。查看类似[项目](https://pytorch-imagenet.herokuapp.com/)的演示及其[源代码](https://github.com/avinassh/pytorch-flask-api-heroku)。
* 在本教程中，我们仅展示了如何构建可以一次返回单个图像预测的服务。我们可以修改服务以能够一次返回多个图像的预测。此外，[service-streamer](https://github.com/ShannonAI/service-streamer)
库自动将对服务的请求排队，并将它们采样到可用于模型的min-batches中。您可以查看[此教程](https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer)。
* 最后，我们鼓励您在页面顶部查看链接到的有关部署PyTorch模型的其他教程。

