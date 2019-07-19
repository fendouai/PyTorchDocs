# 微调基于 torchvision 0.3的目标检测模型
在本教程中，我们将微调在 Penn-Fudan 数据库中对行人检测和分割的已预先训练的 Mask R-CNN 模型。它包含170个图像和345个行人实例，我们
将用它来说明如何在 torchvision 中使用新功能，以便在自定义数据集上训练实例分割模型。

### 1.定义数据集
对于训练对象检测的引用脚本，实例分割和人员关键点检测要求能够轻松支持添加新的自定义数据。数据集应该从标准的类`torch.utils.data.Dataset`
继承而来，并实现`_len`和`_getitem_`

我们要求的唯一特性是数据集的`__getitem__`应该返回：
* 图像：PIL图像大小(H,W)
* 目标：包含以下字段的字典<br/>
        <1> `boxes(FloatTensor[N,4])`：N边框（bounding boxes）坐标的格式[x0,x1,y0,y1]，取值范围是0到W,0到H。<br/>
        <2> `labels(Int64Tensor[N])`：每个边框的标签。<br/>
        <3> `image_id(Int64Tensor[1])`：图像识别器，它应该在数据集中的所有图像中是唯一的，并在评估期间使用。<br/>
        <4> `area(Tensor[N])`：边框的面积，在使用COCO指标进行评估时使用此项来分隔小、中和大框之间的度量标准得分。<br/>
        <5> `iscrowed(UInt8Tensor[N,H,W])`：在评估期间属性设置为`iscrowed=True`的实例会被忽略。<br/>
        <6> (可选)`masks(UInt8Tesor[N,H,W])`：每个对象的分段掩码。<br/>
        <7> (可选)`keypoints (FloatTensor[N, K, 3]`：对于N个对象中的每一个，它包含[x，y，visibility]格式的K个关键点，用
        于定义对象。`visibility = 0`表示关键点不可见。请注意，对于数据扩充，翻转关键点的概念取决于数据表示，您应该调整
        reference/detection/transforms.py 以用于新的关键点表示。<br/>

如果你的模型返回上述方法，它们将使其适用于培训和评估，并将使用 pycocotools 的评估脚本。

此外，如果要在训练期间使用宽高比分组（以便每个批次仅包含具有相似宽高比的图像），则建议还实现`get_height_and_width`方法，
该方法返回图像的高度和宽度。如果未提供此方法，我们将通过`__getitem__`查询数据集的所有元素，这会将图像加载到内存中，但比提供自定义方法时要慢。        

### 2.为 PennFudan 编写自定义数据集
#### 2.1 下载数据集
[下载并解压缩zip文件](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)后，我们有以下文件夹结构：
```buildoutcfg
PennFudanPed/
  PedMasks/
    FudanPed00001_mask.png
    FudanPed00002_mask.png
    FudanPed00003_mask.png
    FudanPed00004_mask.png
    ...
  PNGImages/
    FudanPed00001.png
    FudanPed00002.png
    FudanPed00003.png
    FudanPed00004.png
```
下面是一个图像以及其分割蒙版的例子：
![](image/01.png)
![](image/02.png)

因此每个图像具有相应的分割蒙版，其中每个颜色对应于不同的实例。让我们为这个数据集写一个`torch.utils.data.Dataset`类。

#### 2.2 为数据集编写类
```buildoutcfg
import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 下载所有图像文件，为其排序
        # 确保它们对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 请注意我们还没有将蒙版转换为RGB,
        # 因为每种颜色对应一个不同的实例
        # 0是背景
        mask = Image.open(mask_path)
        # 将PIL图像转换为numpy数组
        mask = np.array(mask)
        # 实例被编码为不同的颜色
        obj_ids = np.unique(mask)
        # 第一个id是背景，所以删除它
        obj_ids = obj_ids[1:]

        # 将颜色编码的蒙版分成一组
        # 二进制格式
        masks = mask == obj_ids[:, None, None]

        # 获取每个蒙版的边界框坐标
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 将所有转换为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 这里仅有一个类
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

### 3.定义模型
现在我们需要定义一个可以上述数据集执行预测的模型。在本教程中，我们将使用 [Mask R-CNN](https://arxiv.org/abs/1703.06870)，
它基于 [Faster R-CNN](https://arxiv.org/abs/1506.01497)。Faster R-CNN 是一种模型，可以预测图像中潜在对象的边界框和类别得分。 
![](image/03.png)

Mask R-CNN 在 Faster R-CNN 中添加了一个额外的分支，它还预测每个实例的分割蒙版。

![](image/04.png)

有两种常见情况可能需要修改`torchvision modelzoo`中的一个可用模型。第一个是我们想要从预先训练的模型开始，然后微调最后一层。
另一种是当我们想要用不同的模型替换模型的主干时（例如，用于更快的预测）。

下面是对这两种情况的处理。
* 1 微调已经预训练的模型
让我们假设你想从一个在COCO上已预先训练过的模型开始，并希望为你的特定类进行微调。这是一种可行的方法：
```buildoutcfg
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 在COCO上加载经过预训练的预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# 将分类器替换为具有用户定义的 num_classes的新分类器
num_classes = 2  # 1 class (person) + background
# 获取分类器的输入参数的数量
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用新的头部替换预先训练好的头部
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

* 2 修改模型以添加不同的主干
```buildoutcfg
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 加载预先训练的模型进行分类和返回
# 只有功能
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN需要知道骨干网中的输出通道数量。对于mobilenet_v2，它是1280，所以我们需要在这里添加它
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
```
