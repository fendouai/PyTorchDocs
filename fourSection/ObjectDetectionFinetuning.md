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
下面是一个图像以及其分割掩膜的例子：
![](image/01.png)
![](image/02.png)

因此每个图像具有相应的分割掩膜，其中每个颜色对应于不同的实例。让我们为这个数据集写一个`torch.utils.data.Dataset`类。

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
        # 请注意我们还没有将mask转换为RGB,
        # 因为每种颜色对应一个不同的实例
        # 0是背景
        mask = Image.open(mask_path)
        # 将PIL图像转换为numpy数组
        mask = np.array(mask)
        # 实例被编码为不同的颜色
        obj_ids = np.unique(mask)
        # 第一个id是背景，所以删除它
        obj_ids = obj_ids[1:]

        # 将颜色编码的mask分成一组
        # 二进制格式
        masks = mask == obj_ids[:, None, None]

        # 获取每个mask的边界框坐标
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

# 我们让RPN在每个空间位置生成5 x 3个锚点
# 具有5种不同的大小和3种不同的宽高比。 
# 我们有一个元组[元组[int]]
# 因为每个特征映射可能具有不同的大小和宽高比
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# 定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小。 
# 如果您的主干返回Tensor，则featmap_names应为[0]。 
# 更一般地，主干应该返回OrderedDict [Tensor]
# 并且在featmap_names中，您可以选择要使用的功能映射。
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# 将这些pieces放在FasterRCNN模型中
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
```
#### 3.1 PennFudan 数据集的实例分割模型
在我们的例子中，我们希望从预先训练的模型中进行微调，因为我们的数据集非常小，所以我们将遵循上述第一种情况。

这里我们还要计算实例分割掩膜，因此我们将使用 Mask R-CNN：
```buildoutcfg
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # 加载在COCO上预训练的预训练的实例分割模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 现在获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 并用新的掩膜预测器替换掩膜预测器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
```

就是这样，这将使模型准备好在您的自定义数据集上进行训练和评估。

### 4.整合
在`references/detection/`中，我们有许多辅助函数来简化训练和评估检测模型。在这里，我们将使用
`references/detection/engine.py`，`references/detection/utils.py`和`references/detection/transforms.py`。 
只需将它们复制到您的文件夹并在此处使用它们。

#### 4.1 为数据扩充/转换编写辅助函数：
```buildoutcfg
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```
#### 4.2 编写执行训练和验证的主要功能
```buildoutcfg
from engine import train_one_epoch, evaluate
import utils


def main():
    # 在GPU上训练，若无GPU，可选择在CPU上训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 我们的数据集只有两个类 - 背景和人
    num_classes = 2
    # 使用我们的数据集和定义的转换
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # 在训练和测试集中拆分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # 定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # 使用我们的辅助函数获取模型
    model = get_model_instance_segmentation(num_classes)

    # 将我们的模型迁移到合适的设备
    model.to(device)

    # 构造一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 和学习率调度程序
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # 训练10个epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # 训练一个epoch，每10次迭代打印一次
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # 更新学习速率
        lr_scheduler.step()
        # 在测试集上评价
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
```
在第一个epoch训练后可以得到下面的结果：
```buildoutcfg
Epoch: [0]  [ 0/60]  eta: 0:01:18  lr: 0.000090  loss: 2.5213 (2.5213)  loss_classifier: 0.8025 (0.8025)  loss_box_reg: 0.2634 (0.2634)  loss_mask: 1.4265 (1.4265)  loss_objectness: 0.0190 (0.0190)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.3121  data: 0.3024  max mem: 3485
Epoch: [0]  [10/60]  eta: 0:00:20  lr: 0.000936  loss: 1.3007 (1.5313)  loss_classifier: 0.3979 (0.4719)  loss_box_reg: 0.2454 (0.2272)  loss_mask: 0.6089 (0.7953)  loss_objectness: 0.0197 (0.0228)  loss_rpn_box_reg: 0.0121 (0.0141)  time: 0.4198  data: 0.0298  max mem: 5081
Epoch: [0]  [20/60]  eta: 0:00:15  lr: 0.001783  loss: 0.7567 (1.1056)  loss_classifier: 0.2221 (0.3319)  loss_box_reg: 0.2002 (0.2106)  loss_mask: 0.2904 (0.5332)  loss_objectness: 0.0146 (0.0176)  loss_rpn_box_reg: 0.0094 (0.0123)  time: 0.3293  data: 0.0035  max mem: 5081
Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 (0.2022)  loss_box_reg: 0.1207 (0.1736)  loss_mask: 0.1705 (0.3585)  loss_objectness: 0.0018 (0.0113)  loss_rpn_box_reg: 0.0075 (0.0112)  time: 0.3407  data: 0.0044  max mem: 5081
Epoch: [0]  [50/60]  eta: 0:00:03  lr: 0.004323  loss: 0.3237 (0.6703)  loss_classifier: 0.0474 (0.1731)  loss_box_reg: 0.1109 (0.1561)  loss_mask: 0.1658 (0.3201)  loss_objectness: 0.0015 (0.0093)  loss_rpn_box_reg: 0.0093 (0.0116)  time: 0.3379  data: 0.0043  max mem: 5081
Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2540 (0.6082)  loss_classifier: 0.0309 (0.1526)  loss_box_reg: 0.0463 (0.1405)  loss_mask: 0.1568 (0.2945)  loss_objectness: 0.0012 (0.0083)  loss_rpn_box_reg: 0.0093 (0.0123)  time: 0.3489  data: 0.0042  max mem: 5081
Epoch: [0] Total time: 0:00:21 (0.3570 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:19  model_time: 0.2152 (0.2152)  evaluator_time: 0.0133 (0.0133)  time: 0.4000  data: 0.1701  max mem: 5081
Test:  [49/50]  eta: 0:00:00  model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)  time: 0.0735  data: 0.0022  max mem: 5081
Test: Total time: 0:00:04 (0.0828 s / it)
Averaged stats: model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.780
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```
因此，在一个epoch训练之后，我们获得了COCO-style mAP为60.6，并且mask mAP为70.4。

经过训练10个epoch后，我得到了以下指标：
```buildoutcfg
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.935
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
```
但预测结果如何呢？让我们在数据集中拍摄一张图像并进行验证。
![](image/05.png)

训练的模型预测了此图像中的9个人物，让我们看看其中的几个，由下图可以看到预测效果很好。
![](image/06.png)

### 5.总结
在本教程中，您学习了如何在自定义数据集上为实例分段模型创建自己的训练管道。为此，您编写了一个`torch.utils.data.Dataset`类，
它返回图像以及地面实况框和分割掩码。您还利用了在COCO train2017上预训练的Mask R-CNN模型，以便对此新数据集执行传输学习。

有关包含multi-machine / multi-gpu training的更完整示例，请检查 torchvision 存储库中的`references/detection/train.py`。

您可以在[此处](https://pytorch.org/tutorials/_static/tv-training-code.py)下载本教程的完整源文件。
