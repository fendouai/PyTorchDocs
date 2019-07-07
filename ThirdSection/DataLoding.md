# 数据加载和处理

PyTorch提供了许多工具来简化和希望数据加载，使代码更具可读性。
###1.下载安装包
* scikit-image：用于图像的IO和变换
* pandas：用于更容易地进行csv解析
```
from __future__ import print_function, division
import os
import torch
import pandas as pd              #用于更容易地进行csv解析
from skimage import io, transform    #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
```

### 2.下载数据集
从[此处](https://download.pytorch.org/tutorial/faces.zip)下载数据集，
数据存于“data / faces /”的目录中。这个数据集实际
上是通过对来自imagenet标记为“face”的一些图像，然后应用优秀的dlib
姿势估计来生成的。这次要处理的数据集是面部姿势，并且每个面部都注释了
68个不同的**地标**点。面部的注释如下图：
![](./image/landmarked_face2.png)
#### 2.1 数据集注释
数据集附带一个带注释的csv文件，如下所示：
```
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
```
### 3 读取数据集
快速读取CSV并在（N，2）数组中获取注释，其中N是脸部**地标**的数量。
读取数据代码如下：
```
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
```
####3.1 数据结果
```
Image name: person-7.jpg
Landmarks shape: (68, 2)
First 4 Landmarks: [[32. 65.]
 [33. 76.]
 [34. 86.]
 [34. 97.]]
```
### 4 编写函数
编写一个简单的辅助函数来显示图像及其标记，并用它来显示样本。
```
def show_landmarks(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()
```
函数展示结果如下图所示:
![](./image/sphx_glr_data_loading_tutorial_001.png)
### 5 数据集类
torch.utils.data.Dataset是表示数据集的抽象类，
因此自定义数据集应继承Dataset并覆盖以下方法
* __len__这样就可以len(dataset)返回数据集的大小。
* __getitem__支持索引，以便dataset[i]可以用来获取i样本。
#### 5.1 建立数据集类
为面部地标数据集创建一个数据集类。通过``` __init__ ``` 方法将csv读取到
``` __getitem__``` 。这样做的目的是为了内存更加高效，因为所有图像不会立即存储在内存中，
而是根据需要读取。然后数据集样本的数据类型会变成``` dict``` 。
``` __init__ ```方法如下图所示：
```
class FaceLandmarksDataset(Dataset):
    """面部标记数据集."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：要应用的可选变换在一个样本上
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```
### 6 数据可视化
实例化这个类并遍历数据样本。我们将打印前4个样本的大小并显示他们的地标。
代码如下图所示：
```
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

```
数据结果：
#### 6.1 图形展示结果
![](./image/sphx_glr_data_loading_tutorial_002.png)

#### 6.2 控制台输出结果:
```
0 (324, 215, 3) (68, 2)
1 (500, 333, 3) (68, 2)
2 (250, 258, 3) (68, 2)
3 (434, 290, 3) (68, 2)
```
### 7 数据变换
从**6.2**输出结果中可以看出样本的维度数量不同，
大多数神经网络都期望维度数量的图。
因此，需要编写一些预处理代码。这里创建三个变换：
* ``` Rescale```：缩放图像
* ``` RandomCrop```：随机裁剪图像。这是数据增加。
* ``` ToTensor```：将numpy图像转换为火炬图像（我们需要交换轴）。

通过编写为可调用类而不是简单函数，这样每次调用时都不需
要传递变换的参数。因此需要实现``` __call__```方法。
如果你需要用到```__init__``` 方法，需要使用下面进行数据
类型转换。
```
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```
数据变换实现：
```
class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

    Args:
        output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。         
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```
### 8 撰写变换
假设现在要将图像的短边重新缩放到256，然后从中随机裁剪一个224的正方形。
可以编写```Rescale``` 和```RandomCrop```。可以
调用```torchvision.transforms.Compose```这个类。具体实现如下图：
```
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# 在样品上应用上述每个变换。
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
```
输出效果：
![](./image/sphx_glr_data_loading_tutorial_003.png)

### 9 迭代数据集
将原始数据集经过变换处理之后，然后进行组装成一个新的数据集。
每次采集此样本的时候经过如下步骤：
* 即时从文件中读取图像
* 变换应用于读取的图像
* 由于其中一个变换是随机的，因此在采样时会增加数据

具体实现如下图所示：
```
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
```
输出结果：
```
0 torch.Size([3, 224, 224]) torch.Size([68, 2])
1 torch.Size([3, 224, 224]) torch.Size([68, 2])
2 torch.Size([3, 224, 224]) torch.Size([68, 2])
3 torch.Size([3, 224, 224]) torch.Size([68, 2])
```
通过使用简单的for循环迭代数据，我不能坐到如下几点：
* 批量处理数据
* 洗牌数据
* 使用```multiprocessingworker``` 并行加载数据。

```torch.utils.data.DataLoader```是一个提供所有这些功能的迭代器。
```torchvisionpackage```提供了一些常见的数据集和转换。
```torchvision```中可用的一个更通用的数据集是```ImageFolder```
```
它假定图像按以下方式组织：
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
```
其中'ants'，'bees'等是类标签。类似于PIL.Image操作的通用转换，
如```RandomHorizontalFlip```。因此，数据加载器可以写成如下形式：
```
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```
