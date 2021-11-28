---
title: Fast Feature Engineering in Python; Image Data
subtitle: Make your images more suitable to feed into ML systems
summary: Make your images more suitable to feed into ML systems
authors:
- admin
tags: []
categories: []
date: "2021-09-16T00:00:00Z"
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  caption: 'Image credit: [**Jonathan Borba**](https://unsplash.com/@jonathanborba)'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []

# Set captions for image gallery.
gallery_item:
- album: gallery
  caption: Default
  image: theme-default.png
- album: gallery
  caption: Ocean
  image: theme-ocean.png
- album: gallery
  caption: Forest
  image: theme-forest.png
- album: gallery
  caption: Dark
  image: theme-dark.png
- album: gallery
  caption: Apogee
  image: theme-apogee.png
- album: gallery
  caption: 1950s
  image: theme-1950s.png
- album: gallery
  caption: Coffee theme with Playfair font
  image: theme-coffee-playfair.png
- album: gallery
  caption: Strawberry
  image: theme-strawberry.png
---

> “Finding patterns is easy in any kind of data-rich environment; that’s what mediocre gamblers do. The key is in determining whether the patterns represent noise or signal.”  
> ― **Nate Silver**

This article is part 2 of my “Fast Feature Engineering” series. If you have not read my first article which talks about tabular data, then I request you to check it out here:

[**Fast Feature Engineering in Python: Tabular Data**](https://towardsdatascience.com/fast-feature-engineering-in-python-tabular-data-d050b68bb178 "https://towardsdatascience.com/fast-feature-engineering-in-python-tabular-data-d050b68bb178")[](https://towardsdatascience.com/fast-feature-engineering-in-python-tabular-data-d050b68bb178)

This article will look at some of the best practices to follow when performing image processing as part of our machine learning workflow.

---
### Libraries

```python
import random  
from PIL import Image  
import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
import json  
import albumentations as A  
import torch  
import torchvision.models as models  
import torchvision.transforms as transforms  
import torch.nn as nn  
from tqdm import tqdm_notebook  
from torch.utils.data import DataLoader  
from torchvision.datasets import CIFAR10
```

---
### Resize/Scale Images

Resizing is the most fundamental transformation done by deep learning practitioners in the field. The primary reason for doing this is to ensure that the input received by our deep learning system is **consistent**.

Another reason for resizing is to **reduce the number of parameters** in the model. Smaller dimensions signify a smaller neural network and hence, saves us the time and computation power required to train our model.

#### **_What about the loss of information?_**

Some information is indeed **lost** when you resize down from a larger image. However, depending on your task, you can choose how much information you’re willing to sacrifice for training time and compute resources.

For example, an [**object detection task**](https://en.wikipedia.org/wiki/Object_detection) will require you to maintain the image's aspect ratio since the goal is to detect the exact position of objects.

In contrast, an image classification task may require you to resize all images down to a specified size (224 x 224 is a good rule of thumb).

![](/posts_img/fast_feature_engineering/img_1.jpeg)

After resizing our image looks like this:

![](/posts_img/fast_feature_engineering/img_2.jpeg)

#### _Why perform image scaling?_

Similar to tabular data, scaling images for classification tasks can help our deep learning model's learning rate to converge to the minima better.

Scaling ensures that a particular dimension does not dominate others. I found a fantastic answer on StackExchange regarding this. You can read it [**here**](https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn).

One type of feature scaling is the process of **standardizing** our pixel values. We do this by subtracting the mean of each channel from its pixel value and then divide it via standard deviation.

This is a popular choice of feature engineering when training models for classification tasks.

```python
mean = np.mean(img_resized, axis=(1,2), keepdims=True)
std = np.std(img_resized, axis=(1,2), keepdims=True)
img_std = (img_resized - mean) / std
```

**_Note: Like resizing, one may not want to do image scaling when performing object detection and image generation tasks._**

The example code above demonstrates the process of scaling an image via standardization. There are other forms of scaling such as **centering** and **normalization**.

---
### Augmentations (Classification)

The primary motivation behind augmenting images is due to the appreciable data requirement for computer vision tasks. Often, obtaining enough images for training can prove to be challenging for a multitude of reasons.

Image augmentation enables us to create new training samples by slightly modifying the original ones.

In this example, we will look at how to apply vanilla augmentations for a classification task. We can use the out of the box implementations of the **Albumentations** library to do this:

![](/posts_img/fast_feature_engineering/img_3.jpeg)
![](/posts_img/fast_feature_engineering/img_4.jpeg)
![](/posts_img/fast_feature_engineering/img_5.jpeg)

By applying image augmentations, our deep learning models can generalize better to the task (avoid overfitting), thereby increasing its predictive power on unseen data.

---
### Augmentations (Object Detection)

The Albumentations library can also be used to create augmentations for other tasks such as object detections. Object detection requires us to create bounding boxes around the object of interest.

Working with raw data can prove to be challenging when trying to annotate images with the coordinates for the bounding boxes.

Fortunately, there are many publicly and freely available datasets that we can use to create an augmentation pipeline for object detection. One such dataset is the [**Chess Dataset**](https://public.roboflow.com/object-detection/chess-full).

The dataset contains 606 images of chess pieces on a chessboard.

Along with the images, a JSON file is provided that contains all the information pertaining to the bounding boxes for each chess piece in a single image.

By writing a simple function, we can visualize the data after the augmentation is applied:

```python
with open("_annotations.coco.json") as f:
    json_file = json.load(f)
    
x_min, y_min, w, h = json_file['annotations'][0]['bbox']
x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

def visualize_bbox(img, bbox, class_name, color=(0, 255, 0), thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img
  
bbox_img = visualize_bbox(np.array(img), 
                          json_file['annotations'][0]['bbox'], 
                          class_name=json_file['categories'][0]['name'])

Image.fromarray(bbox_img)
```

![](/Users/Banner/Downloads/medium-export-aa5b5fa1b4f15ba326f851375de5c386499a5652f183eac85ab56b6ca8924b20/posts/md_1638090489769/img/1__MFakz3EYf73afrl__aT3S2A.jpeg)
![](/posts_img/fast_feature_engineering/img_6.jpeg)

Now, let’s try to create an augmentation pipeline using Albumentations.

The JSON file that contains the annotation information has the following keys:

`dict_keys([‘info’, ‘licenses’, ‘categories’, ‘images’, ‘annotations’])`

`images` contains information about the image file whereas `annotations` contains information about the bounding boxes for each object in an image.

Finally, `categories` contains keys that map to the type of chess pieces in the image.

```
image_list = json_file.get('images')  
anno_list = json_file.get('annotations')  
cat_list = json_file.get('categories')
```

`image_list` :

```
[{'id': 0,  
  'license': 1,  
  'file_name': 'IMG_0317_JPG.rf.00207d2fe8c0a0f20715333d49d22b4f.jpg',  
  'height': 416,  
  'width': 416,  
  'date_captured': '2021-02-23T17:32:58+00:00'},  
 {'id': 1,  
  'license': 1,  
  'file_name': '5a8433ec79c881f84ef19a07dc73665d_jpg.rf.00544a8110f323e0d7721b3acf2a9e1e.jpg',  
  'height': 416,  
  'width': 416,  
  'date_captured': '2021-02-23T17:32:58+00:00'},  
 {'id': 2,  
  'license': 1,  
  'file_name': '675619f2c8078824cfd182cec2eeba95_jpg.rf.0130e3c26b1bf275bf240894ba73ed7c.jpg',  
  'height': 416,  
  'width': 416,  
  'date_captured': '2021-02-23T17:32:58+00:00'},  
.  
.  
.  
.
```

`anno_list` :

```
[{'id': 0,  
  'image_id': 0,  
  'category_id': 7,  
  'bbox': [220, 14, 18, 46.023746508293286],  
  'area': 828.4274371492792,  
  'segmentation':],  
  'iscrowd': 0},  
 {'id': 1,  
  'image_id': 1,  
  'category_id': 8,  
  'bbox': [187, 103, 22.686527154676014, 59.127992255841036],  
  'area': 1341.4088019136107,  
  'segmentation': [],  
  'iscrowd': 0},  
 {'id': 2,  
  'image_id': 2,  
  'category_id': 10,  
  'bbox': [203, 24, 24.26037020843023, 60.5],  
  'area': 1467.752397610029,  
  'segmentation': [],  
  'iscrowd': 0},  
.  
.  
.  
.
```

`cat_list` :

```
[{'id': 0, 'name': 'pieces', 'supercategory': 'none'},  
 {'id': 1, 'name': 'bishop', 'supercategory': 'pieces'},  
 {'id': 2, 'name': 'black-bishop', 'supercategory': 'pieces'},  
 {'id': 3, 'name': 'black-king', 'supercategory': 'pieces'},  
 {'id': 4, 'name': 'black-knight', 'supercategory': 'pieces'},  
 {'id': 5, 'name': 'black-pawn', 'supercategory': 'pieces'},  
 {'id': 6, 'name': 'black-queen', 'supercategory': 'pieces'},  
 {'id': 7, 'name': 'black-rook', 'supercategory': 'pieces'},  
 {'id': 8, 'name': 'white-bishop', 'supercategory': 'pieces'},  
 {'id': 9, 'name': 'white-king', 'supercategory': 'pieces'},  
 {'id': 10, 'name': 'white-knight', 'supercategory': 'pieces'},  
 {'id': 11, 'name': 'white-pawn', 'supercategory': 'pieces'},  
 {'id': 12, 'name': 'white-queen', 'supercategory': 'pieces'},  
 {'id': 13, 'name': 'white-rook', 'supercategory': 'pieces'}]
```

We have to alter the structure of these lists to create an efficient pipeline:

```python
new_anno_dict = {}
new_cat_dict = {}

for item in cat_list:
    new_cat_dict[item['id']] = item['name']
    

for item in anno_list:
    img_id = item.get('image_id')
    if img_id not in new_anno_dict:
        temp_list = []
        temp_list.append(item)
        new_anno_dict[img_id] = temp_list
    else:
        new_anno_dict.get(img_id).append(item)
```

Now, let’s create a simple augmentation pipeline that flips our image horizontally and adds a parameter for bounding boxes:

```python
transform = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)
```

Lastly, we will create a dataset similar to the [**Dataset class**](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset) offered by Pytorch. To do this, we need to define a class that implements the methods `__len__` and `__getitem__`.

```python
class ImageDataset:
    def __init__(self, path, img_list, anno_dict, cat_dict, albumentations=None):
        self.path = path
        self.img_list = img_list
        self.anno_dict = anno_dict
        self.cat_dict = cat_dict
        self.albumentations = albumentations
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # Each image may have multiple objects thereby multiple bboxes
        bboxes = [item['bbox'] for item in self.anno_dict[int(idx)]]
        cat_ids = [item['category_id'] for item in self.anno_dict[int(idx)]]
        categories = [self.cat_dict[idx] for idx in cat_ids]
        image = self.img_list[idx]
        img = Image.open(f"{self.path}{image.get('file_name')}")
        img = img.convert("RGB")
        img = np.array(img)
        if self.albumentations is not None:
            augmented = self.albumentations(image=img, bboxes=bboxes, category_ids=cat_ids)
            img = augmented["image"]
        return {
            "image": img,
            "bboxes": augmented["bboxes"],
            "category_ids": augmented["category_ids"],
            "category": categories
        }

 # path is the path to the json_file and images
dataset = ImageDataset(path, image_list, new_anno_dict, new_cat_dict, transform)
```

Here are some of the results while iterating on the custom dataset:

![](/posts_img/fast_feature_engineering/img_7.jpeg)
![](/posts_img/fast_feature_engineering/img_8.jpeg)
![](/posts_img/fast_feature_engineering/img_9.jpeg)
![](/posts_img/fast_feature_engineering/img_10.jpeg)
![](/posts_img/fast_feature_engineering/img_11.jpeg)

Thus, we can now easily pass this custom dataset to a data loader to train our model.

---
### Feature Extraction

You may have heard of pre-trained models being used to train image classifiers and for other supervised learning tasks.

But, did you know that you can also use pre-trained models for feature extraction of images?

In short feature extraction is a form of dimensionality reduction where a large number of pixels are reduced to a more efficient representation.

This is primarily useful for unsupervised machine learning tasks such as reverse image search.

Let’s try to extract features from images using Pytorch’s pre-trained models. To do this, we must first define our feature extractor class:

```python
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = nn.Sequential(*model.children())[:-1]
    def forward(self, x):
        return self.model(x)
```

Note that in line 4, a new model is created with all of the layers of the original save for the last one. You will recall that the last layer in a neural network is a dense layer used for prediction outputs.

However, since we are only interested in extracting features, we do not require this last layer. Hence, it is excluded.

We then utilize torchvision’s pre-trained `resnet34` model by passing it to the `ResnetFeatureExtractor` constructor.

Let’s use the famous [**CIFAR10 dataset**](https://paperswithcode.com/dataset/cifar-10) (50000 images), and loop over it to extract the features.

![](/posts_img/fast_feature_engineering/img_12.png)

```python
cifar_dataset = CIFAR10("./", transform=transforms.ToTensor(), download=True)
cifar_dataloader = DataLoader(cifar_dataset, batch_size=1, shuffle=True)

feature_extractor.eval()
feature_list = []

for _, data in enumerate(tqdm_notebook(cifar_dataloader)):
    inputs, labels = data
    with torch.no_grad():
        extracted_features = feature_extractor(inputs)
    extracted_features = torch.flatten(extracted_features)
    feature_list.append(extracted_features)
```


We now have a list of 50000 image feature vectors with each feature vector of size 512 (output size of the penultimate layer of the original resnet model).

```
print(f"Number of feature vectors: {len(feature_list)}") #50000  
print(f"Number of feature vectors: {len(feature_list[0])}") #512
```

Thus, this list of feature vectors can now be used by statistical learning models such as KNN to search for similar images.

If you have reached this far then thank you very much for reading this article! I hope you have a fantastic day ahead! 😄

**👉** [**Code used in the article**](https://github.com/Sayar1106/TowardsDataSciencecodefiles/tree/master/fast_feature_engineering)

Until next time! ✋

---
### References:

*   [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
*   [https://www.practicaldeeplearning.ai/](https://www.practicaldeeplearning.ai/) 
