import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2 as cv
from sklearn.model_selection import train_test_split
from PIL import Image


# 返回每一张图片的路径,并且与文本一一对应，并且还需要返回标签信息
def LoadData():
    cur_path = os.path.dirname(__file__)  # 'E:/github文件/my-project'
    img_path = "./data/img"
    ImgPath = []
    n = len(os.listdir(os.path.join(cur_path, img_path)))  # 图像总数　
    for i in range(1, n + 1):
        ImgPath.append(cur_path + "/" +
                       os.path.join(img_path, str(i)) + ".jpg")  # 图像路径 'E:/github文件/my-project/data/img\\3798.jpg'
    return ImgPath


def augmentImage(imgPath):
    img = mpimg.imread(imgPath)
    ## PAN
    if np.random.rand() <= 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)
    ## ZOOM
    if np.random.rand() <= 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    # BRIGHTNESS
    if np.random.rand() <= 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    # FLIP
    if np.random.rand() <= 0.5:
        img = cv.flip(img, 1)
    return img


def preProcessing(img):
    img = img[:, 20:400, :]  # (高,宽,通道)
    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = img / 255
    return img


def batchGen(batch_size, image_path, text, label, original=True, multiple=4):
    """
    image_path:List() 待增强图像的路径
    text:经过word2vec处理过后的文本数据
    original:bool 是否增加原生态图像数量
    multiple:int 增加的数量
    """
    while True:
        imageBatch, textBatch, labelBatch = [], [], []
        for i in range(batch_size):
            ImglittleBatch, txtlittleBatch, labellittleBatch = [], [], []
            index = np.random.randint(0, len(image_path))
            if original:
                image = augmentImage(image_path[index])
                image = preProcessing(image)
                imageBatch.append(image)
                textBatch.append(text[index])
                labelBatch.append(label[index])
            else:
                for j in range(multiple):
                    image = augmentImage(image_path[index])
                    image = preProcessing(image)
                    ImglittleBatch.append(image)
                    txtlittleBatch.append(text[index])
                    labellittleBatch.append(label[index])
                imageBatch += ImglittleBatch
                textBatch += txtlittleBatch
                labelBatch += labellittleBatch

        # 对列表中的元素加入随机性，打乱，固定打乱顺序
        state = np.random.get_state()
        np.random.shuffle(imageBatch)
        np.random.set_state(state)
        np.random.shuffle(textBatch)
        np.random.set_state(state)
        np.random.shuffle(labelBatch)
        yield (np.asarray(imageBatch), np.asarray(textBatch)), np.asarray(labelBatch)


def Img_Txt(imgTest, txtTest, start, end):
    imageBatch, textBatch = [], []
    for i in range(start, end):
        image = mpimg.imread(imgTest[i]) / 255
        image = cv.resize(image, (224, 224))
        text = txtTest[i]
        imageBatch.append(image)
        textBatch.append(text)
    return np.asarray(imageBatch), np.asarray(textBatch)

