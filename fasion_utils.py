from fasion_dataset import *
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
from train_model import get_data
from train_model import text_w2model
from sklearn.model_selection import train_test_split
from load_data import generate_method
import matplotlib.image as mpimg

cpu_count = multiprocessing.cpu_count()


def initialization():
    img_path, txt_data, label, all_words = Fashion_200k(path, label_path)
    model = Word2Vec(size=500,  # 建立一个空的模型对象，设置词向量的维度为100
                     min_count=5,  # 频数
                     window=3,  # 窗口大小
                     workers=cpu_count,
                     iter=5)
    w2indx, w2vec, text_data, _ = text_w2model(model, all_words, max_len=15)
    n_symbols, embedding_weights = get_data(w2indx, w2vec, vocab_dim=500)
    return img_path, text_data, label, n_symbols, embedding_weights


def Processing(imgPath):
    img = mpimg.imread(imgPath)
    img = cv.resize(img, (224, 224))
    img = img / 255
    return img


def BatchGen(batch_size, image_path, text, label):
    while True:
        imageBatch, textBatch, labelBatch = [], [], []
        for _ in range(batch_size):
            index = np.random.randint(0, len(image_path))
            image = Processing(image_path[index])
            imageBatch.append(image)
            textBatch.append(text[index])
            labelBatch.append(label[index])

        # 对列表中的元素加入随机性，打乱，固定打乱顺序
        state = np.random.get_state()
        np.random.shuffle(imageBatch)
        np.random.set_state(state)
        np.random.shuffle(textBatch)
        np.random.set_state(state)
        np.random.shuffle(labelBatch)
        yield (np.asarray(imageBatch), np.asarray(textBatch)), np.asarray(labelBatch)
