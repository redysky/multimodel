# 将数据集处理成npy格式，此为网络输入的一种形式，优点是读取方便，缺点是一次性读取造成内存溢出，
# 解决方法是使用data_generate,批次读入内存
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_npy(use_power_dataset=True):
    cur_path = os.path.dirname(__file__)  # 获取当前文件路径
    parent_path = os.path.dirname(cur_path)  # 获取当前文件夹父目录
    if use_power_dataset:
        # 指定图片和标签的路径
        train_path = './data/power_img/'
        train_txt = './data/labels.txt'
        # 指定转换后的存储路径
        x_train_save_path = './data/power_img_data_npy/x_train3.npy'
        y_train_save_path = './data/power_img_data_npy/y_train3.npy'
        # 数据集总图像数
        total_img = len(os.listdir(train_path))
        # 数据集每类图像数
        class_img = 400
    else:
        # 指定图片和标签的路径
        train_path = './data/img/'
        train_txt = './data/labels.txt'
        # 指定转换后的存储路径
        x_train_save_path = './data/original_img_data_npy/x_train3.npy'
        y_train_save_path = './data/original_img_data_npy/y_train3.npy'
        # 数据集总图像数
        total_img = len(os.listdir(train_path))
        # 数据集每类图像数
        class_img = 100

    # print(total_img)
    # 打标签,该标签以0起始

    def get_label(train_txt, total_img, class_img):
        j = 0
        with open(train_txt, "w") as f:
            for i in range(total_img):
                if i % class_img == 0:
                    j += 1
                text = f.write(str(i + 1) + ".jpg" + " " + str(j - 1) + "\n")
    if not os.path.exists(train_txt):
        get_label(train_txt, total_img, class_img)

    def generated(path, txt):
        f = open(txt, 'r')
        contents = f.readlines()
        f.close()
        x, y_ = [], []
        for content in contents:
            value = content.split()
            img_path = path + value[0]
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.BILINEAR)
            img = np.array(img)
            x.append(img)
            y_.append(value[1])
            print('loading:' + content)
        x = np.array(x)
        # print(x)
        y_ = np.array(y_)
        # print(y_.shape)
        y_ = y_.astype(np.int64)
        return x, y_

    if os.path.exists(x_train_save_path) and os.path.exists(y_train_save_path):
        print('-------Load Datasets---------')
        x_train_save = np.load(x_train_save_path)
        y_train_save = np.load(y_train_save_path)

    else:
        print('------Generate Datasets--------')
        x_train, y_train = generated(train_path, train_txt)

        print('-------Save Datasets--------')
        np.save(x_train_save_path, x_train)
        np.save(y_train_save_path, y_train)
