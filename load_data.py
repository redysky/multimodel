import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from data_preprocess.text_process import *


# 包括将所有text.csv文本组合成一个统一的csv文件，并制作文本标签
def get_loader(text_load_path, list_name):
    list_csv = []

    for i in range(len(list_name)):
        text_path = os.path.join(text_load_path, list_name[i])
        list_csv.append(text_path + '文本.csv')

    # 文本数据,返回的文本是与图片一一对应的
    list_text = np.array([])
    for i in range(len(list_csv)):
        list_text = np.concatenate(
            (list_text, np.array(pd.read_csv(list_csv[i], encoding='gb18030', header=None, index_col=None)[0])))

    # list_num为文本的标签，对应数据集文本排列，每间隔一百变成下一个类
    text_label = np.array([], dtype=np.int32)
    for i in range(len(list_csv)):
        text_label = np.concatenate((text_label, int(i) * np.ones(100, dtype=np.int32)))

    # list_num1增强版文本的标签
    text_labels = np.array([], dtype=np.int32)
    for i in range(len(list_csv)):
        text_labels = np.concatenate((text_labels, int(i) * np.ones(400, dtype=np.int32)))

    return list_text, text_label, text_labels


def load_data_set(data_path, power_data_path, text_data, power_text_data,
                  t_label, tt_label, num_classes=38, use_power_data=False, test_size=0.2):
    if use_power_data:
        # 数据集读取
        data_img = np.load(os.path.join(power_data_path, 'x_train3.npy')) / 255
        data_label = np.load(os.path.join(power_data_path, 'y_train3.npy'))

        # 分割训练集和测试集，按照7：3划分
        text_train, text_test, text_train_label, text_test_label = train_test_split(power_text_data, tt_label,
                                                                                    test_size=test_size,
                                                                                    random_state=5)
        image_train, image_test, image_train_label, image_test_label = train_test_split(data_img, data_label,
                                                                                        test_size=test_size,
                                                                                        random_state=5)
        # 标签独热化
        train_onehot_label = tf.keras.utils.to_categorical(image_train_label,
                                                           num_classes=num_classes)
        test_onehot_label = tf.keras.utils.to_categorical(image_test_label,
                                                           num_classes=num_classes)

        return text_train, text_test, image_train, image_test, train_onehot_label, test_onehot_label
    else:
        data_img = np.load(os.path.join(data_path, 'x_train3.npy')) / 255
        data_label = np.load(os.path.join(data_path, 'y_train3.npy'))

        text_train, text_test, text_train_label, text_test_label = train_test_split(text_data, t_label,
                                                                                    test_size=test_size,
                                                                                    random_state=5)
        image_train, image_test, image_train_label, image_test_label = train_test_split(data_img, data_label,
                                                                                        test_size=test_size,
                                                                                        random_state=5)
        train_onehot_label = tf.keras.utils.to_categorical(image_train_label,
                                                           num_classes=num_classes)
        test_onehot_label = tf.keras.utils.to_categorical(image_test_label,
                                                          num_classes=num_classes)

        return text_train, text_test, image_train, image_test, \
               train_onehot_label, test_onehot_label


# 需要划分训练集，验证集和测试集
def generate_method(image_path, text, label, test_size=0.01, val_size=0.1, fasion=False):
    # 划分训练集，测试集
    imgTrain, imgTest, label_img_Train, labe_img_Tst = train_test_split(image_path, label,
                                                                       test_size=test_size, random_state=5)

    txtTrain, txtTest, label_txt_Train, labe_txt_Tst = train_test_split(text, label,
                                                                       test_size=test_size, random_state=5)
    # 划分训练集，验证集
    imgTrain, imgVal, label_img_Train, labe_img_Val = train_test_split(imgTrain, label_img_Train,
                                                                       test_size=val_size, random_state=5)

    txtTrain, txtVal, label_txt_Train, labe_txt_Val = train_test_split(txtTrain, label_txt_Train,
                                                                       test_size=val_size, random_state=5)
    if fasion:
        Train_label_one_hot = tf.keras.utils.to_categorical(label_img_Train, num_classes=5)
        Val_label_one_hot = tf.keras.utils.to_categorical(labe_img_Val, num_classes=5)
        Tst_label_one_hot = tf.keras.utils.to_categorical(labe_img_Tst, num_classes=5)
    else:
        Train_label_one_hot = tf.keras.utils.to_categorical(label_img_Train, num_classes=38)
        Val_label_one_hot = tf.keras.utils.to_categorical(labe_img_Val, num_classes=38)
        Tst_label_one_hot = tf.keras.utils.to_categorical(labe_img_Tst, num_classes=38)

    return imgTrain, imgVal, txtTrain, txtVal, imgTest, txtTest, \
        Train_label_one_hot, Val_label_one_hot, Tst_label_one_hot