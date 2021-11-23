# 功能：将原始数据集中的图片使用ImageDataGenerator增强，将增强后的图像保存至save_path中
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def make_data_power():
    cur_path = os.path.dirname(__file__)  # 获取当前文件路径
    parent_path = os.path.dirname(cur_path)  # 获取当前文件夹父目录
    tmp_path = 'data/Temporary_folder'  # # 临时文件夹
    train_path = os.path.join(parent_path,tmp_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    start_file = os.path.join(parent_path,'data/img')  # 原始图片集文件夹

    number = 3  # 增强数目
    save_path = os.path.join(parent_path, 'data/power_img')  # 增强图像集文件夹

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.3,
        shear_range=0.2, zoom_range=0.25,
        horizontal_flip=True, vertical_flip=True)

    # 由于库函数的特殊性（需要将待增强的图像放到文件夹里传入库函数），
    # 所以需要保证数据集中的图片一个图片一个文件夹,结构为 文件夹（文件夹（图像））
    # 将图片转存至临时文件夹中
    def gen_file():
        for i in tqdm(range(len(os.listdir(start_file)))):
            save_file = train_path + '/' + str(i + 1) + '/' + str(i + 1) + '/' + str(i + 1)
            os.makedirs(train_path + '/' + str(i + 1) + '/' + str(i + 1))
            img = Image.open(start_file + '/' + str(i + 1) + '.jpg')
            img.save(save_file + '.jpg')

    if len(os.listdir(train_path)) == 0:
        gen_file()
    n = len(os.listdir(train_path))
    # 完成图像的增强操作，并保存至final_path,也即临时文件夹图像中增加到四张
    for i in tqdm(range(n)):
        start_path = train_path + '/' + str(i + 1)  # E/1
        final_path = train_path + '/' + str(i + 1) + '/' + str(i + 1)  # E/1/1
        for i in range(number):
            _, _ = next(train_datagen.flow_from_directory(start_path, target_size=(224, 224),
                                                          batch_size=1, shuffle=True, seed=4, save_to_dir=final_path,
                                                          save_format='jpg', follow_links=True))
    # 完成对临时文件夹的转存操作，将其存储为这种形式 power_data(1,2,3,4,...,total_img)
    for i in tqdm(range(n)):
        list_pic = os.listdir(
            train_path + '/' + str(i + 1) + '/' + str(os.listdir(train_path + '/' + str(i + 1))[0]))  # E:/aaa/1/1
        # 算上原始图像list_pic应该等于4
        # 逐一打开每张图像并将其改名保存至save_path,jpg格式
        for j in range(len(list_pic)):
            img = Image.open(
                train_path + '/' + str(i + 1) + '/' + str(os.listdir(train_path +
                                                                     '/' + str(i + 1))[0]) + '/' + list_pic[j])
            img.save(save_path + '/' + str(j + i * len(list_pic) + 1) + '.jpg')

