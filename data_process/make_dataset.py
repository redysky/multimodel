# 数据集的图片处理---包括将所有图片排号和将所有图片转换为JPG格式
# 对原始数据集的处理，将其转变为这种形式  图片（1，2，3 ……total_img），
# 并存储到final_path中,得到的是原始图像集，和增强后的图像集
import os
from PIL import Image
import cv2 as cv
# from make_power_data import *
# from generate_npy import *


def make_dataset():
    # 两步
    # 1.定位到每一张图片源路径
    # 2.遍历每一张图片转换格式并保存
    cur_path = os.path.dirname(__file__)  # 获取当前文件路径
    parent_path = os.path.dirname(cur_path)  # 获取当前文件夹父目录
    f_path = os.path.join(parent_path,r'商品数据集')  # 源文件位置，使用时需要将 数据集描述.txt删除
    s_path = os.path.join(parent_path,r'E:/aaa')  # mkdir()      #按自己的要求实现的保存路径（不用管）
    final_path = os.path.join(parent_path,r'data/img')  # 需要保存的路径
    list_1 = os.listdir(f_path)
    list_ = []
    total_list = []
    for i in range(len(list_1)):
        s_1 = os.path.join(f_path, list_1[i])  # 一级目录
        list_2 = os.listdir(s_1)  # 这里定位到每个类顺序是随机的
        total_list += list_2
        for j in range(len(list_2)):
            s_2 = os.path.join(s_1, list_2[j])
            s_3 = os.path.join(s_2, '图像')  # 定位到图像
            list_3 = os.listdir(s_3)
            list_3.sort(key=lambda x: int(x[:-4]))
            for k in range(100):
                s_4 = os.path.join(s_3, list_3[k])
                list_.append(s_4)  ##定位到每一张图片的路径

    #		s_5 = os.path.join(s_path,list_2[j])      #s_path的保存路径
    #		os.makedirs(s_5)
    count = 0
    for s in range(len(total_list)):
        wait_save_path = os.path.join(s_path, total_list[s])
        for a in range(s * 100, s * 100 + 100):
            img = Image.open(list_[a])  # 待保存图片的完整路径
            if Image.open(list_[a]).format == 'PNG':  # 将所有的RGBA图片转换为RGB
                ss = Image.open(list_[a]).convert('RGB')
                img.save(final_path + '/' + str(a + 1) + '.jpg')  # .save()的格式为路径+需要保存的图像的名称
                print(final_path + '/' + str(a + 1) + '.jpg', '已保存')
            else:
                pass
                img.save(final_path + '/' + str(a + 1) + '.jpg')
                print(final_path + '/' + str(a + 1) + '.jpg', '已保存')
    print("每个类的顺序:",total_list)
    # 这里打印存储的每个类的顺序，对应main.py中的list_name
    # ['休闲裤', '半身裙', '女牛仔外套', '女牛仔裤', '女衬衫', '女西装', '文胸套装', '无帽卫衣', '棉衣棉服',
    # '毛呢大衣', '皮草', '睡袍', '背心吊带', '渔夫帽', '鸭舌帽', '卫衣', '棉衣', '牛仔外套', '牛仔裤', '短袖T恤',
    # '衬衫', '西装', '风衣', '马 甲', '单肩包', '双肩包', '手提包', '腰包', '钱包', '吊坠', '戒指', '手镯', '中长靴',
    # '商务鞋', '板鞋', '运动鞋', '雪地靴', '高跟鞋']


# 生成原始图像集
make_dataset()
# # 生成增强图像集
# make_data_power()

