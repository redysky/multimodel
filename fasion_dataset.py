import os
from PIL import Image
import numpy as np
from stop_words import *
from tqdm import tqdm
from train_model import text_w2model
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing


path = r'E:\数据集\Fashion-200k'  # 数据集路径
label_path = "labels/labels"     # 标签路径


def Fashion_200k(path, label_path, split="train"):
    all_label_path = os.path.join(path, label_path)
    all_name = os.listdir(all_label_path)
    all_words = []  # 所有文本的词
    all_txt = []  # 文本数据
    label_data = []  # 标签
    i = 0  # 总类 = 5
    all_img_path = []  # 所有图像的路径
    max_len = 0  # 训练集最大 = 13 ，测试集最大 = 12
    count = 1  # 计数用
    for name in all_name:
        if name.split("_")[1] == split:  # dress_train_detect_all
            print(f"正在操作{name}...")
            with open(os.path.join(all_label_path, name), "r", encoding='utf-8') as f:
                contents = f.readlines()
                for content in contents:
                    print(f"正在读取Fasion-200k 第 {count} 行...")
                    value = content.split()
                    tmp = []  # 需要先初始化，保存每一段去停用词后的文本
                    # 先去停用词后将所有词保存至列表，并保存文本序列
                    txt = " ".join(value[2:])  # 字符串  # .gray's delaney crochet sleeve dress
                    for j in symbol:
                        txt = txt.replace(j, " ")  # 干净的字符串
                    txt_list = txt.split(" ")  # ['', 'gray', 's', 'delaney', 'crochet', 'sleeve', 'dress']
                    for word in txt_list:  # 这里需要保证读取顺序
                        if word not in stop_words:
                            tmp.append(word)
                    all_words.append(tmp)  # 分词后的每一段文本一个列表，等待word2vec转换
                    all_txt.append(" ".join(tmp))
                    label_data.append(i)
                    all_img_path.append(os.path.join(path, value[0]))
                    count += 1
            i += 1
    return all_img_path, all_txt, label_data, all_words


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    img_path, txt_data, label, all_words = Fashion_200k(path, label_path)
    # model = Word2Vec(size=500,  # 建立一个空的模型对象，设置词向量的维度为100
    #                  min_count=5,  # 频数
    #                  window=3,  #
    #                  workers=cpu_count,
    #                  iter=5)
    # w2indx, w2vec, text_data, power_text_data = text_w2model(model, all_words, max_len=15)
    #
    # print("最大长度为", max_len)
    # print("img_path", len(img_path))
    # print(f"txt {len(txt_data)}")
    # print("label", len(label))
    # print("词", len(all_words))
    # print(all_words[:3])
