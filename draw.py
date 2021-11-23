from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import multiprocessing
import matplotlib.cm as cm
from load_data import *
import numpy as np
import jieba
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from utils import LoadData
from train_model import text_w2model
from all_colors import *
from time import time
from PIL import Image
import pickle
import wordcloud

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cpu_count = multiprocessing.cpu_count()
parent_path = os.path.dirname(__file__)

vocab_dim = 100  # 词向量的维度
n_iterations = 5  # ideally more..
n_exposures = 3  # 所有频数超过3的词语
window_size = 5
input_length = 25  # 输入序列的长度
max_len = 25  # 经过测试，每个句子的最大长度不超过21
text_load_path = './data/text'
all_img = False
# 读入顺序，对应数据集每个类的顺序,同时对应文本的读取顺序
list_name = ['休闲裤', '半身裙', '女牛仔外套', '女牛仔裤', '女衬衫', '女西装', '文胸套装', '无帽卫衣', '棉衣棉服', '毛呢大衣',
             '皮草', '睡袍', '背心吊带', '渔夫帽', '鸭舌帽', '卫衣', '棉衣', '牛仔外套', '牛仔裤', '短袖T恤', '衬衫', '西装',
             '风衣', '马甲', '单肩包', '双肩包', '手提包', '腰包', '钱包', '吊坠', '戒指', '手镯', '中长靴', '商务鞋', '板鞋', '运动鞋', '雪地靴', '高跟鞋']

text_pre, t_label, _ = get_loader(text_load_path, list_name)
text = [jieba.lcut(document.replace('\n', '')) for document in text_pre]
model = Word2Vec(size=vocab_dim,  # 建立一个空的模型对象
                 min_count=n_exposures,
                 window=window_size,
                 workers=cpu_count,
                 iter=n_iterations)
model.build_vocab(text)  # input: list遍历一次语料库建立词典
model.train(text, epochs=40, total_examples=model.corpus_count)  # 第2次遍历语料库简建立神经网络模型

imgTrain, imgTest, label_img_Train, labe_img_Tst = train_test_split(LoadData(), t_label,
                                                                    test_size=0.2, random_state=5)
imgTrain, imgVal, label_img_Train, labe_img_Val = train_test_split(imgTrain, label_img_Train,
                                                                   test_size=0.1, random_state=5)
_, text_jieba, _, _ = train_test_split(text_pre, t_label,
                                       test_size=0.2, random_state=5)
# l = labe_img_Tst
# index = [None] * 38
# for i in range(38):
#     index[i] = np.where(i == l)[0].shape[0]
# print(index)
# 测试集各个类别样本的数量
test_sizes = [21, 20, 18, 18, 23, 22, 24, 16, 19, 16, 20, 16, 18, 15, 20, 21,
              20, 21, 22, 20, 21, 17, 18, 21, 15, 22, 24, 18, 16, 13, 29, 18, 21,
              30, 19, 20, 20, 28]
all_sizes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


# 词的t-SNE显示
def t_SNE_2d():
    words_ak = []
    embeddings_ak = []
    for word in list(model.wv.vocab):
        embeddings_ak.append(model.wv[word])
        words_ak.append(word)

    tsne_ak_2d = TSNE(perplexity=50, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)

    def tsne_plot_2d(embeddings, words, a=1):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, 1))
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=colors, alpha=a, label="商品文本")
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), color="black",
                         textcoords='offset points', ha='right', va='bottom', size=10, weight="medium")
        plt.legend(loc=4)
        plt.grid(True)
        plt.savefig("t_SNE_2d.png", format='png', dpi=150, bbox_inches='tight')
        plt.show()

    tsne_plot_2d(embeddings_ak_2d, words=words_ak)


# 词的t-SNE显示
def t_SNE_3d():
    words_wp = []
    embeddings_wp = []
    for word in list(model.wv.vocab):
        embeddings_wp.append(model.wv[word])
        words_wp.append(word)
    tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
    embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)

    def tsne_plot_3d(title, embeddings, a=1):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = cm.rainbow(np.linspace(0, 1, 1))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label="文本词")
        plt.legend(loc=4)
        plt.title(title)
        plt.show()

    tsne_plot_3d('商品文本', embeddings_wp_3d, a=0.1)


# 词的t-SNE显示
def PCA():
    # 基于2d PCA拟合数据
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # 可视化展示
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()


# 初始数据集的t-SNE显示，包括训练集和测试集全部数据集的显示也可以
def get_data(img_path, label, img_size=100):
    Img = Img = np.empty((len(img_path), 3 * img_size * img_size), dtype=np.float64)
    for i in range(len(img_path)):
        img = Image.open(img_path[i])
        img = img.resize((img_size, img_size))
        img = np.reshape(img, (1, -1))
        Img[i - 1] = img / 255.0
    return Img, label


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=color_names[label[i] + 40])
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def feature_t_sne(data, label):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         'T-sne embedding of text (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


# 原始测试集t-SNE分布
def test_img_set_t_sne(imgTest, labe_img_Tst):
    data, label = get_data(imgTest, labe_img_Tst)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         'T-sne embedding of test set images (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


# test_img_set_t_sne(imgTest, labe_img_Tst)


# 测试集，图文特征的t-SNE
def test_multi_feature():
    f = open(os.path.join(parent_path, "data/multi_feature.pkl"), "rb")
    multi_feature = pickle.load(f)
    feature_t_sne(multi_feature, labe_img_Tst)


# test_multi_feature()

def test_img_feature():
    f = open(os.path.join(parent_path, "data/img_feature.pkl"), "rb")
    img_feature = pickle.load(f)
    feature_t_sne(img_feature, labe_img_Tst)


# test_img_feature()

def test_txt_feature():
    f = open(os.path.join(parent_path, "data/txt_feature.pkl"), "rb")
    txt_feature = pickle.load(f)
    feature_t_sne(txt_feature, labe_img_Tst)


# test_txt_feature()

def test_txt_lstm_feature():
    f = open(os.path.join(parent_path, "data/lstm_feature.pkl"), "rb")
    lstm_feature = pickle.load(f)
    feature_t_sne(lstm_feature, labe_img_Tst)


# test_txt_lstm_feature()


# 词云展示
def test_all_text_word_wordcloud():
    print("开始绘制词云...")
    # from scipy.misc import imread
    # mask = imread("fivestart.jpg")自动图片的显示
    txt_path = os.path.join(parent_path, "data/txt")
    big_txt = ""
    for i in os.listdir(txt_path):
        Path = os.path.join(txt_path, i)
        f = open(Path, "r", encoding="utf-8")
        txt = f.read()
        big_txt += txt
    ls = jieba.lcut(big_txt)
    txt = " ".join(ls)
    wc = wordcloud.WordCloud(font_path="msyh.ttc", \
                             width=2500, height=1500, background_color="white", max_words=1000)
    wc.generate(txt)
    # w.to_file("grwordcloud.png")
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


# test_all_text_word_wordcloud()

def test_txt_word_cloud(text_jieba):
    print("开始绘制词云...")
    big_txt = ""
    for i in text_jieba:
        big_txt = big_txt + i + "\n"
    ls = jieba.lcut(big_txt)
    txt = " ".join(ls)
    wc = wordcloud.WordCloud(font_path="msyh.ttc", \
                             width=2500, height=1500, background_color="white", max_words=300)
    wc.generate(txt)
    # w.to_file("grwordcloud.png")
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
# test_txt_word_cloud(text_jieba)


def test_set_pie_chart(sizes=test_sizes):
    print("开始绘制饼状图...")
    patches, l_text, p_text = plt.pie(sizes, labels=list_name, colors=color_name_38,
                                      labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                      startangle=90, pctdistance=0.8)

    # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
    # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
    # shadow，饼是否有阴影
    # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
    # pctdistance，百分比的text离圆心的距离
    # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

    # 改变文本的大小
    # 方法是把每一个text遍历。调用set_size方法设置它的属性
    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    # loc: 表示legend的位置，包括'upper right','upper left','lower right','lower left'等
    # bbox_to_anchor: 表示legend距离图形之间的距离，当出现图形与legend重叠时，可使用bbox_to_anchor进行调整legend的位置
    # 由两个参数决定，第一个参数为legend距离左边的距离，第二个参数为距离下面的距离
    plt.title('Data set pie chart', loc="right", fontsize="xx-large")
    plt.grid()
    plt.show(True)
# test_set_pie_chart(sizes=all_sizes)


# 文本检索mAP最终结果
text_mAP = [0.4092, 0.5894, 0.6357]
img_mAP = [0.4000, 0.5608, 0.6367]
cross_loss = [0.43605, 0.63707, 0.6769]
c_t_mAP = [0.4520, 0.6470, 0.7068]


def test_result_Histogram():
    mark = 0.15
    width = 0.15
    R = ["@1", "@5", "@10"]
    Image_only = [0.4092, 0.5894, 0.6357]
    Text_only = [0.4000, 0.5608, 0.6367]
    cross_loss = [0.4360, 0.6370, 0.6769]
    cross_and_triplet_loss = [0.4520, 0.6470, 0.7068]
    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(R))

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.bar(xticks, Image_only, width=width, label="Text_only", color="royalblue")
    ax.bar(xticks + mark, Text_only, width=width, label="Image_only", color="gray")
    ax.bar(xticks + 2*mark, cross_loss, width=width, label="cross_loss", color="burlywood")
    ax.bar(xticks + 3*mark, cross_and_triplet_loss, width=width, label="cross_and_triplet_loss", color="darkblue")

    # 需要你将每个组的起始坐标写到coordinate中，并且将所有点以列表的形式重新添加到ret中
    coordinate = [0.0, 1.0, 2.0]
    ret = [[0.400, 0.5608, 0.6367], [0.4092, 0.5894, 0.6357], [0.4360, 0.6370, 0.6769],
           [0.4520, 0.6470, 0.7068]]

    for i in range(len(ret[0])):
        margin = 0
        for j in range(len(ret)):
            xy = (coordinate[i] + margin, ret[j][i] * (1 + j / 200))
            s = str(ret[j][i])
            ax.annotate(
                s=s,  # 要添加的文本
                xy=xy,  # 将文本添加到哪个位置
                fontsize=10,  # 标签大小
                color="black",  # 标签颜色
                ha="center",  # 水平对齐
                va="baseline"  # 垂直对齐
            )
            margin += mark
    # ax.set_title("Grouped Bar plot", fontsize=15)
    ax.set_ylabel("mAP")
    # ax.set_xlabel("返回样本数")
    ax.legend()
    ax.set_xticks(xticks + 0.2)
    ax.set_xticklabels(R)
    plt.show()
# test_result_Histogram()

# 折线图
def Line_chart():
    R = ["@1", "@5", "@10"]
    Image_only = [0.4092, 0.5894, 0.6357]
    Text_only = [0.4000, 0.5608, 0.6367]
    cross_loss = [0.4360, 0.6370, 0.6769]
    cross_and_triplet_loss = [0.4520, 0.6470, 0.7068]
    ret = [[0.400, 0.5608, 0.6367], [0.4092, 0.5894, 0.6357], [0.4360, 0.6370, 0.6769],
           [0.4520, 0.6470, 0.7068]]

    color = ['red', 'yellow', 'green', 'blue', 'black']
    fig = plt.figure(figsize=(7, 4))
    for i in range(len(ret)):
        plt.plot(range(3), ret[i], c=color[i])
    plt.legend = ('upper left')
    plt.xlabel('Month')
    plt.ylabel('Rate')
    plt.title('Rate to Month')
    plt.tick_params(axis='both')
    plt.show()