import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from keras import backend as K
from keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops import math_ops
import numpy as np
import numpy

tf.keras.backend.set_floatx('float64')


def text_w2model(model, text_after, max_len):
    data = []
    model.build_vocab(text_after)  # input: list遍历一次语料库建立词典
    model.train(text_after, epochs=20, total_examples=model.corpus_count)  # 第2次遍历语料库简建立神经网络模型
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # {'T恤':1, '一':2, '一体':3, ...}1580
    # w2vec = {'T恤':array([-0.10420721, -0.50772285,...])} 1580=词库大小
    w2vec = {word: model.wv[word] for word in w2indx.keys()}

    for sentence in text_after:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)

    # pad_sequences函数是将序列转化为经过填充以后的一个长度相同的新序列
    # (3800, 25) 所有文本的词索引  [0, 0, ... , 167, 139]
    text_data = sequence.pad_sequences(data, maxlen=max_len)  # 大于此长度的序列将被截短，小于此长度的序列将在后部填0，默认为pre
    # 增强后，将源文本扩充四倍
    power_text_data = np.repeat(text_data, 4, axis=0)
    return w2indx, w2vec, text_data, power_text_data


def get_data(index_dict, word_vectors, vocab_dim):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于3的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化索引为0的词语
    for word, index in index_dict.items():  # 0索引都为0，从索引为1的词语开始，对每个词对应一个词向量
        embedding_weights[index, :] = word_vectors[word]
    return n_symbols, embedding_weights


class TextNet(tf.keras.Model):
    def __init__(self, vocab_dim, n_symbols, embedding_weights, input_length):
        super(TextNet, self).__init__()
        # 一个重要的结论(坑)：在构造器中不可以初始化Input函数,否则会报错 TypeError: Expected float64 passed to parameter 'y' of op 'Equal',
        # got 'collections' of type 'str' instead. Error: Expected float64, got 'collections' of type 'str' instead.
        # self.inputs = Input(shape=25, name="text_input")
        # 在embedding层中将会执行，将输入的文本索引转化成索引对应的词向量，
        # 组成完整的句子，维度为(batch_size, 句子的长度, 词向量的维度)
        self.embedding = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True,
                                   weights=[embedding_weights],
                                   input_length=input_length, trainable=False)
        self.lstm = LSTM(64, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense = Dense(512, activation='relu')
        self.dropout2 = Dropout(0.2)
    # 隐藏的坑，call函数中不要加入用不到的调用，会有警告
    def call(self, x, training=False, **kwargs):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        return x


class ImgNet(tf.keras.Model):
    def __init__(self, weight="imagenet"):
        super(ImgNet, self).__init__()
        # self.inputs = Input(shape=(224, 224, 3))
        # VGG16函数返回Model
        self.conv_base = VGG16(include_top=False, weights=weight, input_shape=(224, 224, 3))
        for layer in self.conv_base.layers:
            layer.trainable = False
        self.flatten = Flatten()
        self.dense = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout = Dropout(0.2)

    def call(self, inputs, **kwargs):
        x = self.conv_base(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        return x


# 一个问题：dense_pre需要手动设置成分类的类别，不然会出现调用错误
class MultiModel(tf.keras.Model):
    def __init__(self, vocab_dim, n_symbols, embedding_weights, input_length,
                 num_classes=38, weight="imagenet", out_dim=512):
        super(MultiModel, self).__init__()
        self.img_net = ImgNet(weight)
        self.text_net = TextNet(vocab_dim, n_symbols, embedding_weights, input_length)
        self.dense_256 = Dense(256, activation='relu')
        self.dense_512 = Dense(512, activation='softmax')
        self.dense_64 = Dense(64, activation='relu')
        self.dense_1 = Dense(1, activation='sigmoid')
        self.dense_2 = Dense(2, activation='softmax')
        self.dense_l2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dense_pre = Dense(38, activation='softmax')
        self.dropout = Dropout(0.2)
        self.multiply = Multiply()
        self.add = Add()
        self.concat_2 = Concatenate()
        self.concat_1 = Concatenate(axis=1)
        self.reshape = Reshape((2, 1))
        self.permute = Permute((2, 1))

    # 类通道注意力和残差模块 --> 文本和图像特征均适用
    def res_and_att_block(self, x):
        y = self.dense_256(x)
        y = self.dense_512(y)
        y = self.multiply([y,x])
        y = self.add([y, x])
        return y

    # 图像和文本特征的权值模块
    def weight_block(self, x):
        y = self.dense_64(x)
        y = self.dense_1(y)
        return y

    # 获取图像和文本的权重
    def get_weight_block(self, x, y):
        w = self.concat_2([x, y])
        w = self.dense_2(w)
        w = self.reshape(w)
        return w

    # 加权
    def weighting(self, x, y, w):
        # 转换原始图像和文本特征维度
        def trans_space(x_dim):
            x_dim = tf.expand_dims(x_dim, axis=-1)
            x_dim = self.permute(x_dim)
            return x_dim

        x = trans_space(x)
        y = trans_space(y)
        z = self.concat_1([x, y])
        # 加权
        z = self.multiply([w, z])  # 此处为(2, 1) • (2, 512) 点乘 ,此处batch_size不会影响乘积的结果
        # 加权后将图像特征和文本特征连接
        z = self.concat_2([z[:, 0], z[:, 1]])  # z(batch_size, 指定行, 所有列)
        return z

    # call函数只能接受一个参数,但是这个参数可以是列表或者元组或者字典等形式
    # 这里需要注意，默认在调用Model类时，输入model的参数为(batch_size,...)
    def call(self, inputs, **kwargs):
        res = []
        x, y = inputs
        img = self.img_net(x)
        txt = self.text_net(y)

        x = self.res_and_att_block(img)
        y = self.res_and_att_block(txt)

        img_y = self.weight_block(x)
        txt_y = self.weight_block(y)

        w = self.get_weight_block(img_y, txt_y)
        z = self.weighting(x, y, w)

        # 最后走一遍全连接层进行分类
        z = self.dense_l2(z)
        t = self.dropout(z)
        pre = self.dense_pre(t)  # 38维
        c = self.concat_2([pre, z])

        return c
# 我的损失函数制作成功：
# y_pred为网络预测，并且网络中的call函数只能输出一个参数，
# 这个参数表示成列表或者元组等都不可以，但是可以将模型中的其他特征，可以是好几项通过tf.Concatenate()函数组合成统一的形式输出
# 在构造的Loss子类中通过矩阵切片分别访问各个特征
# 所以在通过模型预测测试集时，不要忘了输出指定特征
# 这里在训练时还有一个坑，就是accurary，loss改变后同时需要自己重写metrics方法,需要定义你是想输出分类准确率还是其他的
class Myloss(tf.keras.losses.Loss):
    def __init__(self, name="Myloss"):
        super().__init__(name=name)
        self.cross_loss = tf.losses.CategoricalCrossentropy()
        self.multiply = Multiply()  # 与此对应的是tf.matmul()叉乘

    def call(self, y_true, y_pred):
        # 可以拉近同类样本
        # def compute_loss(y_true, y_pred):
        #     y_pred = y_pred.numpy()
        #     y_true = y_true.numpy()
        #     y_true = np.argmax(y_true, axis=1)
        #     d_i = []
        #     index = list(range(y_true.shape[0])) + list(range(y_true.shape[0]))
        #     for i in range(len(index)):
        #         for j in range(len(index)):
        #             if index[i] != index[j] and y_true[index[i]] == y_true[index[j]] and \
        #                     sorted([index[i], index[j]]) not in d_i:
        #                 d_i.append([index[i], index[j]])
        #         if len(d_i) == 0:
        #             d_i.append([index[i], index[i]])
        #     n = len(d_i)
        #     loss = 0
        #     for a, b in d_i:
        #         loss += 1 * np.log(K.sum(1+np.exp(K.square(y_pred[a] - y_pred[b]))))
        #     return loss
        def compute_loss(y_true, y_pred):
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
            y_true = np.argmax(y_true, axis=1)
            d = []
            index = list(range(y_true.shape[0])) + list(range(y_true.shape[0]))
            for i in range(len(index)):
                d_i = []
                for j in range(len(index)):
                    if index[i] != index[j] and y_true[index[i]] == y_true[index[j]] and \
                            sorted([index[i], index[j]]) not in d_i:
                        for k in range(len(index)):
                            if index[i] != index[k]:
                                d_i.append([index[i], index[j], index[k]])
                np.random.shuffle(d_i)
                d += d_i[:3]
            np.random.shuffle(d)
            n = len(d)
            loss = 0.0
            triplet_count = 1.0
            for i, j, k in d:
                w = 1.0
                triplet_count += w
                loss += w * np.log(1 +
                                   np.exp(pairwise_distances(y_pred[index[i]], y_pred[index[j]]) -
                                          pairwise_distances(y_pred[index[i]], y_pred[index[k]]), dtype=np.float128))
            loss /= triplet_count
            return loss

        def pairwise_distances(x, y):
            dist = sigmod(K.sum(K.square(x - y)))
            return tf.clip_by_value(dist, 0.0, np.inf)

        def calc_label_sim(label):
            Sim = tf.matmul(label, tf.transpose(label))
            return Sim

        def sigmod(x):
            return tf.keras.activations.sigmoid(x)

        cross_loss = self.cross_loss(y_true, y_pred[:, :38])
        same_class = compute_loss(y_true, y_pred[:, 38:])
        # theta11 = tf.expand_dims(sigmod(K.sum(K.square(y_pred[:, 38:] - y_pred[:, 38:])
        #                                , axis=1)), axis=-1)  # (batch_size, 1)
        # print("theta11",theta11.shape)
        # # 减去相同类
        # losss = cross_loss - tf.matmul(calc_label_sim(y_true), theta11)

        return cross_loss+same_class

def class_metric(y_true, y_pred):
    values = math_ops.cast(
        math_ops.equal(
            math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred[:,:38], axis=-1)),
        K.floatx())
    return values

def text_model(vocab_dim, n_symbols, embedding_weights, input_length):
	x = tf.keras.layers.Input(shape=25, name="text_input")
	x1 = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights],
				   input_length=input_length, trainable=False)(x)
	x2 = LSTM(64, activation='relu')(x1)
	x3 = Dropout(0.2)(x2)
	x4 = Dense(512, activation='relu')(x3)
	x5 = Dropout(0.2)(x4)
	x6 = Dense(38, activation="softmax")(x5)
	model_pre = Model(inputs=x, outputs=x6)
	model_512 = Model(x, x4)
	return model_pre, model_512

def image_model():
	conv_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
	for layer in conv_base.layers:
		layer.trainable = False
	last = conv_base.output
	x = tf.keras.layers.Flatten()(last)
	x1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
	x2 = tf.keras.layers.Dropout(0.2)(x1)
	x3 = Dense(38, activation="softmax")(x2)
	model_pre = Model(inputs=conv_base.input, outputs=x3)
	model_512 = Model(conv_base.input, x1)
	return model_pre, model_512