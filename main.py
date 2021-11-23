import numpy as np
import jieba
import random
import multiprocessing
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from load_data import *
# from data_preprocess.make_dataset import *  # 制作数据集
from data_preprocess.generate_npy import *
from train_model import *
from evaluate import fx_calc_map_label
from utils import *
from compare import *
import pickle
from tensorflow.keras.callbacks import TensorBoard


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)
cpu_count = multiprocessing.cpu_count()
parent_path = os.path.dirname(__file__)  # 获取当前文件路径

if __name__ == "__main__":
    CD = False   # 商品数据集
    Fasion_200k = False  # 因缺少硬件资源，所以未做实验,若切换此数据集需要将所有的38更改为5,You need to do it yourself
    vocab_dim = 100  # 词向量的维度
    n_iterations = 5  # ideally more..
    n_exposures = 3  # 所有频数超过3的词语
    window_size = 5
    input_length = 25  # 输入序列的长度
    max_len = 25  # 经过测试，每个句子的最大长度不超过21
    num_classes = 38  # 类别总数
    old_method = False  # 选择旧方法进行训练
    new_method = False  # 选择新方法训练
    draw = False
    compared = True  # 对比实验用
    out_dim = 512
    demonstration = True  # 演示用
    # weight_path = 'mnt/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    text_load_path = './data/text'
    power_data_path = "./data/power_img_data_npy"
    data_path = "./data/original_img_data_npy"
    # 读入顺序，对应数据集每个类的顺序,同时对应文本的读取顺序
    list_name = ['休闲裤', '半身裙', '女牛仔外套', '女牛仔裤', '女衬衫', '女西装', '文胸套装', '无帽卫衣', '棉衣棉服', '毛呢大衣',
                 '皮草', '睡袍', '背心吊带', '渔夫帽', '鸭舌帽', '卫衣', '棉衣', '牛仔外套', '牛仔裤', '短袖T恤', '衬衫', '西装',
                 '风衣', '马甲', '单肩包', '双肩包', '手提包', '腰包', '钱包', '吊坠', '戒指', '手镯', '中长靴', '商务鞋', '板鞋', '运动鞋', '雪地靴', '高跟鞋']
    # generate_npy(use_power_dataset=True)  # 生成npy格式图像数据和标签

    text_pre, t_label, tt_label = get_loader(text_load_path, list_name)  # 纯文本， 数据集标签， 增强数据集标签

    # 文本分词
    text_after = [jieba.lcut(document.replace('\n', '')) for document in text_pre]

    model = Word2Vec(size=vocab_dim,  # 建立一个空的模型对象，设置词向量的维度为100
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    # text_data原文本数据(词向量的索引)， power_text_data增强文本数据(词向量的索引)
    w2indx, w2vec, text_data, power_text_data = text_w2model(model, text_after, max_len)

    print('You will succeed...')
    n_symbols, embedding_weights = get_data(w2indx, w2vec, vocab_dim)

    # 需要进行对比试验吗
    if compared:
        text_train, text_test, image_train, image_test, train_onehot_label, test_onehot_label = \
            load_data_set(data_path, power_data_path, text_data, power_text_data,
                          t_label, tt_label)
        if demonstration:
            Image_Only(test_onehot_label)
            Text_Only(test_onehot_label)
            cross_loss(test_onehot_label)
            Ours(test_onehot_label)
        else:
            word_vec_test_mAP(text_test, vocab_dim, n_symbols, embedding_weights, input_length, test_onehot_label)
            txt_result(text_train, text_test, train_onehot_label, test_onehot_label,
                       vocab_dim, n_symbols, embedding_weights, input_length)
            img_result(text_data, t_label)

    if old_method:
        # 训练集和测试集的划分
        text_train, text_test, image_train, image_test, train_onehot_label, test_onehot_label = \
            load_data_set(data_path, power_data_path, text_data, power_text_data,
                          t_label, tt_label, use_power_data=False)

        # 构建多模态模型
        multi_model = MultiModel(vocab_dim, n_symbols, embedding_weights, input_length, out_dim)
        # run_eagerly指示模型是否应急切运行的可设置属性,这对于自定义的损失函数和张量的流动很有用
        # 急切地运行意味着您的模型将逐步运行，就像 Python 代码一样。您的模型可能运行得较慢，但您应该更容易通过进入各个层调用来调试它。
        # 默认情况下，我们会尝试将您的模型编译为静态图以提供最佳执行性能
        # 默认为False. 如果True，thisModel的逻辑将不会被包装在 a 中tf.function。建议将其保留为None除非您Model无法在 tf.function. 使用时不支持
        multi_model.compile(loss=Myloss(), optimizer='adam', run_eagerly=True)

        early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        history = multi_model.fit([image_train, text_train], train_onehot_label, batch_size=64, epochs=3,
                                  validation_split=0.3,
                                  verbose=1, callbacks=[early_stopping])
    if new_method:
        # 训练集：测试集 = 8：2
        # 训练集：验证集 = 9：1
        if CD:
            imgTrain, imgVal, txtTrain, txtVal, imgTest, txtTest, \
            Train_label_one_hot, Val_label_one_hot, Tst_label_one_hot = \
                generate_method(LoadData(), text_data, t_label)

            train_size = len(imgTrain)
            val_size = len(imgVal)

            multi_model = MultiModel(vocab_dim, n_symbols, embedding_weights, input_length, out_dim)

            multi_model.compile(loss=Myloss(), optimizer='adam', metrics=[class_metric], run_eagerly=True)
            Tensorboard = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)
        if Fasion_200k:
            from fasion_utils import  *

            vocab_dim = 500
            input_length = 15
            num_classes = 5

            img_path, text_data, label, n_symbols, embedding_weights = initialization()

            imgTrain, imgVal, txtTrain, txtVal, imgTest, txtTest, \
            Train_label_one_hot, Val_label_one_hot, Tst_label_one_hot = \
                generate_method(img_path, text_data, label, 0.3, 0.3, True)

            multi_model = MultiModel(vocab_dim, n_symbols, embedding_weights, input_length, out_dim)

            multi_model.compile(loss=Myloss(), optimizer='adam', metrics=[class_metric], run_eagerly=True)
            Tensorboard = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)

            history = multi_model.fit(BatchGen(batch_size=3, image_path=imgTrain,
                                               text=txtTrain, label=Train_label_one_hot),
                                      steps_per_epoch=2, epochs=2,
                                      validation_data=BatchGen(batch_size=1, image_path=imgVal,
                                                               text=txtVal, label=Val_label_one_hot),
                                      validation_steps=2,verbose=1)
        if CD:
            early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
            history = multi_model.fit(batchGen(batch_size=3, image_path=imgTrain,
                                               text=txtTrain, label=Train_label_one_hot),
                                      steps_per_epoch=2, epochs=2,
                                      validation_data=batchGen(batch_size=1, image_path=imgVal,
                                               text = txtVal, label=Val_label_one_hot),validation_steps=2,
                                      verbose=1, callbacks=[early_stopping])

    # 提取测试集特征
    if old_method:
        multi_feature = multi_model.predict([imgTest, txtTest], batch_size=32)
        label = tf.argmax(test_onehot_label, axis=1)
        result = fx_calc_map_label(multi_feature[:,num_classes:], label)
        print('...多模态图像检索 MAP = {}'.format(result))
    if new_method:
        Batch_size = 3
        n = len(imgTest)
        multi_feature = np.empty((n, out_dim), dtype=np.float32)
        num = n // Batch_size + 1 # 为了防止内存溢出
        for i in range(num):
            start = i * Batch_size
            end = (i + 1) * Batch_size
            end = min(end, n)
            print(f"正在进行{start} to {end}的预测...")
            feature = multi_model.predict(Img_Txt(imgTest, txtTest, start, end))
            multi_feature[start:end] = feature[:, num_classes:]
        # mAP计算
        label = tf.argmax(Tst_label_one_hot, axis=1)
        result = fx_calc_map_label(multi_feature, label)
        print('...多模态图像检索 MAP = {}'.format(result))

    if draw:
        # 绘制训练和验证的损失图像
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
