from load_data import load_data_set
from train_model import text_model, image_model
from evaluate import fx_calc_map_label
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from utils import *
from tensorflow.keras.callbacks import TensorBoard
from load_data import *
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pickle


def batchGenimg(batch_size, image_path, label, original=True, multiple=4):
    """
    image_path:List() 待增强图像的路径
    text:经过word2vec处理过后的文本数据
    original:bool 是否增加原生态图像数量
    multiple:int 增加的数量
    """
    while True:
        imageBatch, labelBatch = [], []
        for i in range(batch_size):
            ImglittleBatch, labellittleBatch = [], []
            index = np.random.randint(0, len(image_path))
            if original:
                image = augmentImage(image_path[index])
                image = preProcessing(image)
                imageBatch.append(image)
                labelBatch.append(label[index])
            else:
                for j in range(multiple):
                    image = augmentImage(image_path[index])
                    image = preProcessing(image)
                    ImglittleBatch.append(image)
                    labellittleBatch.append(label[index])
                imageBatch += ImglittleBatch
                labelBatch += labellittleBatch

        # 对列表中的元素加入随机性，打乱，固定打乱顺序
        state = np.random.get_state()
        np.random.shuffle(imageBatch)
        np.random.set_state(state)
        np.random.shuffle(labelBatch)
        yield np.asarray(imageBatch), np.asarray(labelBatch)


# 只进行文本检索
def txt_result(text_train, text_test, train_onehot_label, test_onehot_label,
               vocab_dim, n_symbols, embedding_weights, input_length):
    txt_pre, img_out = text_model(vocab_dim, n_symbols, embedding_weights, input_length)
    txt_pre.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    Tensorboard = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)
    history_txt = txt_pre.fit(text_train, train_onehot_label, batch_size=32, epochs=30,
                              validation_split=0.2, verbose=1, callbacks=[early_stopping])
    lstm_feature = img_out.predict(text_test)
    f = open('./data/lstm_feature.pkl', 'wb')
    pickle.dump(lstm_feature, f)
    f.close()
    label = tf.argmax(test_onehot_label, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(lstm_feature, label, k=R)
        print(f'...只进行文本检索@{R} = MAP = {result}')


# 只进行图像检索
def img_result(text_data, t_label):
    imgTrain, imgVal, txtTrain, txtVal, imgTest, txtTest, \
    Train_label_one_hot, Val_label_one_hot, Tst_label_one_hot = \
        generate_method(LoadData(), text_data, t_label)

    model_pre, model_out = image_model()
    model_pre.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    history = model_pre.fit(batchGenimg(batch_size=2, image_path=imgTrain, label=Train_label_one_hot),
                            steps_per_epoch=2, epochs=20,
                            validation_data=batchGenimg(batch_size=1, image_path=imgVal,
                                                        label=Val_label_one_hot), validation_steps=1,
                            verbose=1, callbacks=[early_stopping])
    Batch_size = 3
    n = len(imgTest)
    img_feature = np.empty((n, 512), dtype=np.float32)
    num = n // Batch_size + 1  # 为了防止内存溢出
    for i in range(num):
        start = i * Batch_size
        end = (i + 1) * Batch_size
        end = min(end, n)
        print(f"正在进行{start} to {end}的预测...")
        feature = model_out.predict(Img_Txt(imgTest, txtTest, start, end)[0])
        img_feature[start:end] = feature
    label = tf.argmax(Tst_label_one_hot, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(img_feature, label, k=R)
        print(f'...只进行图像的检索@{R} = MAP = {result}')


# 经过word2vec后的检索精度
def word_vec_test_mAP(text_test, vocab_dim, n_symbols, embedding_weights, input_length, test_onehot_label):
    inputs = tf.keras.layers.Input(shape=25, name="text_input")
    x = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True,
                  weights=[embedding_weights],
                  input_length=input_length, trainable=False)(inputs)
    model = Model(inputs, x)
    txt_feature = model.predict(text_test)
    txt_feature = np.reshape(txt_feature, (len(text_test), -1))  # 注意resize和reshape的区别
    f = open('./data/txt_feature.pkl', 'wb')
    pickle.dump(txt_feature, f)
    f.close()
    label = tf.argmax(test_onehot_label, axis=1)
    result = fx_calc_map_label(txt_feature, label)
    print('...输入网络前的word2vec文本检索 MAP = {}'.format(result))


def Image_Only(test_onehot_label):
    f = open("./data/img_feature.pkl", "rb")
    img_feature = pickle.load(f)
    label = tf.argmax(test_onehot_label, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(img_feature, label, k=R)
        print('...Image Only @{} MAP = {}'.format(R, result))


def Text_Only(test_onehot_label):
    f = open("./data/lstm_feature.pkl", "rb")
    lstm_feature = pickle.load(f)
    label = tf.argmax(test_onehot_label, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(lstm_feature, label, k=R)
        print('...Text Only MAP @{} = {}'.format(R, result))


def cross_loss(test_onehot_label):
    f = open("./data/cross_loss.pkl", "rb")
    cross_loss = pickle.load(f)
    label = tf.argmax(test_onehot_label, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(cross_loss, label, k=R)
        print('...cross loss @{} = {}'.format(R, result))


def Ours(test_onehot_label):
    f = open("./data/multi_feature.pkl", "rb")
    multi_feature = pickle.load(f)
    label = tf.argmax(test_onehot_label, axis=1)
    for R in [1, 5, 10]:
        result = fx_calc_map_label(multi_feature, label, k=R)
        print('...Ours MAP @{} = {}'.format(R, result))
