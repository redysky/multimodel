import numpy as np
import scipy.spatial


# 结果数组中的第一行数据表示的是image数组中第一个元素点与image数组中各个元素点的距离，计算两点之间的距离
def fx_calc_map_label(image, label, k=10, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, image, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, image, 'cosine')

    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []  # mAP,测试集100个样本的平均准确率
    for i in range(numcases):  # 所有行的循环，待检索图像/文本的循环
        order = ord[i]
        p = 0.0  # 精度,分母为当前返回的图像个数，大白话 --> 返回的7张图像中有来自同一类的个数
        r = 0.0  # 计数,在数据库中,与当前待检索数据库中当前待检索图像，同一类的图像个数

        for j in range(1, k+1):  # 被检索数据库的循环,这里的一个坑，不要将自身纳入待检索库中
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


if __name__ == "__main__":
    img = np.random.randint(0, 100, (10, 3))
    label = np.random.randint(0, 4, 100)
    print(fx_calc_map_label(img, label))
