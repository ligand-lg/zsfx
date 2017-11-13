import numpy as np
import math
from homework import conf_hw
#import conf_hw
from sklearn.decomposition import PCA
import time

def knn(x, X_train, Y_train, k, distance_type='o', knn_type='naive'):
    # 算距离
    distances = []
    index = 0
    for x_train in X_train:
        if distance_type == 'o':
            diff = x_train - x
            dist = math.sqrt((diff.T * diff)[0,0])
        elif distance_type == 'm':
            diff = x_train - x
            diff = [abs(x) for x in diff[0, :]]
            dist = sum(diff)
        else:
            e = Exception("unknown distance_type")
            raise e
        distances.append((dist, Y_train[index]))
        index += 1
    #  排序
    distances.sort()

    # 按前面k个赋值
    y_hat = 0
    if knn_type == 'naive':
        ones = 0
        for _, y in distances[:k]:
            if y == 1:
                ones += 1
        if ones > k/2:
            y_hat = 1
        else:
            y_hat = 0
    elif knn_type == 'improve':
        #  根据距离来生成权重
        W1 = []
        max_dist = distances[k][0]
        max_diff = max_dist - distances[0][0]
        scores = 0
        for dist, y in distances[:k]:
            w = (max_dist-dist)/max_diff
            W1.append(w)
        # 归一化
        W = [w/sum(W1) for w in W1]
        index = 0
        for _, y in distances[:k]:
            scores += W[index] * y
            index += 1
        y_hat = 1 if scores > 0.5 else 0
    else:
        e = Exception("unknown knn type")
        raise e
    return y_hat


if __name__ == '__main__':
    start = time.time()
    result = dict()
    for dis_type in ['o', 'm']:
        for knn_type in ['naive', 'improve']:
            error_ratios_in_k = []
            for k in range(3, 11):
                errors = 0
                rights = 0
                for qid in range(201, 205):
                    qid = str(qid)
                    print(qid)
                    X_train, Y_train = conf_hw.read_train(qid, type='class')
                    X_test, Y_test = conf_hw.read_test(qid, type='class')
                    # PCA降维
                    print('PCA...')
                    start_in = time.time()
                    pca = PCA(n_components=200)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                    print("PCA:{0}".format(time.time() - start_in))
                    start_in = time.time()
                    print('test...')
                    i = 0
                    for x in X_test:
                        y_hat = knn(x, X_train, Y_train, k, distance_type=dis_type, knn_type=knn_type)
                        if y_hat == Y_test[i]:
                            rights += 1
                        else:
                            errors += 1
                        i += 1
                error_ratio = errors/(errors+rights)
                print("test end :{0}".format(time.time() - start_in))
                print("error ratio: {0}".format(error_ratio))
                error_ratios_in_k.append(error_ratio)
            result[dis_type+'_'+knn_type] = error_ratios_in_k
    print(result)



