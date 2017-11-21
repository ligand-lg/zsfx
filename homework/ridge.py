# coding:utf-8

import numpy as np
from homework import conf_hw
#import conf_hw
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def ridge(X_train, y_train, k, X_test):
    m, n = X_train.shape
    I = np.mat(np.eye(n))
    k_I = k * I
    aim_mat = X_train.T * X_train
    aim_mat = aim_mat + k_I
    while True:
        try:
            pinv = np.linalg.pinv(aim_mat)
            break
        except Exception as e:
            print(e)
            print('again')
    params_mat = pinv * X_train.T * y_train
    return X_test * params_mat


def select_best_k():
    X_trains = dict()
    Y_trains = dict()
    X_tests = dict()
    Y_tests = dict()
    for qid in range(201, 251):
        qid = str(qid)
        e1, e2 = conf_hw.read_train(qid)
        pca = PCA(n_components=300)
        e1 = np.mat(pca.fit_transform(e1))
        e1 = np.column_stack((e1, np.ones((e1.shape[0], 1))))
        X_trains[qid] = e1
        Y_trains[qid] = e2
        e1, e2 = conf_hw.read_test(qid)
        e1 = np.mat(pca.transform(e1))
        e1 = np.column_stack((e1, np.ones((e1.shape[0], 1))))
        X_tests[qid] = e1
        Y_tests[qid] = e2

    result = []
    k_range = list(range(20))
    for k in k_range:
        time_in = time.time()
        avg_mae = 0
        for qid in range(201, 251):
            print(qid)
            qid = str(qid)
            X_train = X_trains[qid]
            Y_train = Y_trains[qid]
            X_test = X_tests[qid]
            Y_test = Y_tests[qid]

            y_hat = ridge(X_train, Y_train, k, X_test)
            mae = sum(abs(y_hat - Y_test))[0, 0] / X_test.shape[0]
            avg_mae += mae
        avg_mae /= 50
        print('k={0}, mae={1}'.format(k, avg_mae))
        print('speed {0} min'.format((time.time() - time_in) / 60))
        result.append(avg_mae)
    plt.plot(k_range, result)
    plt.xlabel('k')
    plt.ylabel('MAE')
    plt.show()
# best : k = 10


def predict(k):
    fout = open('../data/predict_ridge.txt', 'wt', encoding='utf-8')
    for qid in range(201, 251):
        qid = str(qid)
        X_train, Y_train = conf_hw.read_train(qid)
        pca = PCA(n_components=300)
        X_train = np.mat(pca.fit_transform(X_train))
        X_train = np.column_stack((X_train, np.ones((X_train.shape[0], 1))))
        X_test, Y_test = conf_hw.read_test(qid)
        X_test = np.mat(pca.transform(X_test))
        X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))

        y_hat = ridge(X_train, Y_train, k, X_test)
        mae = sum(abs(y_hat - Y_test))[0, 0] / X_test.shape[0]

        i = 0
        for relationship in conf_hw.coll_test.find({'query_id': str(qid)}):
            article_id = relationship['article_id']
            id_code = relationship['id_code']
            score = y_hat[i]
            fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
            i += 1
    conf_hw.MAE('ridge')


if __name__ == '__main__':
    # select_best_k()
    predict(k=10)

# MAE: 13.892704692969026

