from homework import conf_hw
#import conf_hw
import numpy as np


# normalize row vetcors
def normalize(Vs):
    row, column = Vs.shape
    for i in range(row):
        sum_ = Vs[i,:] * Vs[i, :].T
        if sum_[0, 0] == 0:
            continue
        Vs[i, :] /= np.sqrt(sum_)
    return Vs


def distantce(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


# load data
X_train, Y_train = conf_hw.read_train(201, 'class')
X_test, Y_test = conf_hw.read_test(201, 'class')
X_train = normalize(X_train)
X_test = normalize(X_test)
row, column = X_train.shape


weight_row, weight_column = (2, column)
W = np.random.random((weight_row, weight_column))
W = np.mat(W)
W = normalize(W)


# training
alpha = 0.5
T = 5
w0_lab = 0
w1_lab = 0
w_sum = [0, 0]
for t in range(T):
    for r in range(row):
        dist0 = distantce(W[0], X_train[r])
        dist1 = distantce(W[1], X_train[r])
        if dist0 < dist1:
            W[0] += alpha*(X_train[r]-W[0])
        else:
            W[1] += alpha*(X_train[r]-W[1])
        normalize(W)
for r in range(row):
    dist0 = distantce(W[0], X_train[r])
    dist1 = distantce(W[1], X_train[r])
    score = Y_train[r, 0]
    if dist0 < dist1:
        w_sum[0] += score
    else:
        w_sum[1] += score
if w_sum[0] > w_sum[1]:
    w0_lab = 1
else:
    w1_lab = 1


# test
y_hat = np.mat(np.zeros((X_test.shape[0], 1)))
for r in range(X_test.shape[0]):
    dist0 = distantce(W[0], X_test[r])
    dist1 = distantce(W[1], X_test[r])
    if dist0 < dist1:
        y_hat[r, 0] = w0_lab
    else:
        y_hat[r, 0] = w1_lab
abs(y_hat-Y_test).sum()/X_test.shape[0]


i = 0
with open('../data/predict_som.txt', 'wt', encoding='utf-8') as fout:
    for relationship in conf_hw.coll_class_test.find({'query_id': '201'}):
        article_id = relationship['article_id']
        id_code = relationship['id_code']
        score = y_hat[i]
        fout.write(
            '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format('201', article_id, id_code, score))
        i += 1
