# coding:utf-8

import numpy as np
#from homework import conf_hw
import conf_hw


def batch_gradient_descent():
    # 两次迭代损失函数之差小于改阈值时停止迭代
    epsilon = 0.0005
    # 步长/学习率
    alpha = 0.05
    for qid in range(201, 251):
        time = 0
        eroor1 = 0
        error0 = 0
        diff = 0
        data_mat, score_mat = conf_hw.read_train(qid)
        row, column = data_mat.shape
        # 添加截距最后一列
        data_mat = np.column_stack((data_mat, np.ones((row, 1))))
        data_mat = np.mat(data_mat)

        # 初始化参数, 列向量
        theta_v = np.mat(np.zeros((1+column, 1)))
        while True:
            diff = (data_mat * theta_v) - score_mat
            error1 = (diff.T * diff / row)[0, 0]
            #print('mse: {0}'.format(error1))
            if abs(error1-error0) < epsilon:
                break
            else:
                error0 = error1
            time += 1
            theta_v -= ((alpha/row) * (diff.T * data_mat)).T
        # 存储model
        theta_v = theta_v.T
        theta_v = theta_v.tolist()
        with open('../data/model/model_bgd_{0}.txt'.format(qid), 'wt', encoding='utf-8') as f:
            f.write('{0}\n'.format(theta_v))
        print('mse:{0}'.format(error1))
        print('{0} over, times: {1}'.format(qid, time))


if __name__ == '__main__':
    batch_gradient_descent()

# mae 2.1517221822971626
