# coding:utf-8

import numpy as np
from homework import conf_hw
from sklearn.decomposition import PCA
#import conf


def loess(test_point, x_mat, y_mat):
    m, n = x_mat
    weight_mat = np.mat(np.eye(m))
    # tri-cube weight function
    dists = []
    for i in range(m):
        diff_mat = test_point - x_mat[i, :]
        dists.append((diff_mat*diff_mat.T)[0, 0])
    max_dist = max(dists)
    for i in range(m):
        weight_mat[i, i] = (1-(dists[i]/max_dist)**3)**3
    # pseudo-inverse
    pinv = np.linalg.pinv(x_mat.T * weight_mat * x_mat, 0.01)
    params_mat = pinv * x_mat.T * weight_mat * y_mat
    return test_point * params_mat


if __name__ == '__main__':
    with open('../data/predict_loess.txt', 'wt', encoding='utf-8') as fout:
        for qid in range(201, 202):
            train_data, train_lab = conf_hw.read_train(qid)
            test_data, test_lab = conf_hw.read_test(qid)
            test_r, test_c = test_data.shape
            train_data = np.column_stack((train_data, np.ones((train_data.shape[0], 1))))
            test_data = np.column_stack((test_data, np.ones((test_data.shape[0], 1))))
            # PCA 降维
            pca = PCA(n_components=300)
            train_data = pca.fit_transform(train_data)
            test_data = pca.transform(test_data)

            predicts = []
            for r in range(test_r):
                predict_lab = loess(test_r[r, :], train_data, train_lab)
                predicts.append(predict_lab[0, 0])
                print(r)

            # write to file
            i = 0
            for relationship in conf_hw.coll_test.find({'query_id': str(qid)}):
                article_id = relationship['article_id']
                id_code = relationship['id_code']
                score = predicts[i]
                fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
                i += 1
    conf_hw.MAE('loess')