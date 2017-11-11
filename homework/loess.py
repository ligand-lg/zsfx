# coding:utf-8

import numpy as np
#from homework import conf_hw
import conf_hw
from sklearn.decomposition import PCA
import time

def loess(test_point, x_mat, y_mat):
    m, n = x_mat.shape
    weight_mat = np.mat(np.eye(m))
    # tri-cube weight function
    dists = []
    for i in range(m):
        diff_mat = test_point - x_mat[i, :]
        dists.append((diff_mat*diff_mat.T)[0])
    max_dist = max(dists)
    for j in range(m):
        weight_mat[j, j] = (1-(dists[j]/max_dist)**3)**3
    # pseudo-inverse
    while True:
        try:
            pinv = np.linalg.pinv(x_mat.T * weight_mat * x_mat, 0.01)
            break;
        except Exception as e:
            print(e)
            print('again')

    params_mat = pinv * x_mat.T * weight_mat * y_mat
    return test_point * params_mat


if __name__ == '__main__':
    start = time.time()
    with open('../data/predict_loess.txt', 'wt', encoding='utf-8') as fout:
        for qid in range(201, 251):
            print(qid)
            train_data, train_lab = conf_hw.read_train(qid)
            test_data, test_lab = conf_hw.read_test(qid)
            test_r, test_c = test_data.shape
            # 截距
            train_data = np.column_stack((train_data, np.ones((train_data.shape[0], 1))))
            test_data = np.column_stack((test_data, np.ones((test_data.shape[0], 1))))
            # PCA 降维
            print('PCA...')
            start_in = time.time()
            pca = PCA(n_components=300)
            train_data = pca.fit_transform(train_data)
            test_data = pca.transform(test_data)
            print('PCA: {0}'.format(time.time()-start_in))
            start_in = time.time()
            print('test...')
            predicts = []
            for r in range(test_r):
                predict_lab = loess(test_data[r, :], train_data, train_lab)
                predicts.append(predict_lab[0, 0])
                print(r)
            print('test end :{0}'.format(time.time()-start_in))
            # write to file
            i = 0
            for relationship in conf_hw.coll_test.find({'query_id': str(qid)}):
                article_id = relationship['article_id']
                id_code = relationship['id_code']
                score = predicts[i]
                fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
                i += 1
    conf_hw.MAE('loess')
# MAE: 13.892704692969026
# Total time: 3651 Min