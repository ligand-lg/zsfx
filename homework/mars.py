# coding:utf-8
#from homework import conf_hw
import conf_hw
import numpy as np
from pyearth import Earth
from sklearn.decomposition import PCA
import time

with open('../data/predict_mars.txt', 'wt', encoding='utf-8') as fout:
    start = time.time()
    start_in = time.time()
    for qid in range(201, 251):
        print(qid)
        x_train, y_train = conf_hw.read_train(qid)
        x_test, y_test = conf_hw.read_test(qid)
        # 截距
        x_train = np.column_stack((x_train, np.ones((x_train.shape[0], 1))))
        x_test = np.column_stack((x_test, np.ones((x_test.shape[0], 1))))
        # PCA降维
        print ('PCA...')
        start_in = time.time()
        pca = PCA(n_components=100)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        print('PCA over {0}'.format(time.time()-start_in))
        start_in = time.time()

        print('mars fiting...')
        model = Earth()
        model.fit(x_train, y_train)
        print('fiting over {0}'.format(time.time()-start_in))

        y_hat = model.predict(x_test)
        i = 0
        for relationship in conf_hw.coll_test.find({'query_id': str(qid)}):
            article_id = relationship['article_id']
            id_code = relationship['id_code']
            score = y_hat[i]
            fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
            i += 1
    conf_hw.MAE('mars')
    end = time.time()
    print('total : {0}'.format((end-start)/60))
