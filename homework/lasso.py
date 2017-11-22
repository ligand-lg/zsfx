from sklearn.linear_model import Lasso
from homework import conf_hw
import numpy as np
import time

if __name__ == '__main__':
    fout = open('../data/predict_lasso.txt', 'wt', encoding='utf-8')
    for qid in range(201, 251):
        print(qid)
        time_in = time.time()
        qid = str(qid)
        X_train, Y_train = conf_hw.read_train(qid)
        X_test, Y_test = conf_hw.read_test(qid)
        # 线性回归截距
        X_train = np.column_stack((X_train, np.ones((X_train.shape[0], 1))))
        X_test = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))

        y_train = np.array(Y_train).ravel()
        y_test = np.array(Y_test).ravel()

        model = Lasso(alpha=0.1)
        model.fit(X_train, Y_train)
        y_hat = model.predict(X_test)

        i = 0
        for relationship in conf_hw.coll_test.find({'query_id': qid}):
            article_id = relationship['article_id']
            id_code = relationship['id_code']
            score = y_hat[i]
            fout.write(
                '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
            i += 1
        mae = sum(abs(y_hat-y_test)) / X_test.shape[0]
        print('time: {0}'.format(time.time()- time_in))
    fout.close()
    conf_hw.MAE('lasso')

# MAE: 2.4338371069187454