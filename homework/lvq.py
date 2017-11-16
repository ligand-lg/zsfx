import glvq
import conf_hw
import numpy as np


def lvq():
    error = 0
    total = 0
    for qid in range(201, 251):
        qid = str(qid)
        X_train, Y_train = conf_hw.read_train(qid, type='class')
        X_test, Y_test = conf_hw.read_test('201', type='class')
        model = glvq.GlvqModel()
        model.fit(X_train, np.array(Y_train).ravel())
        y_hats = model.predict(X_test)
        y_hats = np.mat(y_hats).reshape(-1, 1)
        error += abs(y_hats-Y_test).sum()
        total += X_test.shape[0]
        i = 0
        with open('../data/predict_lvq.txt', 'at', encoding='utf-8') as fout:
            for relationship in conf_hw.coll_class_test.find({'query_id': qid}):
                article_id = relationship['article_id']
                id_code = relationship['id_code']
                score = y_hats[i, 0]
                fout.write(
                    '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
                i += 1
    print('error ratio: {0}'.format(error/total))

if '__name__' == '__main__':
    lvq()