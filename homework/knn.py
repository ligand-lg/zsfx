import math
from homework import conf_hw
#import conf_hw
import time
import matplotlib.pyplot as plt


def knn(x, X_train, Y_train, k, distance_type='o', knn_type='naive'):
    # 算距离
    distances = []
    index = 0
    for x_train in X_train:
        if distance_type == 'o':
            diff = x_train - x
            dist = math.sqrt((diff * diff.T)[0, 0])
        elif distance_type == 'm':
            diff = abs(x_train - x)
            dist = diff.sum()
        else:
            e = Exception("unknown distance_type")
            raise e
        distances.append((dist, Y_train[index, 0]))
        index += 1
    #  排序
    distances.sort()
    y_hat = None
    # 按前面k个赋值
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
        if max_diff == 0:
            #print('the same diff, reset to naive knn')
            return knn(x, X_train, Y_train, k, distance_type=distance_type, knn_type='naive')
        scores = 0
        for dist, y in distances[:k]:
            w = (max_dist-dist)/max_diff
            W1.append(w)
        # 归一化
        sum_w1 = sum(W1)
        W = [w/sum_w1 for w in W1]
        index = 0
        for _, y in distances[:k]:
            scores += W[index] * y
            index += 1
        y_hat = 1 if scores > 0.5 else 0
    else:
        e = Exception("unknown knn type")
        raise e
    return y_hat


def test(k, distance_type, knn_type):
    with open('../data/predict_knn.txt', 'wt', encoding='utf-8') as fout:
        for qid in range(201, 251):
            qid = str(qid)
            y_hats = []
            X_train, Y_train = conf_hw.read_train(qid, type='class')
            X_test, Y_test = conf_hw.read_test(qid, type='class')
            for x in X_test:
                y_hat = knn(x, X_train, Y_train, k, distance_type=distance_type, knn_type=knn_type)
                y_hats.append(y_hat)
            i = 0
            for relationship in conf_hw.coll_class_test.find({'query_id':qid}):
                article_id = relationship['article_id']
                id_code = relationship['id_code']
                score = y_hats[i]
                fout.write(
                    '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
                i += 1
    return conf_hw.MAE('knn')


def select_params():
    k_range = list(range(1, 20))
    X_trains = dict()
    Y_trains = dict()
    X_tests = dict()
    Y_tests = dict()
    for qid in range(201, 251):
        qid = str(qid)
        e1, e2 = conf_hw.read_train(qid, type='class')
        X_trains[qid] = e1
        Y_trains[qid] = e2
        e1, e2 = conf_hw.read_test(qid, type='class')
        X_tests[qid] = e1
        Y_tests[qid] = e2
    start = time.time()
    result = dict()
    for dis_type in ['m', 'o']:
        for knn_type in ['improve', 'naive']:
            error_ratios_in_k = []
            for k in k_range:
                errors = 0
                rights = 0
                start_in = time.time()
                for qid in range(201, 211):
                    qid = str(qid)
                    X_train = X_trains[qid]
                    Y_train = Y_trains[qid]
                    X_test = X_tests[qid]
                    Y_test = Y_tests[qid]
                    i = 0
                    for x in X_test:
                        y_hat = knn(x, X_train, Y_train, k, distance_type=dis_type, knn_type=knn_type)
                        if y_hat == Y_test[i]:
                            rights += 1
                        else:
                            errors += 1
                        i += 1
                error_ratio = errors / (errors + rights)
                print("test end :{0}".format(time.time() - start_in))
                print("error ratio: {0}".format(error_ratio))
                error_ratios_in_k.append(error_ratio)
            result[dis_type + '_' + knn_type] = error_ratios_in_k
    print(result)
    for k, v in result.items():
        plt.plot(k_range, v, label=k)
    plt.xlabel('k')
    plt.ylabel('error ratio')
    plt.legend()
    plt.show()
    # error_ration = 0.27,k=4, o_imporve


if __name__ == '__main__':
    test(4, distance_type='o', knn_type='improve')