from homework import conf_hw
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import matplotlib.pyplot as plt


def select_paras():
    data = []
    for qid in range(201, 251):
        x_train, y_train = conf_hw.read_train(qid, type='class')
        x_train = np.array(x_train)
        y_train = np.array(y_train).ravel()
        x_test, y_test = conf_hw.read_test(qid, type='class')
        x_test = np.array(x_test)
        y_test = np.array(y_test).ravel()
        data.append((x_train, x_test, y_train, y_test))

    num_of_trees = list(range(5, 15))
    cost_times = []
    error_ration = []
    for num_of_tree in num_of_trees:
        start_time = time.time()
        error_num = 0.0
        total = 0.0

        for i in range(len(data)):
            x_train, x_test, y_train, y_test = data[i]

            cls = RandomForestClassifier(n_estimators=num_of_tree,
                                         criterion='entropy',
                                         max_features='sqrt',
                                         bootstrap=True,
                                         n_jobs=-1)
            cls.fit(x_train, y_train)
            y_hat = cls.predict(x_test)
            for j in range(len(y_hat)):
                if y_hat[j] != y_test[j]:
                    error_num += 1
            total += len(y_hat)
        cost_times.append(time.time() - start_time)
        error_ration.append(error_num/total)

    min1 = min(cost_times)
    min2 = min(error_ration)
    max1 = max(cost_times)
    max2 = max(error_ration)
    for i in range(len(cost_times)):
        cost_times[i] = cost_times[i] - min1 / max1 - min1
        error_ration[i] = error_ration[i] - min2 / max2 - min2

    plt.plot(num_of_trees, cost_times, label='cost_time')
    plt.plot(num_of_trees, error_ration, label='error_ration')
    plt.xlabel('number of trees')
    plt.ylabel('scala')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    select_paras()
