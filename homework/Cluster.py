import numpy as np
import conf_hw


def L2(v1, v2):
    diff = np.array(v1) - np.array(v2)
    return np.sqrt(sum(diff ** 2))


class KMeans(object):
    data = None
    row = None
    column = None
    end_with = None
    count = 0

    def init_labels(self):
        return [np.random.randint(0, 2) for _ in range(self.row)]

    def class_to_lab(self, c0, c1):
        labels = []
        for d in self.data:
            dist_0 = L2(d, c0)
            dist_1 = L2(d, c1)
            label = None
            if dist_0 < dist_1:
                label = 0
            else:
                label = 1
            labels.append(label)
        return labels

    def lab_to_class(self, labels):
        c0 = np.zeros(self.column)
        c1 = np.zeros(self.column)
        zeros = 0
        for i in range(self.row):
            if labels[i] == 0:
                c0 += self.data[i]
                zeros += 1
            else:
                c1 += self.data[i]
        c0 /= zeros
        c1 /= self.row - zeros
        return c0, c1

    def cluster(self, data):
        self.data = data
        self.row, self.column = data.shape
        init_labels = self.init_labels()
        return self._cluster(init_labels, 100)

    def _cluster(self, labels, left_times):
        self.count += 1
        if left_times > 0:
            c0, c1 = self.lab_to_class(labels)
            n_labels = self.class_to_lab(c0, c1)
            if labels != n_labels:
                self._cluster(n_labels, left_times-1)
            else:
                self.end_with = '收敛'
        else:
            self.end_with = '超时'
        return labels


if __name__ == '__main__':
    error = 0
    total = 0
    alr = 'k-means'
    fout = open('../data/predict_{0}.txt'.format(alr), 'wt', encoding='utf-8')
    for qid in range(201, 251):
        data, y_test = conf_hw.read_train(qid, type='class')
        test = KMeans()
        y_hat = test.cluster(np.array(data))
        for r in range(len(y_hat)):
            if y_hat[r] != y_test[r]:
                error += 1
        total += len(y_hat)

        # write file
        i = 0
        for relationship in conf_hw.coll_class_train.find({'query_id': str(qid)}):
            article_id = relationship['article_id']
            id_code = relationship['id_code']
            score = y_hat[i]
            fout.write(
                '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
            i += 1
    fout.close()

    mae = error / total
    if mae > 0.5:
        mae = 1 - mae
    print("MAE: {0}".format(mae))
    print(test.count)

