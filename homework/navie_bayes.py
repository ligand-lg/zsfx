import conf_hw
import math
import numpy as np
from sklearn.naive_bayes import  GaussianNB, MultinomialNB, BernoulliNB


class NaiveBayes:
    def __init__(self, lam=1):
        self.lam = lam
        self.prior = []
        self.conditional_0 = []
        self.conditional_1 = []

    def fit(self, x_train, y_train):
        # 计算先验 P(Y)
        self.prior.append(list(y_train).count(0) / len(y_train))
        self.prior.append(list(y_train).count(1) / len(y_train))

        # 计算条件概率 P(X|Y) = p(x1|Y) * p(x2|Y) * p(x3|Y)....
        for y in [0,1]:
            select_simples = x_train[y_train[:] == y, :]
            for xi in select_simples.T:
                zeros = (list(xi).count(0) + self.lam) / (len(xi) + self.lam * 2)
                ones = (list(xi).count(1) + self.lam) / (len(xi) + self.lam * 2)
                if y == 0:
                    self.conditional_0.append([zeros, ones])
                else:
                    self.conditional_1.append([zeros, ones])

    def predict(self, x_test):
        y_hat = []
        for x in x_test:
            posterior = [0, 0]
            for y in [0, 1]:
                conditional = self.conditional_0 if y == 0 else self.conditional_1
                for xi, conditional_xi in zip(x, conditional):
                    posterior[y] += math.log(conditional_xi[int(xi)])
                posterior[y] += math.log(self.prior[y])
            y_hat.append(0 if posterior[0] > posterior[1] else 1)
        return y_hat

if __name__ == '__main__':
    alr = 'naive_bayes'
    print('start....')
    error_num = 0
    total_num = 0
    fout = open('../data/predict_{0}.txt'.format(alr), 'wt', encoding='utf-8')
    for qid in range(201, 251):
        x_train, y_train = conf_hw.read_train(qid, type='class')
        x_train = np.array(x_train)
        y_train = np.array(y_train).ravel()
        x_test, y_test = conf_hw.read_test(qid, type='class')
        x_test = np.array(x_test)
        y_test = np.array(y_test).ravel()
        if alr == 'naive_bayes':
            clf = NaiveBayes()
        elif alr == 'gaussian_bayes':
            clf = GaussianNB()
        elif alr == 'multinomial_bayes':
            clf = MultinomialNB()
        elif alr == 'aode':
            clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_hat = clf.predict(x_test)
        # error ration
        for r in range(len(y_hat)):
            if y_hat[r] != y_test[r]:
                error_num += 1
        total_num += len(y_hat)

        # write file
        i = 0
        for relationship in conf_hw.coll_class_test.find({'query_id': str(qid)}):
            article_id = relationship['article_id']
            id_code = relationship['id_code']
            score = y_hat[i]
            fout.write(
                '{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
            i += 1
    fout.close()
    print('error_ration: {0}'.format(float(error_num) / float(total_num)))
    # naive bayes: 0.21048472075869337
    # gaussian_bayes: 0.28793466807165435
    # multinomail_bayes: 0.2363013698630137
    # bernoulli_bayes: 0.21048472075869337