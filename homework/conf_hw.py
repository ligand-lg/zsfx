# coding:utf-8
from pymongo import MongoClient
from collections import defaultdict
import json
import numpy as np
import ast
import sys
mongo_client = MongoClient()
coll_articles = mongo_client.nlp.articles
coll_documents = mongo_client.nlp.documents
coll_querys = mongo_client.nlp.querys
coll_train = mongo_client.nlp.train_relationship
coll_test = mongo_client.nlp.test_relationship


query_clean_path = '../data/query_simple.txt'
document_clean_path = '../data/documents_simple.tx'


def read_doucments(url=document_clean_path):
    docs = defaultdict(str)
    with open(url, 'rt', encoding='utf-8') as f:
        for line in f:
            docs.update(json.loads(line))
    return docs


def one_hot_encoding():
    articles = read_doucments(document_clean_path)
    for i in range(201, 251):
        qid = str(i)
        relation_articles = list()
        scores = list()
        # 读取数据
        for relationship in coll_train.find({'query_id': qid}):
            relation_articles.append(articles[relationship['article_id']])
            scores.append(float(relationship['score']))

        # 建立词表
        word_list = set()
        for article in relation_articles:
            word_list.update(article)
        word_list = list(word_list)
        word_list_len = len(word_list)
        # 训练集
        mat_shape = (len(relation_articles), word_list_len)
        with open('../data/one-hot/train_{0}.txt'.format(qid), 'wt', encoding='utf-8') as fout:
            fout.write('{0}\n'.format(mat_shape))
            for article in relation_articles:
                indexs = set()
                for word in article:
                    if word in word_list:
                        indexs.add(word_list.index(word))
                fout.write('{0}\n'.format(list(indexs)))
        with open('../data/one-hot/train_score_{0}.txt'.format(qid), 'wt', encoding='utf-8') as fout:
            fout.write(str(scores))

        # 测试集
        test_relation_articles = list()
        test_scores = list()
        for test_relation in coll_test.find({'query_id': qid}):
            test_relation_articles.append(articles[test_relation['article_id']])
            test_scores.append(float(test_relation['score']))

        test_shape = (len(test_relation_articles), word_list_len)
        with open('../data/one-hot/test_{0}.txt'.format(qid), 'wt', encoding='utf-8') as fout:
            fout.write('{0}\n'.format(test_shape))
            for article in test_relation_articles:
                indexs = set()
                for word in article:
                    if word in word_list:
                        indexs.add(word_list.index(word))
                fout.write('{0}\n'.format(list(indexs)))
        with open('../data/one-hot/test_score_{0}.txt'.format(qid), 'wt', encoding='utf-8') as fout:
            fout.write(str(test_scores))


def read_one_hot(data_set, qid):
    data_url = '../data/one-hot/{0}_{1}.txt'.format(data_set, qid)
    score_url = '../data/one-hot/{0}_score_{1}.txt'.format(data_set, qid)
    with open(data_url, 'rt', encoding='utf-8') as fin:
        shape = ast.literal_eval(fin.readline())
        data_mat = np.mat(np.zeros(shape))
        i = 0
        for indexs_str in fin:
            indexs = ast.literal_eval(indexs_str)
            data_mat[i, indexs] = 1
            i += 1
    with open(score_url, 'rt', encoding='utf-8') as fin:
        scores = ast.literal_eval(fin.readline())
    scores_mat = np.mat(scores).T
    return data_mat, scores_mat


def read_train(qid):
    return read_one_hot('train', qid)


def read_test(qid):
    return read_one_hot('test', qid)


def test(algori, start, end):
    with open('../data/predict_{0}.txt'.format(algori), 'wt', encoding='utf-8') as fout:
        for qid in range(start, end):
            # 加载模型
            with open('../data/model/model_{0}_{1}.txt'.format(algori, qid), 'rt', encoding='utf-8') as fin:
                params_mat = ast.literal_eval(fin.readline())
                params_mat = np.mat(params_mat).T
                if algori=='logistic':
                    max, min = ast.literal_eval(fin.readline())
            # 加载测试集
            test_data, test_score = read_test(qid)
            # 添加截距
            row, c = test_data.shape
            test_data = np.column_stack((test_data, np.ones((row,1))))
            predict_score = test_data * params_mat

            # 读取数据
            i = 0
            for relationship in coll_test.find({'query_id': str(qid)}):
                article_id = relationship['article_id']
                id_code = relationship['id_code']
                score = predict_score[i, 0]
                # logisitc 分数映射
                if algori == 'logistic':
                    score = score*100
                fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))
                i += 1
            
    return MAE(algori)


def MAE(algori):
    with open('/home/fry/zsfx/data/predict_{0}.txt'.format(algori), 'rt', encoding='utf-8') as fin:
        sum_error = 0.0
        n = 0.0
        for line in fin:
            line = line.replace('\n', '')
            split_line = line.split(' ')
            predict_label = float(split_line[4])
            id = split_line[0] + split_line[2]
            item = coll_test.find({'_id': id})[0]
            label = float(item['score'])
            error = abs(predict_label - label)
            sum_error += error
            n += 1
        print("MAE: {0}".format(sum_error / n))


if __name__ == '__main__':
    start = 201
    end = 251
    algori = sys.argv[1]
    if (len(sys.argv) == 3):
        end = int(sys.argv[2])
    elif(len(sys.argv) == 4):
        start = int(sys.argv[2])
        end = int(sys.argv[3])
    test(algori, start, end)
    #print('staring one-hot encoding...')
    #one_hot_encoding()
    #print('over')
