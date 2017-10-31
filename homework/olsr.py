#coding=utf-8



# 1. 预处理(nltk.download('stopwords'), nltk.download('wordnet')
#   1.1 标点符号、去除停止词、xml转义符号、非打印字符、url、长度小于2的字符。
#   1.2 porter stemmer 词干提取/ wordnet lemmatizer 词形还原

import re
import nltk
from pymongo import MongoClient
import json
import numpy as np
mongo_client = MongoClient()
coll_articles = mongo_client.nlp.articles
coll_documents = mongo_client.nlp.documents
coll_querys = mongo_client.nlp.querys
coll_train = mongo_client.nlp.train_relationship
coll_test = mongo_client.nlp.test_relationship


query_clean_path = '../data/query_simple.txt'
document_clean_path = '../data/documents_simple.tx'


def pre_process():
    # xml转义符 &...;
    xml_symbols_re = re.compile(r'&.*?;')
    stop_words = nltk.corpus.stopwords.words('english')
    # pick out sequences of alphanumeric characters as tokens and drop everything else
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    

    def clean(s):
        s = xml_symbols_re.sub(lambda m: ' ', s)
        return [word for word in tokenizer.tokenize(s) if word not in stop_words and len(word) > 1]

    def lemmatize(words):
        return [wordnet_lemmatizer.lemmatize(word) for word in words]

    with open(query_clean_path, 'wt') as f:
        for item in coll_querys.find({}):
            new_item = dict()
            new_item[item['_id']] = lemmatize(clean(item['query']))
            s_tmp = '{0}\n'.format(new_item)
            f.write(s_tmp.replace("'", '"'))

    with open(document_clean_path, 'wt', encoding='utf-8') as f:
        for item in coll_documents.find({}):
            new_item = dict()
            new_item[item['_id']] = lemmatize(clean(item['title']))
            s_tmp = '{0}\n'.format(new_item)
            f.write(s_tmp.replace("'", '"'))


def read_doucments(url=document_clean_path):
    docs = dict()
    with open(url, 'rt', encoding='utf-8') as f:
        for line in f:
            docs.update(json.loads(line))
    return docs


def train(articles):
    model = []
    for i in range(201, 202):
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
        # 使用空格来表示其他词
        word_list.append(" ")
        word_list_len = len(word_list)

        # one-hot向量化article
        data_mat = np.zeros((len(relation_articles), word_list_len+1))
        print(data_mat.shape)
        #最后一列为截距
        data_mat[:, -1] = 1
        i = 0
        for article in relation_articles:
            for word in article:
                if word in word_list:
                    index = word_list.index(word)
                    data_mat[i, index] = 1
            i += 1

        score_mat = np.mat(scores).T
        print('vecting over')
        # 对data_mat 进行SVD分解
        #u, sigma, vt = np.linalg.svd(data_mat)
        #pinv_data_mat = vt.T * pinv_sigma * u.T
        pinv_data_mat = np.linalg.pinv(data_mat)
        params_mat = pinv_data_mat * score_mat
        params_mat = params_mat.T
        params_mat = list(params_mat[0, :])
        model.append((qid, params_mat, word_list))
    return model


def test(model, aritcles):




if __name__ == '__main__':
    articles = read_doucments()
    model = train(articles)
    with open('../data/model.txt', 'wt', encoding='utf-8') as f:
        for qid, params_mat, word_list in model:
            f.write('{0}\n{1}\n{2}\n'.format(qid, params_mat, word_list))


    #pre_process()
    #r = read_doucments()
    #print(len(r), r[0])




