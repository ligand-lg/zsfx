#coding=utf-8

# 1. 预处理(nltk.download('stopwords'), nltk.download('wordnet')
#   1.1 标点符号、去除停止词、xml转义符号、非打印字符、url、长度小于2的字符。
#   1.2 porter stemmer 词干提取/ wordnet lemmatizer 词形还原
from collections import defaultdict
import ast
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
    docs = defaultdict(str)
    with open(url, 'rt', encoding='utf-8') as f:
        for line in f:
            docs.update(json.loads(line))
    return docs


def train(articles):
    model = []
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
        try:
            pinv_data_mat = np.linalg.pinv(data_mat, 0.01)
        except Exception as e:
            print(e)
            continue
        params_mat = pinv_data_mat * score_mat
        params_mat = params_mat.T
        params_mat = params_mat.tolist()
        model.append((qid, params_mat[0], word_list))
        with open('../data/model_{0}.txt'.format(qid), 'wt', encoding='utf-8') as f:
            f.write('{0}\n{1}\n'.format(params_mat, word_list))
        print('{0} over'.format(qid))
    return model


def predict(article, word_list, params_mat):
    # 向量化article
    word_list_len = len(word_list)
    vector = np.mat(np.zeros((1, word_list_len+1)))
    # 截距
    vector[0, -1] = 1
    for word in article:
        # word不在词表时，默认倒数第二列
        index = -2
        if word in word_list:
            index = word_list.index(word)
        vector[0, index] = 1
    # 计算score
    score = vector*params_mat 
    return score[0, 0]


def test(aritcles):
    with open('../data/predict.txt', 'wt', encoding='utf-8') as fout:
        for i in range(201, 251):
            qid = str(i)
            # 加载模型
            with open('../data/model_{0}.txt'.format(qid), 'rt', encoding='utf-8') as f:
                params_mat = f.readline().replace('\n', '')
                params_mat = ast.literal_eval(params_mat)
                params_mat = np.mat(params_mat).T
                word_list = f.readline().replace('\n', '')
                word_list = ast.literal_eval(word_list)
            # 读取数据
            for relationship in coll_test.find({'query_id': qid}):
                article_id = relationship['article_id']
                article = articles[article_id]
                score = predict(article, word_list, params_mat)
                id_code = relationship['id_code']
                fout.write('{0} Q0 {1} {2} {3} Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'.format(qid, article_id, id_code, score))



if __name__ == '__main__':
    
    articles = read_doucments()
    test(articles)
    #model = train(articles)
    #pre_process()
    #r = read_doucments()
    #print(len(r), r[0])


# MAE:0.01 -> 2.287481465126515

