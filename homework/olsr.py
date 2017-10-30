



# 1. 预处理(nltk.download('stopwords'), nltk.download('wordnet')
#   1.1 标点符号、去除停止词、xml转义符号、非打印字符、url、长度小于2的字符。
#   1.2 porter stemmer 词干提取/ wordnet lemmatizer 词形还原


import nltk
from pymongo import MongoClient
mongo_client = MongoClient()
coll_articles = mongo_client.nlp.articles
coll_documents = mongo_client.nlp.documents
coll_querys = mongo_client.nlp.querys
coll_relation = mongo_client.nlp.relation


query_clean_path = './data/query_simple.txt'
document_clean_path = './data/documents_simple.tx'


def pre_process():
    stop_words = nltk.corpus.stopwords.words('english')
    # pick out sequences of alphanumeric characters as tokens and drop everything else
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

    def clean(s):
        return [word for word in tokenizer.tokenize(s) if word not in stop_words and len(word) > 2]

    def lemmatize(words):
        return [wordnet_lemmatizer.lemmatize(word) for word in words]

    with open(query_clean_path, 'wt') as f:
        for item in coll_querys.find({}):
            new_item = dict()
            new_item['id'] = item['_id']
            new_item['query'] = lemmatize(clean(item['query']))
            f.write('{0}\n'.format(new_item))

    with open(document_clean_path, 'wt') as f:
        for item in coll_documents.find({}):
            new_item = dict()
            new_item['id'] = item['_id']
            new_item['title'] = lemmatize(clean(item['title']))
            f.write('{0}\n'.format(new_item))

if __name__ == '__main__':
    pre_process()



