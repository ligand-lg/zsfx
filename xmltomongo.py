from conf import *
import re

# article string --> article dict
# state_code: 0 ok
#             1 title parse error
#             2 body parse error
#             3 title and body parse error


def parse_article(article):
    id_re = r'<article_id>(.*?)</article_id>'
    title_re = r'<title>(.*?)</title>'
    body_re = r'<body>(.*?)</body>'
    id_find = re.findall(id_re, article, re.DOTALL)
    title_find = re.findall(title_re, article, re.DOTALL)
    body_find = re.findall(body_re, article, re.DOTALL)
    item = dict()
    item['_id'] = ''
    item['title'] = ''
    item['body'] = ''
    state_code = 0

    item['_id'] = id_find[0]
    if len(title_find) == 1:
        item['title'] = title_find[0]
    else:
        state_code += 1
        print('-------wrong title------\n' + article + '\n---------------')
    if len(body_find) == 1:
        item['body'] = body_find[0]
    else:
        state_code += 2
        print('-------wrong body------\n' + article + '\n---------------')
    item['state_code'] = state_code
    return item


# articles collection --> documents collection
def parse_doc():
    print("start parse")
    i = 0
    for article in coll_articles.find({}):
        i += 1
        item = parse_article(article['article'])
        item['content'] = article['article']
        coll_documents.save(item)
    print("total:{0}".format(i))


def article_mongo(url):
    print('reading.')
    with open(url, 'rt', encoding='utf-8') as f:
        content = f.read()
    print('read done!')

    article_re = r'<article>(.*?)</article>'
    for art in re.findall(article_re, content, re.DOTALL):
        item = dict()
        item['article'] = art
        coll_articles.insert_one(item)


def parse_querys(url):
    print('reading.')
    with open(url, 'rt', encoding='utf-8') as f:
        content = f.read()
    print('read done!')
    topic_re = r'<topic>(.*?)</topic>'
    qid_re = r'<qid>(.*?)</qid>'
    query_re = r'<query>(.*?)</query>'
    description_re = r'<description>(.*?)</description>'
    subtopic_re = r'<subtopic>(.*?)</subtopic>'
    for topic in re.findall(topic_re, content, re.DOTALL):
        item = dict()
        item['_id'] = re.findall(qid_re, topic, re.DOTALL)[0]
        item['query'] = re.findall(query_re, topic, re.DOTALL)[0]
        item['description'] = re.findall(description_re, topic, re.DOTALL)[0]
        item['subtopic'] = re.findall(subtopic_re, topic, re.DOTALL)
        coll_querys.insert(item)


def parse_relation(url):
    print('reading..')
    with open(url, 'rt', encoding='utf-8') as f:
        for line in f:
            item = dict()
            line = line.replace('\n', '')
            split_line = line.split(" ")
            item['query_id'] = split_line[0]
            item['article_id'] = split_line[2]
            item['_id'] = item['query_id'] + item['article_id']
            item['id_code'] = split_line[3]
            item['score'] = split_line[4]
            coll_relation.save(item)



#parse_relation(relation_path)
#parse_doc()
#article_mongo(documents_path)

#parse_querys(querys_path)