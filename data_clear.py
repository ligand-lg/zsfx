# main: documents.title/body query.description/query/subtopic
from conf import *

def clear_str(str):
    if len(str) == 0:
        return str
    aim_symbols = [' ', '\n', '\t']
    if str[0] in aim_symbols:
        return clear_str(str[1:])
    if str[-1] in aim_symbols:
        return clear_str(str[:-1])
    return str


def clear_query():
    for item in coll_querys.find({}):
        item['description'] = clear_str(item['description'])
        item['query'] = clear_str(item['query'])
        subtopics = []
        for subtopic in item['subtopic']:
            subtopics.append(clear_str(subtopic))
        item['subtopic'] = subtopics
        coll_querys.save(item)


def clear_document():
    for item in coll_documents.find({}):
        item['title'] = clear_str(item['title'])
        item['body'] = clear_str(item['body'])
        coll_documents.save(item)


# clear_query()
clear_document()