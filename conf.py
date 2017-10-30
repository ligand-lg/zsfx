from pymongo import MongoClient
documents_path = '/home/fry/NLP/data/documents.txt'
querys_path = '/home/fry/NLP/data/querys.xml'
relation_path = '/home/fry/NLP/data/Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res'

mongo_client = MongoClient()
coll_articles = mongo_client.nlp.articles
coll_documents = mongo_client.nlp.documents
coll_querys = mongo_client.nlp.querys
coll_relation = mongo_client.nlp.relation