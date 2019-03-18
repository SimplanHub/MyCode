#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : FileResearch.py
# @Time    : 2018/11/06
# @Author  : Simplan (Simplan@aliyun.com)

import jieba.posseg as pseg
import codecs
from gensim import corpora
from gensim.summarization import bm25
import os
import re

stop_words = '/Code/Python/FileResearch_PY3/stop_words.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]

stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


# ----------------分词函数----------------
def tokenization(filename):
    result = []
    with open(
            '/Code/Python/FileResearch_PY3/articles/' + filename,
            'r',
            encoding='UTF-8') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


# ---------------读文件-------------------
corpus = []
dirname = '/Code/Python/FileResearch_PY3/articles'
filenames = []
pattern = re.compile(r'[\u4e00-\u9fa5]*.txt')
for root, dirs, files in os.walk(dirname):
    for f in files:
        if re.match(pattern, f):
            corpus.append(tokenization(f))
            filenames.append(f)

dictionary = corpora.Dictionary(corpus)
# print(len(dictionary))
# -------------词排序------------------
doc_vectors = [dictionary.doc2bow(text) for text in corpus]
vec1 = doc_vectors[0]
vec1_sorted = sorted(vec1, key=lambda y: y, reverse=True)
# print(len(vec1_sorted))
'''
for term, freq in vec1_sorted[:5]:
    print(dictionary[term])
'''
# -------------BM25计算词权重------------------
bm25Model = bm25.BM25(corpus)
average_idf = sum(
    map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(
        bm25Model.idf.keys())
# -------------根据权重再次排序------------------
query_str = '高血压 患者'
query = []
for word in query_str.strip().split():
    query.append(word)
scores = bm25Model.get_scores(query, average_idf)
# scores.sort(reverse=True)
print(scores)
# -------------得到排序最高文件的序号------------------
idx = scores.index(max(scores))
print(idx)
# -------------得到排序最高文件的文件名------------------
fname = filenames[idx]
print(fname)
# -------------打开文件------------------
with open(
        '/Code/Python/FileResearch_PY3/articles/' + fname, 'r',
        encoding='UTF-8') as f:
    print(f.read())
