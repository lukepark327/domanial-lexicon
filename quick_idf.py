import os
import json
from math import log


years = list(range(2000, 2018))  # years range: 2000 ~ 2017
keywords = ['불행', '행복']

N = len(years) * len(keywords)  # total number of document in the corpus, N = |D|

groups = []

for year in years:
    for keyword in keywords:
        IN_PATH = './corpus' + '/' + str(year)

        # 파일 읽기
        with open(IN_PATH + '/' + keyword + '.txt', 'r', encoding='UTF-8') as f:
            text = f.read()
            words = text.split()
            print("OPEN: " + IN_PATH + '/' + keyword + '.txt')

        words = list(set(words))  # 중복 제거
        for target in keywords:
            if target in words:
                words.remove(target)  # 키워드 제거

        groups.append(words)

"""
idf 계산
"""
idf_dict = {}

for group in groups:
    for word in group:
        if word in idf_dict:
            idf_dict[word] += 1
        else:
            idf_dict[word] = 1

# tf-idf
for key in idf_dict:
    idf_dict[key] = log(N / idf_dict[key])

# idf.json 업데이트
OUT_PATH = './tfidf'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

with open('./tfidf/idf.json', 'w') as wf:
    wf.write(json.dumps(idf_dict))
