import os
import operator
import json
from math import log


def idf(t, N):
    years = list(range(2000, 2018))  # years range: 2000 ~ 2017
    keywords = ['불행', '행복']

    DF = 0  # number of documents where the term t appears

    for year in years:
        for keyword in keywords:
            IN_PATH = './corpus' + '/' + str(year)

            f = open(IN_PATH + '/' + keyword + '.txt', 'r', encoding='UTF-8')  # 파일 열기

            text = f.read()
            words = text.split()

            if t in words:
                # print(year, keyword)
                DF += 1

            f.close()  # 파일 닫기

    return log(N / DF)


"""
TF-IDF
"""
if __name__ == "__main__":
    years = list(range(2000, 2018))  # years range: 2000 ~ 2017
    keywords = ['불행', '행복']

    # total number of document in the corpus, N = |D|
    N = len(years) * len(keywords)

    # for idf
    if os.path.isfile('./tfidf/idf.json'):
        with open('./tfidf/idf.json', 'r') as rf:
            idf_dict = json.loads(rf.read())
    else:
        idf_dict = {}

    for year in years:
        for keyword in keywords:
            IN_PATH = './corpus' + '/' + str(year)
            OUT_PATH = './tfidf' + '/' + str(year)
            if not os.path.exists(OUT_PATH):
                os.makedirs(OUT_PATH)

            if os.path.isfile(OUT_PATH + '/' + keyword + '.txt'):
                continue

            dict = {}  # for tf

            # 파일 열기
            with open(IN_PATH + '/' + keyword + '.txt', 'r', encoding='UTF-8') as f:
                text = f.read()
                words = text.split()

            # f: 특정 문서 내 단어 출현 횟수
            for word in words:
                if word == keyword:
                    continue
                elif word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1

            # tf-idf
            cnt = 1
            for key in dict:
                if key not in idf_dict:
                    idf_dict[key] = idf(key, N)

                dict[key] *= idf_dict[key]
                # tf = log(1 + dict[key])  # log scale
                # dict[key] = tf * idf_dict[key]

                # log
                print('complete: tf-idf of ' + key + '(' + str(cnt) + '/' + str(len(dict)) + ')' +
                      ' in ' + IN_PATH + '/' + keyword +
                      ' is ' + str(dict[key]))

                cnt += 1

            tfidf = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples

            # idf.json 업데이트 per document
            with open('./tfidf/idf.json', 'w') as wf:
                wf.write(json.dumps(idf_dict))

            # 파일 쓰기
            with open(OUT_PATH + '/' + keyword + '.txt', 'w', encoding='UTF-8') as f:
                for elem in tfidf:
                    f.write(elem[0])
                    f.write('\n')
                    f.write(str(elem[1]))
                    f.write("\n\n")

            print("complete: SAVE " + OUT_PATH + '/' + keyword + '.txt')
