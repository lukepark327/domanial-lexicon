import json
import operator
import os
import csv
from math import log
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def list_percentage(tar):
    tot = 0
    for item in tar:
        tot += item

    for i, item in enumerate(tar):
        tar[i] = item / tot

    return tar


def init_main(files):
    # read sa dict.
    sa = {}
    for file in files:
        with open('./sa/'+file+'.json', 'r') as rf:
            sa_ = json.loads(rf.read())
            sa = dict(sa, **sa_)

    # read attr. list
    with open('./sa/data/attr.txt', 'r') as rf:
        attr = rf.readlines()
        for i, val in enumerate(attr):
            if val[-1] == '\n':
                attr[i] = val[:-1]

    return sa, attr


def del_dependence_dict(pre, stops):
    post = {}

    for key in pre:
        tmp = []

        for i, _ in enumerate(pre[key]):
            if not i in stops:
                tmp.append(pre[key][i])

        post[key] = tmp

    return post


def del_dependence_list(pre, stops):
    post = []

    for i, _ in enumerate(pre):
        if not i in stops:
            post.append(pre[i])

    return post


def cal_weight(weight):
    tot = 0  # total count
    for i in weight:
        tot += i

    for i, _ in enumerate(weight):
        if weight[i] is not 0:
            weight[i] = log(tot / weight[i])
        else:
            weight[i] = 1

    return weight


def attr_sort(attr, tmp):
    # result
    ret = {}
    for i, _ in enumerate(tmp):
        ret[attr[i]] = tmp[i]

    return sorted(ret.items(), key=operator.itemgetter(1), reverse=True)


def csv_writer(OUT_PATH, limit=0):
    fw = open(OUT_PATH, 'w', newline='')
    wt = csv.writer(fw)

    if limit == 0:
        wt.writerow(['level', 'count'] + [num for num in range(1, 21)])
    else:
        wt.writerow(['level', 'count'] + [num for num in range(1, (limit+1))])

    for tar in sa_list:
        # 속성별로 tf-idf 순으로 정렬
        ret = sorted(words[attr.index(tar[0])].items(), key=operator.itemgetter(1), reverse=True)

        if limit == 0:
            wt.writerow([tar[0]] + [len(ret)] + [x[0] for x in ret])
            wt.writerow([tar[1]] + [''] + [x[1] for x in ret])
        else:
            wt.writerow([tar[0]] + [len(ret)] + [x[0] for x in ret][:limit])
            wt.writerow([tar[1]] + [''] + [x[1] for x in ret][:limit])

    fw.close()
    print("complete: WRITE -", OUT_PATH)


if __name__ == "__main__":
    """
    init

    sa: sa dict. - const.
    attr: list of attr. - const.
    scaler: norm. - const.
    stops: list of dependency attr. - const.
    limit: max len. of csv's rows
    """
    stops = [0, 1, 4, 8, 16, 23, 27, 33, 38, 43]  # 종속적인/값이없는 속성 제외
    limit = 20

    sa, attr = init_main(['close_0', 'open_0'])
    sa = del_dependence_dict(sa, stops)
    attr = del_dependence_list(attr, stops)
    num_ = len(attr)

    scaler = MinMaxScaler(feature_range=(0, 1))

    """
    LOOP
    """
    years = list(range(2000, 2018))  # years range: 2000 ~ 2017
    keywords = ['불행', '행복']

    for year in years:
        for keyword in keywords:

            """
            read tf-idf
            """
            tfidf = {}
            with open('./tfidf' + '/' + str(year) + '/' + keyword + '.txt', 'r', encoding='UTF-8') as rf:
                while True:
                    name = rf.readline()[:-1]
                    val = rf.readline()[:-1]
                    if name == '' and val == '':
                        break
    
                    if float(val) != 0.0:
                        tfidf[name] = float(val)  # except what tf-idf is 0.0
    
                    rf.readline()

            """
            most dominated attr.
            
            words: 속성마다 속하는 단어 list-tfidf 쌍을 저장
            """
            tmp = []
            for i in range(num_):
                tmp.append(0)

            weight = []
            for i in range(num_):
                weight.append(0)

            words = []
            for i in range(num_):
                words.append({})

            # cal
            for key in tfidf:
                if key in sa:
                    for i, _ in enumerate(sa[key]):
                        if sa[key][i] == 1:
                            tmp[i] += tfidf[key]
                            weight[i] += 1
                            (words[i])[key] = tfidf[key]

            weight = cal_weight(weight)
            for i, _ in enumerate(tmp):
                tmp[i] *= weight[i]  # apply weight

            """
            result
            
            sa_list: series(list) of most influential attr.
            """
            # tmp = list_percentage(tmp)  # percentage
            tmp = list((scaler.fit_transform(np.array(tmp).reshape(-1, 1)).reshape(1, -1))[0])

            sa_list = [x for x in attr_sort(attr, tmp)]
            # print(year, keyword, [x[0] for x in attr_sort(attr, tmp)])

            """
            words list per attr.
            """
            OUT_PATH = './sa/csv/' + str(year) + '_' + keyword + '.csv'
            if not os.path.exists('./sa/csv'):
                os.makedirs('./sa/csv')
            csv_writer(OUT_PATH, limit=0)

            OUT_PATH = './sa/csv_limit/' + str(year) + '_' + keyword + '.csv'
            if not os.path.exists('./sa/csv_limit'):
                os.makedirs('./sa/csv_limit')
            csv_writer(OUT_PATH, limit=limit)  # apply limit
