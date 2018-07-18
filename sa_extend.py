"""
- most_sim 으로 짜보기
"""


import os
import json
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import word2vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping


def init_main():
    def read_corpus(IN_PATH, OUT_PATH):
        text = ''

        years = list(range(2000, 2018))  # years range: 2000 ~ 2017
        for year in years:
            with open(IN_PATH + '/' + str(year) + '/morpheme.txt', 'r', encoding='UTF-8') as rf:
                text += rf.read()

        with open(OUT_PATH + '/corpus.txt', 'w', encoding='UTF-8') as wf:
            wf.write(text)

    """
    LOAD corpus.txt
    """
    if not os.path.isfile('./sa/data/corpus.txt'):
        read_corpus('./corpus', './sa/data')

    with open('./sa/data/corpus.txt', 'r', encoding='UTF-8') as rf:
        words = list(set(rf.read().split()))
    print("complete: LOAD corpus.txt")

    """
    LOAD word2vec.model
    """
    if not os.path.isfile('./sa/data/word2vec.model'):
        data = word2vec.LineSentence('./sa/data/corpus.txt')
        model = word2vec.Word2Vec(data)  # default: size=100, window=5
        model.save('./sa/data/word2vec.model')  # save word2vec.model

    model = word2vec.Word2Vec.load('./sa/data/word2vec.model')
    print("complete: LOAD word2vec.model")

    return words, model


def read_close(IN_PATH):
    import openpyxl

    wb = openpyxl.load_workbook(filename=IN_PATH, data_only=True)  # load
    ws = wb.get_sheet_by_name("분류확인")  # get sheet

    attr = []
    sa_close = {}

    for i, r in enumerate(ws.rows):
        if i == 0:
            for j in range(3, 47):
                attr.append(r[j].value)
        elif i == 1:
            continue
        else:
            word = r[1].value
            level = []
            for j in range(3, 47):
                if r[j].value == None:
                    level.append(0)
                else:
                    level.append(1)

            if not level.count(0) == len(attr):
                sa_close[word] = level  # 속성없는 단어가 아닐 경우에만

    wb.close()  # close

    return attr, sa_close


def del_dependence(pre, stops):
    post = {}

    for key in pre:
        tmp = []

        for i, _ in enumerate(pre[key]):
            if not i in stops:
                tmp.append(pre[key][i])

        post[key] = tmp

    return post


def re_dependence(pre, attr_num):
    post = {}

    for key in pre:
        tmp = []
        for i in range(attr_num):
            tmp.append(0)

        if pre[key][0] == 1:  tmp[0] = tmp[1] = tmp[2] = 1
        if pre[key][1] == 1:  tmp[0] = tmp[1] = tmp[3] = 1
        if pre[key][2] == 1:  tmp[0] = tmp[4] = tmp[5] = 1
        if pre[key][3] == 1:  tmp[0] = tmp[4] = tmp[6] = 1
        if pre[key][4] == 1:  tmp[0] = tmp[4] = tmp[7] = 1
        if pre[key][5] == 1:  tmp[8] = tmp[9] = 1
        if pre[key][6] == 1:  tmp[8] = tmp[10] = 1
        if pre[key][7] == 1:  tmp[8] = tmp[11] = 1
        if pre[key][8] == 1:  tmp[8] = tmp[12] = 1
        if pre[key][9] == 1:  tmp[8] = tmp[13] = 1
        if pre[key][10] == 1: tmp[8] = tmp[14] = 1
        if pre[key][11] == 1: tmp[15] = 1
        if pre[key][12] == 1: tmp[16] = tmp[17] = 1
        if pre[key][13] == 1: tmp[16] = tmp[18] = 1
        if pre[key][14] == 1: tmp[16] = tmp[19] = 1
        if pre[key][15] == 1: tmp[16] = tmp[20] = 1
        if pre[key][16] == 1: tmp[16] = tmp[21] = 1
        if pre[key][17] == 1: tmp[16] = tmp[22] = 1
        if pre[key][18] == 1: tmp[23] = tmp[24] = 1
        if pre[key][19] == 1: tmp[23] = tmp[25] = 1
        if pre[key][20] == 1: tmp[23] = tmp[26] = 1
        if pre[key][21] == 1: tmp[27] = tmp[28] = 1
        if pre[key][22] == 1: tmp[27] = tmp[29] = 1
        if pre[key][23] == 1: tmp[27] = tmp[30] = 1
        if pre[key][24] == 1: tmp[27] = tmp[31] = 1
        if pre[key][25] == 1: tmp[32] = 1
        if pre[key][26] == 1: tmp[33] = tmp[34] = 1
        if pre[key][27] == 1: tmp[33] = tmp[35] = 1
        if pre[key][28] == 1: tmp[33] = tmp[36] = 1
        if pre[key][29] == 1: tmp[33] = tmp[37] = 1
        if pre[key][30] == 1: tmp[38] = tmp[39] = 1
        if pre[key][31] == 1: tmp[38] = tmp[40] = 1
        if pre[key][32] == 1: tmp[41] = 1
        if pre[key][33] == 1: tmp[42] = 1

        post[key] = tmp

    return post


def cal_sa_raw(words, sa_, num_, model):
    sa_raw = {}
    inters = []  # words which in sa_ and words both
    outers = []  # others
    afters = []  # words which in inters and over (windows=5)

    for word in words:
        if word in sa_:
            inters.append(word)
        else:
            outers.append(word)

    # sa_raw
    for outer in outers:
        tmp = []
        for i in range(num_):
            tmp.append(0)

        for inter in inters:
            # window=5 이하 단어 무시 필요
            try:
                sim = model.similarity(outer, inter)
                for i in range(num_):
                    tmp[i] += sim * sa_[inter][i]

                afters.append(inter)

            except:
                pass

        if not tmp.count(0) == num_:
            sa_raw[outer] = tmp

    # mul weight
    weight = []
    for i in range(num_):
        weight.append(0)

    for after in afters:
        for i in range(num_):
            weight[i] += sa_[after][i]  # counting

    tot = 0  # total count
    for i in weight:
        tot += i

    for i, _ in enumerate(weight):
        if weight[i] is not 0:
            weight[i] = tot / weight[i]
        else:
            weight[i] = 1

    for key in sa_raw:
        for j, _ in enumerate(sa_raw[key]):
            sa_raw[key][j] *= weight[j]  # 가중치 적용

    return sa_raw


def cal_sa_test(words, sa_, num_, model):
    inters = {}  # words which in words and sa_ both
    for word in words:
        if word in sa_:
            inters[word] = sa_[word]

    sa_test = {}  # result
    afters = []

    for inter in inters:
        ctrs = inters.copy()
        del ctrs[inter]  # copy inters except a word 'inter'

        tmp = []
        for i in range(num_):
            tmp.append(0)

        for ctr in ctrs:
            # window=5 이하 단어 무시 필요
            try:
                sim = model.similarity(inter, ctr)
                for i in range(num_):
                    tmp[i] += sim * sa_[ctr][i]

                afters.append(ctr)

            except:
                pass

        if not tmp.count(0) == num_:
            sa_test[inter] = tmp

    # nul weight
    weight = []
    for i in range(num_):
        weight.append(0)

    for after in afters:
        for i in range(num_):
            weight[i] += sa_[after][i]

    tot = 0
    for i in weight:
        tot += i

    for i, _ in enumerate(weight):
        if weight[i] is not 0:
            weight[i] = tot / weight[i]
        else:
            weight[i] = 1

    for key in sa_test:
        for j, _ in enumerate(sa_test[key]):
            sa_test[key][j] *= weight[j]  # 가중치 적용

    return sa_test


def create_dataset(sa_test, sa_close, num_):
    X = np.array([]).reshape(0, num_)  # 학습 데이터 생성
    Y = np.array([]).reshape(0, num_)  # 정답 레이블 생성

    scaler = MinMaxScaler(feature_range=(-1, 1))  # 정규화

    for key in sa_test:
        tmp_test = []
        tmp_close = []

        for i in range(num_):
            tmp_test.append(sa_test[key][i])
            tmp_close.append(sa_close[key][i])

        X = np.vstack((X, scaler.fit_transform(np.array(tmp_test).reshape(-1, 1)).reshape(1, -1)))  # 정규화
        Y = np.vstack((Y, np.array(tmp_close)))

    return X, Y


def create_model(n_in, n_hiddens, n_out):
    p_keep = 0.5
    activation = 'relu'

    clf = Sequential()
    for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
        clf.add(Dense(
            n_hiddens[i],
            input_dim=input_dim,
            kernel_initializer='he_normal'))
        # clf.add(BatchNormalization())
        clf.add(Activation(activation))
        clf.add(Dropout(p_keep))

    clf.add(Dense(
        n_out,
        kernel_initializer='he_normal'))
    clf.add(Activation('sigmoid'))

    return clf


def training_and_testing(X, Y, clf, epochs, batch_size):
    def plot_hist(hist):
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()

    # split to train, testing and validation
    N_test = int(len(Y) * 0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=N_test)
    N_val = int(len(Y_train) * 0.1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=N_val)

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

    hist = clf.fit(X_train, Y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val),
                   callbacks=[early_stopping],
                   verbose=0)

    # draw plot
    # plot_hist(hist)

    scores = clf.evaluate(X_test, Y_test, verbose=0)
    return scores


def cal_sa_open(sa_raw, num_, clf):
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 정규화

    sa_open = {}

    for key in sa_raw:
        tmp_open = []
        for i in range(num_):
            tmp_open.append(sa_raw[key][i])

        P = scaler.fit_transform(np.array(tmp_open).reshape(-1, 1)).reshape(1, -1)  # 정규화
        preds_ = clf.predict(P)  # 예측
        preds_[preds_ >= 0.5] = 1
        preds_[preds_ < 0.5] = 0  # multi-label 분류

        # preds_ = (clf.predict(P) >= 0.5).astype(int)
        preds = preds_[0].tolist()

        if not preds.count(0) == num_:
            sa_open[key] = preds
            # print('(' + str(i) + '/' + str(len(sa_raw)) + ')', key, ':', preds)

    return sa_open


if __name__ == "__main__":
    """
    init
    
    words: list of all words in corpus - const.
    model: model of word2vec - const.
    stops: list of dependent attr.
    """
    words, model = init_main()
    stops = [0, 1, 4, 8, 16, 23, 27, 33, 38, 43]  # 종속적인/값이없는 속성 제외
    stop_num = len(stops)

    renew_th = 0  # dict. renew threshold
    loop_th = 1  # loop threshold
    idx = 0  # loop index

    sa_close = {}
    sa_open = {}
    attr = []
    attr_num = 0

    # LOOP
    while idx < loop_th:
        # print("LOOP", idx)

        """
        sa_close; pre-existing dict. - var.
        attr; attributes of dict. - const.
        attr_num: length of attr  - const.
        """
        if idx == 0:
            attr, sa_close = read_close('./sa/data/close.xlsx')  # load
            attr_num = len(attr)
        else:
            sa_close = dict(sa_close, **sa_open)  # expand

        with open('./sa/close_' + str(idx) + '.json', 'w') as wf:
            wf.write(json.dumps(sa_close))  # save json file
        print("complete: SAVE sa_close -", str(idx) + ", size:", len(sa_close))

        """
        sa_raw: new dict. But isn't refined yet - var.
              : already remove dependent attr.
        """
        sa_raw = cal_sa_raw(words, del_dependence(sa_close, stops), (attr_num - stop_num), model)
        print("complete: LOAD sa_raw   -", str(idx) + ", size:", len(sa_raw))

        # condition: END LOOP
        if not len(sa_raw) > renew_th:
            break

        """
        classification
        - training again per every loop
        - using earlystopping and dropout
        """
        # training and test set
        sa_test = cal_sa_test(words, del_dependence(sa_close, stops), (attr_num - stop_num), model)
        print("complete: LOAD sa_test  -", str(idx) + ", size:", len(sa_test))

        # create dataset X, Y
        X, Y = create_dataset(sa_test, del_dependence(sa_close, stops), (attr_num - stop_num))

        # create NN model
        clf = create_model(n_in=len(X[0]), n_hiddens=[200], n_out=len(Y[0]))
        # clf.summary()  # print info. of model
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # training and testing model
        scores = training_and_testing(X, Y, clf, epochs=1000, batch_size=200)
        print('complete: %s = %.2f%%' % (clf.metrics_names[1], scores[1] * 100))

        """
        sa_open: new dict. - var.
        """
        sa_open = re_dependence(cal_sa_open(sa_raw, (attr_num - stop_num), clf), attr_num)

        with open('./sa/open_' + str(idx) + '.json', 'w') as wf:
            wf.write(json.dumps(sa_open))
        print("complete: SAVE sa_open  -", str(idx) + ", size:", len(sa_open))

        """
        end of while loop
        """
        idx += 1  # increasing index
