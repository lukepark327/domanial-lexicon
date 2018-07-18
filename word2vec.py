from gensim.models import word2vec
import os


years = list(range(2000, 2018))  # years range: 2000 ~ 2017
keywords = ['불행', '행복']

"""
word2vec
"""
for year in years:
    IN_PATH = './corpus' + '/' + str(year)
    OUT_PATH = './word2vec' + '/' + str(year)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # word2vec 모델 만들기
    # model 파일이 없다면
    if not os.path.isfile(OUT_PATH + '/' + 'word2vec.model'):
        data = word2vec.LineSentence(IN_PATH + '/' + 'morpheme.txt')

        model = word2vec.Word2Vec(data)  # default: size=100, window=5
        model.save(OUT_PATH + '/' + 'word2vec.model')
        print("complete: SAVE " + OUT_PATH + '/' + 'word2vec.model')

    # 모델 활용
    # 결과 파일이 없다면
    for keyword in keywords:
        if not os.path.isfile(OUT_PATH + '/' + keyword + '.txt'):

            # 모델 읽기
            model = word2vec.Word2Vec.load(OUT_PATH + '/' + 'word2vec.model')
            print("complete: READ " + OUT_PATH + '/' + 'word2vec.model')
            model.init_sims(replace=True)  # 메모리 최적화

            # 결과 저장
            f = open(OUT_PATH + '/' + keyword + '.txt', 'w', encoding='UTF-8')

            results = model.wv.most_similar(positive=[keyword], topn=20)
            for res in results:
                f.write(res[0])
                f.write("\n")
                f.write(str(res[1]))
                f.write("\n\n")

            f.close()
            print("complete: SAVE " + OUT_PATH + '/' + keyword + '.txt')
