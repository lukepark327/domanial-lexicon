import os
from gensim.models import word2vec
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib import pyplot, font_manager, rc


"""
폰트 설정
"""
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 대처
font_location = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_location).get_name()
rc('font', family=font_name)

"""
PCA
"""
years = list(range(2000, 2018))  # years range: 2000 ~ 2017
keywords = ['불행', '행복']
n_keywords = len(keywords)

OUT_PATH = './pca'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

for year in years:
    PATH = './word2vec' + '/' + str(year)

    # 모델 읽기
    model = word2vec.Word2Vec.load(PATH + '/' + 'word2vec.model')
    print("complete: READ " + PATH + '/' + 'word2vec.model')
    model.init_sims(replace=True)  # 메모리 최적화

    # fit a 2D PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    # 데이터 분석
    each = []  # 키워드별 word2vec 상위 20개 항목
    for keyword in keywords:
        targets = [elem[0] for elem in model.wv.most_similar(positive=[keyword], topn=20)]
        each.append(targets)

    both = []  # 중복 항목
    both_ = {}
    for each_ in each:
        for elem in each_:
            if elem in both_:
                both_[elem] += 1
            else:
                both_[elem] = 1

    for key in both_:
        if both_[key] == n_keywords:
            both.append(key)

    """
    draw
    """
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        if word == keywords[0]:
            pyplot.scatter(result[i, 0], result[i, 1], color='red', s=7, label='불행')
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='red', size=11)  # '불행'

        elif word == keywords[1]:
            pyplot.scatter(result[i, 0], result[i, 1], color='blue', s=7, label='행복')
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='blue', size=11)  # '행복'

        elif word in both:
            pyplot.scatter(result[i, 0], result[i, 1], color='purple', s=3)
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='purple', size=7)  # both

        elif word in each[0]:
            pyplot.scatter(result[i, 0], result[i, 1], color='red', s=3)
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='red', size=7)  # 불행 top20

        elif word in each[1]:
            pyplot.scatter(result[i, 0], result[i, 1], color='blue', s=3)
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='blue', size=7)  # 행복 top20

        """
        else:
            pyplot.scatter(result[i, 0], result[i, 1], color='gray', s=1)
        """

    pyplot.title(str(year))
    # pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)

    # pyplot.show()
    pyplot.savefig(OUT_PATH + '/' + str(year) + '.png', additional_artists=[], bbox_inches="tight")  # save image
    print("complete: SAVE "+ OUT_PATH + '/' + str(year) + '.png')

    pyplot.gcf().clear()  # clear
