import os
import sys
from konlpy.tag import Komoran
komoran = Komoran()


def text_read(keyword, year):
    text = ''

    # 파일 열기
    IN_PATH = './result' + '/' + keyword + '/' + year
    dates = os.listdir(IN_PATH)

    for date in dates:
        SEMI_PATH = IN_PATH + '/' + date
        # print('open:', SEMI_PATH)

        pages = os.listdir(SEMI_PATH)
        for page in pages:
            FULL_PATH = SEMI_PATH + '/' + page

            with open(FULL_PATH, 'r', encoding='UTF-8') as f:
                text += f.read()

    return text


def morpheme(text, stopwords, targets=[]):
    res = []

    lines = text.replace('...', '\n').replace('.', '\n').split('\n')  # replace '...': SE and '.': SF to '\n'
    for line in lines:
        r = []

        # 형태소 분석
        words = komoran.pos(line)
        for word in words:

            # 일반명사/고유명사만 취함
            if word[1] in ['NNG', 'NNP']:
                word_ = word[0]

                # 모든 키워드에 대한 중복 제외
                for target in targets:
                    if word_[0:len(target)] == target:
                        word_ = target

                # 불용어일 경우 제외
                if word_ not in stopwords:
                    r.append(word_)

        # 단어 한 개 이하로 구성된 문장은 분석에서 제외
        if len(r) > 1:
            r_ = (" ".join(r)).strip()

            if r_ not in ['']:
                res.append(r_)
                # print(r_)

    res.append('')
    return res


if __name__ == "__main__":
    # PATH 설정
    IN_PATH = './result'
    if not os.path.exists(IN_PATH):
        print("There is no PATH: ./result")
        sys.exit()

    OUT_PATH = './corpus'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    STOPWORD = './stopword/stopwords.txt'
    # 불용어 리스트가 있다면
    if os.path.isfile(STOPWORD):
        with open(STOPWORD, 'r', encoding='UTF-8') as f:
            stopwords = f.read().split()
    else:
        stopwords = []

    keywords = os.listdir(IN_PATH)  # keywords
    years = list(range(2000, 2018))  # years range: 2000 ~ 2017

    for year in years:
        # 출력 경로가 없다면
        SEMI_PATH = OUT_PATH + '/' + str(year)
        if not os.path.exists(SEMI_PATH):
            os.makedirs(SEMI_PATH)

        for keyword in keywords:
            FULL_PATH = SEMI_PATH + '/' + keyword + '.txt'

            # keyword + '.txt' 파일이 없다면
            if not os.path.isfile(FULL_PATH):
                text = text_read(keyword, str(year))  # return text
                print("complete: return text " + FULL_PATH)

                corpus = morpheme(text, stopwords, targets=keywords)  # return res
                print("complete: return corpus " + FULL_PATH)

                # 형태소 분석 결과 저장
                with open(FULL_PATH, 'w', encoding='UTF-8') as f:
                    f.write("\n".join(corpus))
                    print("complete: SAVE " + FULL_PATH)

        # morpheme 파일이 없다면
        if not os.path.isfile(SEMI_PATH + '/' + 'morpheme.txt'):
            text_total = ''

            for keyword in keywords:
                FULL_PATH = SEMI_PATH + '/' + keyword + '.txt'

                with open(FULL_PATH, 'r', encoding='UTF-8') as f:
                    text_total += f.read()

            with open(SEMI_PATH + '/' + 'morpheme.txt', 'w', encoding='UTF-8') as f:
                f.write(text_total)
                print("complete: SAVE " + SEMI_PATH + '/' + 'morpheme.txt')
