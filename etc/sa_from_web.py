import os
import csv
from time import sleep
from random import random
from selenium import webdriver


def save_words():
    years = list(range(2000, 2018))  # years range: 2000 ~ 2017
    keywords = ['불행', '행복']

    groups = []

    for year in years:
        for keyword in keywords:
            IN_PATH = './corpus' + '/' + str(year)

            # 파일 읽기
            with open(IN_PATH + '/' + keyword + '.txt', 'r', encoding='UTF-8') as f:
                text = f.read()
                words = text.split()
                # print("OPEN: " + IN_PATH + '/' + keyword + '.txt')

            groups += words

    groups = list(set(groups))
    groups.sort()
    # print(len(groups))

    """
    단어 저장
    """
    OUT_PATH = './sa'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    with open(OUT_PATH + '/words.txt', 'w', encoding='UTF-8') as f:
        f.write(("\n".join(groups)).strip())


def naver_news_cnt(words, attr, headless):
    # 브라우저 열기
    if headless:
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        driver = webdriver.Chrome(chrome_options=options)
    else:
        driver = webdriver.Chrome()

    """
    속성 조합
    """
    res = []

    for word in words:
        res_ = []  # 속성 cnt 수집

        for i, tar in enumerate(attr):
            temp = []
            for j in range(len(attr)):
                if j != i:
                    temp.append(attr[j])

            # https://search.naver.com/search.naver?sm=tab_hty.top&where=news&query="권력"+%2B가족+-정치+-경제
            url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=news&query="' +\
                word + '"+%2B' + tar + '+-' + '+-'.join(temp)
            # print(url)

            driver.get(url)  # 주소 접근

            # 검색결과가 없으면 loop continue
            rt = random() * 1 + 2
            sleep(rt)

            pre_elem = driver.find_element_by_id("content")
            try:
                elems = pre_elem.find_elements_by_id("notfound")
            except:
                elems = []

            if len(elems) != 0:
                res_.append(0)
                continue

            # total 4~9 sec random time 만큼 일시정지
            # bot 탐지를 피하고, 페이지 로딩 시간을 확보
            rt = random() * 4 + 2
            sleep(rt)

            # 기사 수 수집
            pre_elem = driver.find_element_by_id("content")
            elem = pre_elem.find_element_by_class_name("section_head")
            cnt_s = elem.find_element_by_tag_name("span")  # cnt 추출

            cnt_ = list(((cnt_s.text).split('/')[1]).strip()[:-1])
            while ',' in cnt_:
                cnt_.remove(',')

            cnt = int("".join(cnt_))
            res_.append(cnt)

        # print(res_)  # collection of int
        print(word + ":", res_)
        res.append(res_)

    # close
    driver.close()

    return res


if __name__ == "__main__":
    # file check
    IN_PATH = './sa/words.txt'
    if os.path.isfile(IN_PATH):
        pass
    else:
        save_words()
        print("complete: SAVE:", IN_PATH)

    # load
    words = []
    with open(IN_PATH, 'r', encoding='UTF-8') as f:
        while True:
            word_ = f.readline().strip()
            if word_ is not '':
                words.append(word_)
            else:
                break
        print("complete: LOAD:", IN_PATH)

    """
    Sentiment Analysis
    """
    OUT_PATH = './sa/dict.csv'
    LOG_PATH = './sa/log.txt'

    log = 0
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, 'w', encoding='UTF-8') as f:
            f.write(str(0))
    else:
        with open(LOG_PATH, 'r', encoding='UTF-8') as f:
            log = int(f.readline().strip())
            print("complete: LOAD: " + LOG_PATH + ' - ' + str(log))

    """
    hyperparameter
    """
    attr = ["정치", "경제", "가족", "문화"]  # attributes
    TH = 100  # threshold 단위로 save

    # do
    for num in range(log, len(words), TH):
        values = naver_news_cnt(words[num:num + TH], attr, False)  # headless?

        # csv 쓰기
        if os.path.isfile(OUT_PATH):
            fw = open(OUT_PATH, 'a', newline='')
            wt = csv.writer(fw)
        else:
            fw = open(OUT_PATH, 'w', newline='')
            wt = csv.writer(fw)
            wt.writerow([''] + attr)

        for i, value in enumerate(values):
            wt.writerow(words[num + i:num + i + 1] + value)

        fw.close()

        # update log
        limit = num + TH
        if limit > len(words):
            limit = len(words)

        with open(LOG_PATH, 'w', encoding='UTF-8') as f:
            f.write(str(limit))

        print("complete: WRITE (" + str(limit) + '/' + str(len(words)) + "):", OUT_PATH)
