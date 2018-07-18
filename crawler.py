import os
from datetime import datetime, timedelta
from time import sleep
from random import random
from selenium import webdriver


def query(keyword, startdate, enddate, page):
    """
    startdate, enddate: datetime
    page: start from 0 page
    """

    s1 = startdate.strftime('%Y.%m.%d')
    s2 = startdate.strftime('%Y%m%d')
    e1 = enddate.strftime('%Y.%m.%d')
    e2 = enddate.strftime('%Y%m%d')

    url = "https://search.naver.com/search.naver?ie=utf8&where=news&query=" \
          + str(keyword) + \
          "&sm=tab_pge&sort=2&photo=0&field=1&reporter_article=&pd=3&ds=" \
          + str(s1) + \
          "&de=" \
          + str(e1) + \
          "&docid=&nso=so:da,p:from" \
          + str(s2) + \
          "to" \
          + str(e2) + \
          ",a:t&mynews=0&start=" \
          + str(page) + "1" \
          "&refresh_start=0"

    return url


def naver_news_crawler(keyword, startdate, terminate, headless):
    # 브라우저 열기
    if headless:
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        driver = webdriver.Chrome(chrome_options=options)
    else:
        driver = webdriver.Chrome()

    # date
    date_flag = True

    while date_flag:
        enddate = startdate

        # 예외 처리
        if enddate > terminate:
            date_flag = False
            break

        if enddate > datetime.now():
            enddate = datetime.now()
            date_flag = False

        # 출력 경로 생성
        OUT_PATH = './result/' + keyword + '/' + str(startdate.strftime('%Y%m%d'))
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)

        # page
        page = 0

        while True:
            # get URL
            url = query(keyword, startdate, enddate, page)

            # 주소 접근
            driver.get(url)
            if not keyword + " : 네이버 뉴스검색" in driver.title:
                break # 예외처리

            # 검색결과가 없으면 loop break
            rt = random() * 1 + 2
            sleep(rt)

            pre_elem = driver.find_element_by_id("content")
            try:
                elems = pre_elem.find_elements_by_id("notfound")
            except:
                elems = []

            if len(elems) != 0:
                break

            # total 4~9 sec random time 만큼 일시정지
            # bot 탐지를 피하고, 페이지 로딩 시간을 확보
            rt = random() * 4 + 2
            sleep(rt)

            # 파일 쓰기; UTF-8
            FULL_PATH = OUT_PATH + '/' + str(page + 1) + '.txt'
            output = open(FULL_PATH, 'w', encoding='UTF-8')

            # 타이틀, 본문, 정보, 링크 수집
            pre_elem = driver.find_element_by_id("content")
            elem = pre_elem.find_element_by_class_name("type01")

            for i in range(1, 11):
                # 뉴스 하나 추출
                try:
                    elem_id = 'sp_nws' + str(page * 10 + i)
                    target = elem.find_element_by_id(elem_id)
                except:
                    break

                title = target.find_element_by_tag_name("dt")  # title 추출
                # link = title.find_element_by_css_selector('a').get_attribute('href')  # link 추출
                info_body = target.find_elements_by_tag_name("dd")
                # info = info_body[0]  # info 추출
                body = info_body[1]  # body 추출

                output.write(title.text)
                output.write('\n')
                output.write(body.text)
                # output.write('\n')
                # output.write(info.text)
                # output.write('\n')
                # output.write(link)
                output.write('\n\n')

            # 파일 닫기
            output.close()
            print("complete:", FULL_PATH)

            page += 1  # end of page loop

        startdate += timedelta(days=1)  # end of date loop

        # 예외 처리
        if startdate > datetime.now():
            startdate = datetime.now()
            date_flag = False

    # 브라우저 닫기
    driver.close()

    return None


if __name__ == "__main__":
    """
    argu: (keyword, start_date, terminate_date, headless?)
    """
    try:
        naver_news_crawler("행복", datetime(2014, 12, 5), datetime(2014, 12, 31), False)  # 19900101-20171231
        print("SUCCESS")
    except:
        print("FAIL")
