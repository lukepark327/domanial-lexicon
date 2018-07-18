import csv
import os


targets = ['word2vec', 'tfidf']
limit = 20

years = list(range(2000, 2018))  # years range: 2000 ~ 2017
keywords = ['불행', '행복']

for target in targets:
    for keyword in keywords:
        # csv create
        OUT_PATH = './csv/' + target + '_' + keyword + '.csv'
        if not os.path.exists('./csv'):
            os.makedirs('./csv')
        fw = open(OUT_PATH, 'w', newline='')
        wt = csv.writer(fw)

        for year in years:
            IN_PATH = './' + target + '/' + str(year) + '/' + keyword + '.txt'

            words = []
            values = []

            # txt read
            with open(IN_PATH, 'r', encoding='UTF-8') as fr:
                while True:
                    word = fr.readline()
                    if word is '':
                        break
                    elif word is '\n':
                        continue
                    value = fr.readline()

                    words.append(word.strip())
                    values.append(value.strip())

            wt.writerow([str(year)] + words[:limit])
            wt.writerow([''] + values[:limit])

        fw.close()
        print("complete: WRITE:", OUT_PATH)
