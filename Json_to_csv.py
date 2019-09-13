# Создание единой базы новостей в файле формата csv из оригинальных json файлов

import pandas as pd
import json
import os.path

df = pd.DataFrame()

for i in range(1, 5):
    f = 1
    while True:
        if os.path.isfile('Original_data/' + str(i) + '/news_' + '0' * (7 - len(str(f))) + str(f) + '.json'):
            data = []
            for line in open('Original_data/' + str(i) + '/news_' + '0' * (7 - len(str(f))) + str(f) + '.json', 'r',
                             encoding='utf-8'):
                data.append(json.loads(line))
            df = df.append(pd.DataFrame(data))
        elif os.path.isfile('Original_data/' + str(i) + '/blogs_' + '0' * (7 - len(str(f))) + str(f) + '.json'):
            data = []
            for line in open('Original_data/' + str(i) + '/blogs_' + '0' * (7 - len(str(f))) + str(f) + '.json', 'r',
                             encoding='utf-8'):
                data.append(json.loads(line))
            df = df.append(pd.DataFrame(data))
        else:
            break
        f += 1

df.to_csv('news_data.csv')
