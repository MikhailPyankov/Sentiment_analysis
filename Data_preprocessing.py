# Предварительная обработка данных и создание основного датафрейма (df) для последующего построения моделей

import pandas as pd
from Companies import get_forbes_comp, alternative_names
import gc
from nltk import sent_tokenize
import re
import numpy as np

comp = get_forbes_comp()
df = pd.DataFrame()
class_bounds = {}  # пограничные значения для последующей классификации наблюдений по группам

# 1) Создание датафрейма (df), содержащего изменения цен акций финансовых компаний из списка Форбс
# за январь-май 2018

# Источник - yahoo.finance
for c in comp:
    try:
        aux = pd.read_csv('Stock_data/' + str(c) + '.csv', usecols=[0, 5],
                          names=['date', 'adj_close'], skiprows=[0])
    except:
        continue

    # Наблюдения за пределами нужного временного интервала выбрасываются, сохраняюся только даты для дальнейшего
    # соединения с датафреймом новостей
    aux['date'] = pd.to_datetime(aux['date'], yearfirst=True, errors='coerce')
    aux['date'] = aux['date'].dt.date

    # Подсчет изменений цен акций
    aux['adj_close'] = aux['adj_close'].astype(float)
    aux['return'] = aux['adj_close'].pct_change()

    # Сохранение пограничных значения для классов для каждой компании
    bound = aux[(aux['date'] >= pd.to_datetime('2014-01-01').date()) &
              (aux['date'] <= pd.to_datetime('2017-12-31').date())]
    lower = np.quantile(bound['return'].values, 0.33)
    upper = np.quantile(bound['return'].values, 0.67)
    if lower == lower:
        class_bounds[c] = [lower, upper]

    aux = aux[((aux['date'] >= pd.to_datetime('2017-12-29').date()) &
               (aux['date'] <= pd.to_datetime('2018-05-31').date()))]

    aux['company'] = c
    df = df.append(aux)
    del aux, bound
    gc.collect()

# Сортировка наблюдений, удаление лишней колонки и неполных наблюдений, добавление колонки для новостей
# Сортировка по названиям компаний оптимизирует работу алгоритма поиска соответствующих новостей
df = df.sort_values(by=['company', 'date'])
df = df.drop(columns=['adj_close'])
df = df.dropna().reset_index(drop=True)
df['sentences'] = '/'

# 2) Добавление новостей, соответствующих каждой строке основного датафрейма df
# (уникальной комбинации компании и даты) из базы новостей

# Процесс происходит в 2 этапа:
# а) Новости, относящиеся к конкретной компании, сохраняются во вспомогательный датафрейм (bux)
# б) Для каждого наблюдения, относящегося к этой компании, новости, соответствующие дате, выделяются из bux
# и добавляются в основной датафрейм (df)

pr_company = ''
aux = pd.read_csv('news_data.csv', usecols=['published', 'title', 'text'])  # датафрейм новостей
aux['published'] = pd.to_datetime(aux['published'], utc=True)

# Пробелы необходимы, чтобы алгоритм поиска названий компаний в тексте мог распознать название
# в первом или последнем слове статьи
aux['title'] = ' ' + aux['title'] + ' '
aux['text'] = ' ' + aux['text'] + ' '

r = 0
bux = pd.DataFrame()  # вспомогательный датафрейм

# Алгоритм обрабатывает каждую строчку основного датафрейма (df), но обновляет вспомогательный датафрейм (bux)
# только когда переходит к новой компании, для чего наблюдения были отсортированы по компаниям
while r < len(df):
    date = df.loc[r, 'date']
    company = df.loc[r, 'company']

    # обновление вспомогательного датафрейма при переходе к новой компании
    if company != pr_company:
        del bux
        gc.collect()

        # Статья считается относящейся к компании, если названии компании содержится в заголовке
        def get_condition(aux, ccc):
            return aux['title'].str.contains('\W' + ccc + '\W', na=False)

        bux = aux[get_condition(aux, company)][:]

        # Добавление новостей с учетом альтернативных названий некоторых компаний
        alt_names = alternative_names()
        if company in alt_names.keys():
            for x in alt_names[company]:
                bux = bux.append(aux[get_condition(aux, x)][:])
            all_names = [company] + alt_names[company]
        else:
            all_names = [company]

        bux['company'] = company
        bux = bux.drop_duplicates().reset_index(drop=True)

        # Новости, опубликованные после закрытия биржи, относятся к следующему торговому дню
        bux['date'] = ''
        rr = 0
        while rr < len(bux):
            dt = bux.loc[rr, 'published']
            if dt.month in [3, 4, 5]:
                if dt.time() >= pd.Timestamp('20:00').time():
                    bux.loc[rr, 'date'] = (dt + pd.DateOffset(1)).date()
                else:
                    bux.loc[rr, 'date'] = dt.date()
            else:
                if dt.time() >= pd.Timestamp('21:00').time():
                    bux.loc[rr, 'date'] = (dt + pd.DateOffset(1)).date()
                else:
                    bux.loc[rr, 'date'] = dt.date()
            rr += 1

    # Выделение релевантных новостей и удаление дубликатов
    relevant = bux[bux['date'] == date][:]
    titles = list(relevant['title'].values)
    texts = list(relevant['text'].values)

    del relevant
    gc.collect()

    # Релевантные новости записываются в единых массив: [заголовок, тело, ...]
    combined = []
    num = 0
    while num < len(titles):
        combined.append(titles[num])
        combined.append(texts[num])
        num += 1

    # Каждый элемент массива новостей разбивается на отдельные предложения
    sentences = []
    for item in combined:
        sentences.append(sent_tokenize(item))
    sentences = [item for sublist in sentences for item in sublist]

    # Предложения, не содержащие названия релевантной компании, удаляются
    s = 0
    while s < len(sentences):
        ind = 0
        for cny in all_names:
            if bool(re.search('\W' + cny + '\W', ' ' + sentences[s] + ' ')):
                ind = 1
        if ind == 0:
            del sentences[s]
        else:
            s += 1

    # Полученный массив, состоящий из релевантных для компании частей новостей, добавляется в основной датафрейм
    df.at[r, 'sentences'] = sentences

    pr_company = company
    r += 1

# Наблюдения без релевантных новостей удаляются
df = df[(df['sentences'].astype(bool)) & (df['company'].isin(class_bounds.keys()))].reset_index(drop=True)

# 3) Классификация наблюдений в 3 группы согласно изменениям цен акций:
# цена значительно выросла, значительно упала, изменение незначительно

# Классификация производится на основании значений 33-го и 67-го перцентилей для цен акции компании за 2014 - 2017
# Границы классов различны для разных компаний

df['change'] = 0
for c in df['company'].unique():
    df.loc[(df['company'] == c) & (df['return'] >= class_bounds[c][1]), 'change'] = 1
    df.loc[(df['company'] == c) & (df['return'] <= class_bounds[c][0]), 'change'] = -1
    print(class_bounds[c][0])

df = df.sort_values('date').reset_index(drop=True)
df.to_csv('preprocessed.csv')
