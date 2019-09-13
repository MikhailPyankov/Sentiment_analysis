# Классификация текстов по 5 шкалам согласно базе SenticNet, оценка XGBoost

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from senticnet.senticnet import SenticNet
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('preprocessed.csv')

lemmatizer = WordNetLemmatizer()    # преобразование слов в исходную форму
sn = SenticNet()   # база данных, содержащая классификацию слов и выражений по значению и настроению


# Функция, возвращающая оценки введенного текста по 5 шкалам базы SenticNet: polarity intensity,
# pleasantness, attention, sensitivity и aptitude. Оценка формируется как сумма оценок
# всех включенных в базу слов и выражений из текста и нормируется на кол-во слов в тексте
def SN(data):
    # Преобразование текста в вектор, формирование словаря слов и словосочетаний длинной до 3 слов включительно
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))
    vec = vectorizer.fit_transform([data]).todense()
    k = 0
    polarity_intense = sentics_pleasant = sentics_attention = sentics_sense = sentics_aptitude = 0
    for i in vectorizer.vocabulary_.keys():
        try:  # Попытка найти i-ое слово/выражение в базе
            num_repetitions = vec[0, vectorizer.vocabulary_[i]]
            polarity_intense += (float(sn.polarity_intense(i)) * num_repetitions)
            sentics_pleasant += (float(sn.sentics(i)['pleasantness']) * num_repetitions)
            sentics_attention += (float(sn.sentics(i)['attention']) * num_repetitions)
            sentics_sense += (float(sn.sentics(i)['sensitivity']) * num_repetitions)
            sentics_aptitude += (float(sn.sentics(i)['aptitude']) * num_repetitions)
            k += num_repetitions

        except:  # В случае неудачи каждое слово преобразуется в начальную форму и поиск повторяется
            ii = i.split(' ')
            w = 0
            while w < len(ii):
                ii[w] = lemmatizer.lemmatize(ii[w])
                w += 1
            ii = ' '.join(ii)
            try:
                num_repetitions = vec[0, vectorizer.vocabulary_[i]]
                polarity_intense += (float(sn.polarity_intense(ii)) * num_repetitions)
                sentics_pleasant += (float(sn.sentics(ii)['pleasantness']) * num_repetitions)
                sentics_attention += (float(sn.sentics(ii)['attention']) * num_repetitions)
                sentics_sense += (float(sn.sentics(ii)['sensitivity']) * num_repetitions)
                sentics_aptitude += (float(sn.sentics(ii)['aptitude']) * num_repetitions)
                k += num_repetitions

            except:
                continue

    # Нормирование на количество слов в тексте
    k = len(vectorizer.vocabulary_)
    polarity_intense = polarity_intense / k
    sentics_pleasant = sentics_pleasant / k
    sentics_attention = sentics_attention / k
    sentics_sense = sentics_sense / k
    sentics_aptitude = sentics_aptitude / k

    return polarity_intense, sentics_pleasant, sentics_attention, sentics_sense, sentics_aptitude


# Добавление оценок согласно базе SenticNet в основной датафрейм
indep = []
for col in ['polarity_intense', 'sentics_pleasant', 'sentics_attention', 'sentics_sense', 'sentics_aptitude']:
    df[col] = 0
    indep.append(col)
r = 0
while r < len(df):
    for cat in ['']:
        if df.loc[r, 'sentences'] != '[]':
            a,b,c,d,e = SN(df.loc[r, 'sentences'])
            df.loc[r, cat+'polarity_intense'] = a
            df.loc[r, cat+'sentics_pleasant'] = b
            df.loc[r, cat + 'sentics_attention'] = c
            df.loc[r, cat + 'sentics_sense'] = d
            df.loc[r, cat + 'sentics_aptitude'] = e
    r += 1

X_train, X_test, Y_train, Y_test = train_test_split(df[indep], df['change'], test_size=0.20, shuffle=False)

# Значения гипер-параметров для поиска оптимального сочетания
tuned_params = [{'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
                 'min_child_weight': [1, 3, 5, 7],
                 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                 'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}]

# Оптимизация гипер-параметров
model = GridSearchCV(xgboost.XGBClassifier(), tuned_params, cv=5)
model.fit(X_train, Y_train)

# Оценка готовой модели
final_model = xgboost.XGBClassifier(learning_rate=model.best_params_['learning_rate'],
                                    max_depth=model.best_params_['max_depth'],
                                    min_child_weight=model.best_params_['min_child_weight'],
                                    gamma=model.best_params_['gamma'],
                                    colsample_bytree=model.best_params_['colsample_bytree'])

final_model.fit(X_train, Y_train)
pred = final_model.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(Y_test, pred))
