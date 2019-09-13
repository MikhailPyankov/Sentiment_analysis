# Представление текста в виде числового вектора методом bag-of-words, построение SVM модели

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('preprocessed.csv')

# Выборка разбивается на подвыборки для тренировки и тестирования модели
th = int(len(df) * 8 / 10 // 1)
training = df[:th]
testing = df[th:]

# Представление текста в виде числового вектора: на основании отдельных слов формируется словарь, после чего
# текст представляется как числовой вектор, длина которого равна размеру словаря, а каждый елемент которого
# показывает частоту использования в тексте соответствующего слова
# Однокоренные слова рассматриваются как одно слово
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([PorterStemmer().stem(w) for w in analyzer(doc)])


# Функция, трансформирующая массив текстов в матрицу (словарь создается на основании текстов
# или задается вручную)
# Слова, не несущие смысл вне контекста (stop words), а также те, что присутствуют в >50% или <1% выборки,
# игнорируются
def get_matrix(voc=None):
    if voc is None:
        return StemmedCountVectorizer(stop_words='english', analyzer='word', max_df=0.5, min_df=0.01)
    else:
        return StemmedCountVectorizer(vocabulary=voc)

# Формирование входной матрицы для тренировки модели и соответствующего словаря
vectorizer = get_matrix()
vec_train = vectorizer.fit_transform(training['sentences']).todense()
vocab = vectorizer.vocabulary_

# Формирование входной матрицы для тестирования модели на основании словаря тренировочной подвыборки
vectorizer = get_matrix(voc=vocab)
vec_test = vectorizer.fit_transform(testing['sentences']).todense()

# Значения гипер-параметров для поиска оптимального сочетания
tuned_params = [{'kernel': ['rbf'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
                {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]

# Оптимизация гипер-параметров
model = GridSearchCV(SVC(), tuned_params, cv=5)
model.fit(vec_train, training['change'])

# Оценка готовой модели
final_model = SVC(C=model.best_params_['C'], kernel=model.best_params_['kernel'],
                  gamma=model.best_params_['gamma'])
final_model.fit(vec_train, training['change'])
pred = final_model.predict(vec_test)
print('Accuracy: ', metrics.accuracy_score(testing['change'], pred))
