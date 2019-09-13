# Применение рекуррентной нейронной сети архитектуры LSTM

import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import keras

df = pd.read_csv('preprocessed.csv')

# Удаление символов и слов, наличие которых не улучшит качество модели
symbols_to_space = re.compile('[/(){}\[\]\|@,;]')
symbols_to_delete = re.compile('[^0-9a-z #+_]')
stopwords_list = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = symbols_to_space.sub(' ', text)
    text = symbols_to_delete.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in stopwords_list)
    return text

df['sentences'] = df['sentences'].apply(clean_text)
df['sentences'] = df['sentences'].str.replace('\d+', '')

# Подготовка данных: представление новостей в виде числовых векторов
tokenizer = keras.preprocessing.text.Tokenizer(num_words=100000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['sentences'])
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df['sentences'])
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=4000)
Y = pd.get_dummies(df['change'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

# Архитектура нейронной сети
model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 100, input_length=X.shape[1]))
model.add(keras.layers.SpatialDropout1D(0.2))
model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

# Обучение модели
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Тестирование модели
print('LSTM accuracy:', model.evaluate(X_test, Y_test))
