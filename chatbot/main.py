import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Khởi tạo bộ lemmatizer
lemmatizer = WordNetLemmatizer()

# Đọc file intents.json với encoding UTF-8 để xử lý tiếng Việt
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize các patterns trong intents, đồng thời xử lý tiếng Việt
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  # Giữ nguyên tiếng Việt có dấu
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Chuẩn hóa danh sách các từ, không phân biệt chữ hoa/chữ thường
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Sắp xếp các intent
classes = sorted(set(classes))

# Lưu lại danh sách từ và lớp để sử dụng sau
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# Tạo túi từ (bag of words) để mô hình học
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Chuyển về chữ thường
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Tạo output cho mỗi câu
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle dữ liệu và chuyển sang numpy array
random.shuffle(training)
training = np.array(training)

# Tách ra trainX và trainY
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Xây dựng mô hình neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Sử dụng SGD optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train mô hình
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Lưu mô hình
model.save('chatbot_model.h5', hist)
print('Đã hoàn thành việc huấn luyện!')
