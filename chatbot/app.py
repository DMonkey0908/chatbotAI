import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from flask import Flask, render_template, request, jsonify

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load chatbot files
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.h5")

# Xử lý câu đầu vào
def clean_up_sentence(sentence):
    sentence = sentence.lower()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Giao diện chính của chatbot
@app.route('/')
def home():
    return render_template("template\\index.html")

# API để chatbot xử lý tin nhắn
@app.route('/get', methods=['POST'])
def chatbot_response():
    message = request.form['msg']  # Lấy tin nhắn từ người dùng
    ints = predict_class(message)  # Dự đoán intent
    res = get_response(ints, intents)  # Lấy phản hồi từ chatbot
    return jsonify({'response': res})  # Trả về phản hồi dưới dạng JSON

if __name__ == "__main__":
    app.run(debug=True)
