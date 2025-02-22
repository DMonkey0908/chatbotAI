import random
import json
import pickle
import numpy as np
import nltk
import tkinter as tk
from tkinter import Scrollbar, Text

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Đọc file intents với encoding UTF-8
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_model.h5")

# Hàm để xử lý câu input
def clean_up_sentence(sentence):
    sentence = sentence.lower()  # Chuyển câu thành chữ thường
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Tạo túi từ (bag of words)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Hàm dự đoán intent
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

# Hàm lấy phản hồi từ intent
def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Hàm xử lý khi người dùng gửi tin nhắn
def send_message():
    message = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", tk.END)

    if message != '':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + message + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        ints = predict_class(message)
        res = get_response(ints, intents)

        chat_log.insert(tk.END, "Bot: " + res + '\n\n')
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

# Tạo giao diện bằng Tkinter
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Tạo khung chat
chat_log = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=tk.DISABLED)

# Thanh cuộn
scrollbar = Scrollbar(root, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Nơi người dùng nhập tin nhắn
entry_box = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
entry_box.bind("<Return>", lambda event: send_message())

# Nút gửi tin nhắn
send_button = tk.Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send_message)

# Vị trí các widget
scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=6, y=401, height=90, width=265)
send_button.place(x=275, y=401, height=90)

# Chạy giao diện
root.mainloop()
