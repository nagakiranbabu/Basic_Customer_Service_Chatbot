import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import math
import tkinter
from tkinter import *

# Define the NaiveBayesClassifier class
class NaiveBayesClassifier:
    def __init__(self):
        self.classes_prob = {}
        self.word_cond_prob = {}
        self.class_count = {}

    def train(self, X, y):
        for i in range(len(y)):
            cls = tuple(y[i])
            if cls not in self.class_count:
                self.class_count[cls] = 0
                self.word_cond_prob[cls] = [0] * len(X[0])
            self.class_count[cls] += 1
            for j in range(len(X[0])):
                self.word_cond_prob[cls][j] += X[i][j]
        
        total_samples = len(y)
        for cls in self.class_count:
            self.classes_prob[cls] = self.class_count[cls] / total_samples
            self.word_cond_prob[cls] = [(self.word_cond_prob[cls][j] + 1) / (self.class_count[cls] + 2) for j in range(len(X[0]))]

    def predict(self, X):
        results = []
        for x in X:
            class_scores = {}
            for cls in self.classes_prob:
                log_prob = math.log(self.classes_prob[cls])
                for i in range(len(x)):
                    log_prob += x[i] * math.log(self.word_cond_prob[cls][i])
                class_scores[cls] = log_prob
            max_class = max(class_scores, key=class_scores.get)
            results.append(max_class)
        return results

# Load the model, words, and classes
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Define helper functions
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return bag

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict([p])[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand that."
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Define the GUI
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg:
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("ChatBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
