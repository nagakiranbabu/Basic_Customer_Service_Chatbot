import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import math
import pickle

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize data structures
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0] * len(words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        if w in pattern_words:
            bag[words.index(w)] = 1
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and split data
random.shuffle(training)
train_x, train_y = zip(*training)

# Implement Naive Bayes Classifier
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

# Train and save the model
model = NaiveBayesClassifier()
model.train(train_x, train_y)

# Save model, words, and classes
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Model trained and saved.")
