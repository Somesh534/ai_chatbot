import json
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open("intents.json", "r") as file:
    data = json.load(file)

# Prepare training data
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Custom tokenizer (no nltk)
def simple_tokenizer(text):
    # Lowercase, remove punctuation, split on spaces
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

# Vectorizer using custom tokenizer
vectorizer = CountVectorizer(tokenizer=simple_tokenizer)
X = vectorizer.fit_transform(sentences)

# Train model
clf = MultinomialNB()
clf.fit(X, labels)

# Chat loop
print("🤖 Chatbot is ready! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    X_test = vectorizer.transform([user_input])
    predicted_tag = clf.predict(X_test)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            print("Bot:", random.choice(intent["responses"]))
            break
