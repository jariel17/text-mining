import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def translate_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    
    stop_words = set(stopwords.words('english'))
    borra_caracteres = str.maketrans('', '', string.punctuation)

    cleaned_text = (
        text
        .str.lower()
        .apply(lambda txt: txt.translate(borra_caracteres))
        .apply(lambda txt: ' '.join(
             w for w in txt.split() if w not in stop_words
        ))
    )
    tokens = cleaned_text.apply(word_tokenize)
    return tokens

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = tokens.apply(pos_tag)

    lemmas = tagged_tokens.apply(
        lambda tagged_list: [
            lemmatizer.lemmatize(word, translate_pos(pos))
            for word, pos in tagged_list
        ]
    )
    return lemmas

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} - Evaluation Metrics\n" + "-"*40)
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()