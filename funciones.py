import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


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
