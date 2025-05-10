import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()


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

def lemmatize_tokens(tokens):
    return tokens
