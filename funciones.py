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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

## NLP
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

def get_common_words(series: pd.Series, top_n: int = 20) -> None:

    all_words = series.explode()

    # Obtenemos la frecuencia de cada palabra
    word_counts = (
        all_words
        .value_counts()
        .head(top_n)
        .rename_axis('word')
        .reset_index(name='count')
    )

    # Graficamos las palabras más comunes
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=word_counts)
    plt.title(f"Top {top_n} Palabras Más Comunes")
    plt.xlabel('Conteo')
    plt.ylabel('Palabra')
    plt.show()

def get_common_words_by_label(df, col: str, top_n: int = 20) -> None:
    # Filtramos los datos por etiqueta
    ham_words = df[df['label'] == 'ham'][col].explode()
    spam_words = df[df['label'] == 'spam'][col].explode()

    # Obtenemos la frecuencia de cada palabra
    ham_word_counts = (
        ham_words
        .value_counts()
        .head(top_n)
        .rename_axis('word')
        .reset_index(name='count')
    )
    spam_word_counts = (
        spam_words
        .value_counts()
        .head(top_n)
        .rename_axis('word')
        .reset_index(name='count')
    )

    # Graficamos las palabras más comunes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(x='count', y='word', data=ham_word_counts, ax=axes[0])
    axes[0].set_title(f"Top {top_n} Palabras Más Comunes en Ham")
    axes[0].set_xlabel('Conteo')
    axes[0].set_ylabel('Palabra')

    sns.barplot(x='count', y='word', data=spam_word_counts, ax=axes[1])
    axes[1].set_title(f"Top {top_n} Palabras Más Comunes en Spam")
    axes[1].set_xlabel('Conteo')
    axes[1].set_ylabel('Palabra')

    plt.tight_layout()
    plt.show()


##MODELOS

def confusion_matrix_plot(y_test, y_pred, name = "Modelo") -> None:
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        cbar=False,
        annot_kws={"size": 20}
    )

    # Añadir detalles
    plt.title(f'Matriz de Confusión {name}', fontsize=14)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Real', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()

def get_top_words(model, vectorizer, top_n: int=10):
    feature_names = vectorizer.get_feature_names_out()

    if isinstance(model, LogisticRegression):
        # Para Regresión Logística: usamos coef_
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({'word': feature_names, 'weight': coefficients})
        top_spam = coef_df.sort_values(by='weight', ascending=False).head(top_n)
        top_ham = coef_df.sort_values(by='weight').head(top_n)

    elif isinstance(model, MultinomialNB):
        # Para MultinomialNB: usamos feature_log_prob_
        log_probs = model.feature_log_prob_
        spam_log_prob = log_probs[1, :]
        ham_log_prob = log_probs[0, :]
        # La diferencia de la probabilidad de que una palaba sea spam o ham indicará a que se asocia más la palabra
        diff_log_prob = spam_log_prob - ham_log_prob
        
        coef_df = pd.DataFrame({
            'word': feature_names,
            'diff': diff_log_prob
        })
        top_spam = coef_df.sort_values(by='diff', ascending=False).head(top_n)
        top_ham = coef_df.sort_values(by='diff').head(top_n)
    
    else:
        raise ValueError("Modelo no soportado. Usa LogisticRegression o MultinomialNB")

    return top_spam, top_ham