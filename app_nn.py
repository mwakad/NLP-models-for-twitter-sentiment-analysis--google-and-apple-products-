# Saved streamlit_app_nn
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import re

# --- Preprocessing ---
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

def pos_lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return "a"
        elif tag.startswith("V"):
            return "v"
        elif tag.startswith("N"):
            return "n"
        elif tag.startswith("R"):
            return "r"
        else:
            return "n"
    return ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags])

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = pos_lemmatize_text(text)
    return text

# --- Load models ---
product_model = joblib.load("logistic_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
sentiment_model = load_model("neural_network.keras")

# --- Streamlit App ---
st.title("Tweet Classifier: Product & Sentiment using LR and NN")

tweet = st.text_area("Enter a tweet below:")

if st.button("Predict"):
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        # Predict product directly using pipeline
        product = product_model.predict([tweet])[0]

        # Preprocess for sentiment prediction
        cleaned = preprocess_text(tweet)
        vectorized = vectorizer.transform([cleaned]).toarray()

        # Predict sentiment
        sentiment_probs = sentiment_model.predict(vectorized)
        sentiment_label = np.argmax(sentiment_probs, axis=1)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map[sentiment_label]

        st.subheader("Predictions:")
        st.write(f"**Product**: {product}")
        st.write(f"**Sentiment**: {sentiment}")
