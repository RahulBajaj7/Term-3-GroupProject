import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import requests
import os
import json

# GitHub URLs for model and tokenizer
model_url = "https://raw.githubusercontent.com/RahulBajaj7/Term-3-GroupProject/main/RNN/rb36tn52_sentiment_model.h5"
tokenizer_url = "https://raw.githubusercontent.com/RahulBajaj7/Term-3-GroupProject/main/RNN/rb36tn52_tokenizer.json"
model_path = "rb36tn52_sentiment_model.h5"
tokenizer_path = "rb36tn52_tokenizer.json"

# Download model if not present
def download_file(url, path):
    if not os.path.exists(path):
        st.write(f"📥 Downloading from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            st.write(f"✅ Downloaded {os.path.basename(path)} successfully.")
        else:
            st.error(f"❗ Failed to download {os.path.basename(path)}. Check the URL.")
            st.stop()

# Download model and tokenizer
download_file(model_url, model_path)
download_file(tokenizer_url, tokenizer_path)

# Load the model
st.write("🔎 Loading the model...")
try:
    model = load_model(model_path)
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❗ Error loading model: {e}")
    st.stop()

# Load tokenizer
st.write("🔎 Loading Tokenizer...")
try:
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = f.read()
        tokenizer = tokenizer_from_json(tokenizer_data)
    st.write("✅ Tokenizer Loaded Successfully!")
except Exception as e:
    st.error(f"❗ Error loading tokenizer: {e}")
    st.stop()

# Function to predict sentiment
def predict_sentiment(review, threshold=0.7):
    """
    Predicts sentiment using the RNN model.
    """
    # Convert review to sequence
    seq = tokenizer.texts_to_sequences([review])
    if not seq or len(seq[0]) == 0:
        st.error("❗ Unable to tokenize input. Please enter a valid review.")
        return
    padded_seq = pad_sequences(seq, maxlen=100)

    # Predict sentiment
    prediction = model.predict(padded_seq)[0][0]
    sentiment = 'Positive' if prediction >= threshold else 'Negative'
    confidence = prediction

    # Display result
    st.write(f"**Review:** {review}")
    st.write(f"**Predicted Sentiment:** {sentiment}")
    st.write(f"**Confidence Score:** {confidence:.4f}")

    if confidence >= 0.9:
        st.write("🔎 Interpretation: Very strong confidence in the predicted sentiment.")
    elif confidence >= 0.7:
        st.write("🔎 Interpretation: Strong confidence in the predicted sentiment.")
    elif confidence >= 0.5:
        st.write("🔎 Interpretation: Moderate confidence in the predicted sentiment.")
    else:
        st.write("🔎 Interpretation: Low confidence in the predicted sentiment.")

# Streamlit UI
st.title("📊 Sentiment Analysis Using RNN")
st.write("Enter a product review below to predict its sentiment.")

# Input box for user review
user_review = st.text_area("Enter Review:", "")

# Predict button
if st.button("Predict Sentiment"):
    if user_review.strip():
        predict_sentiment(user_review)
    else:
        st.warning("⚠️ Please enter a review to predict.")


