import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the model
st.write("Loading the model...")
model = load_model('rb36tn52_sentiment_model.h5')
st.write("Model Loaded Successfully!")

# Tokenizer setup
tokenizer = Tokenizer(num_words=5000)

# Function to predict sentiment
def predict_sentiment(review, threshold=0.7):
    # Convert review to sequence
    seq = tokenizer.texts_to_sequences([review])
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
        st.write("Interpretation: Very strong confidence in the predicted sentiment.")
    elif confidence >= 0.7:
        st.write("Interpretation: Strong confidence in the predicted sentiment.")
    elif confidence >= 0.5:
        st.write("Interpretation: Moderate confidence in the predicted sentiment.")
    else:
        st.write("Interpretation: Low confidence in the predicted sentiment.")

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
        st.write("⚠️ Please enter a review to predict.")

