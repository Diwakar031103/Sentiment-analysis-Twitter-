#### streamlit

import streamlit as st
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the maximum sequence length
max_length = 100

# Function to predict sentiment
def predict_sentiment(text):
    sequences = tokenizer.text_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]
    return sentiment

# Set up Streamlit app
st.title("Sentiment Analysis App")
st.write("Enter your text below to analyze sentiment.")

# Text input from user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")


#### PyCharm  Terminal
* pip install streamlit
* pip install tensorflow
* streamlit run app.py



