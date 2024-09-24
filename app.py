import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.image("x_logo.png", width=100)
st.title('X Sentiment Analysis Prediction')

model = joblib.load('lstm.joblib')
le = joblib.load('label_encoder.joblib')
tokenizer = joblib.load('tokenizer.joblib')

input_text = st.text_area("Input text")

lemma = WordNetLemmatizer()
stopwords = stopwords.words("english")

def cleaning_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', str(text)) 
    text = re.sub(r'r\s+[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    words = word_tokenize(text)
    
    words = [lemma.lemmatize(word) for word in words]
    words = [word for word in words if word not in stopwords]
    words = [word for word in words if len(word) > 3]
    
    indices = np.unique(words, return_index=True)[1]
    cleaned_text = np.array(words)[np.sort(indices)].tolist()
    return cleaned_text

if st.button("Predict"):
    if input_text:
        x_test = cleaning_text(input_text)
        tokenizer.fit_on_texts(x_test)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=100)
        
        prediction = model.predict(x_test)
        
        if prediction.ndim > 1:
            prediction = prediction.argmax(axis = 1)
        
        prediction_label = le.inverse_transform(prediction)
        
        st.write(f"Sentiment Prediction: {prediction_label[0]}")
    else:
        st.write("Please enter the text!")
        
        
