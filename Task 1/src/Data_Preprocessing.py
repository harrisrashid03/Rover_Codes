import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_data(df):
    df['clean_plot'] = df['plot'].apply(clean_text)
    return df[['clean_plot', 'genre']]
