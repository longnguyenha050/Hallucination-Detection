import pandas as pd
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def preprocess_data(df):
    df['context'] = df['context'].apply(clean_text)
    df['prompt'] = df['prompt'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)
    return df

def load_data(path):
    df = pd.read_csv(path)
    df = preprocess_data(df)
    return df
