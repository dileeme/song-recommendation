import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(csv_path):
  df = pd.read_csv(csv_path)
  return df

def train_model(df):
  vectorizer = TfidfVectorizer(stop_words='english')
  song_vectors = vectorizer.fit_transform(df['title'])
  return vectorizer, song_vectors


