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

def recommend_songs(input_song, df, vectorizer, song_vectors, top_n=5):
    input_vector = vectorizer.transform([input_song])
    similarity_scores = cosine_similarity(input_vector, song_vectors).flatten()
    top_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    return df.iloc[top_indices]['title'].tolist()

if __name__ == "__main__":
    csv_path = "songs.csv"  
    df = load_dataset(csv_path)
    vectorizer, song_vectors = train_model(df)
    
    input_song = # need to update song name here each time
    recommendations = recommend_songs(input_song, df, vectorizer, song_vectors)
    print("Recommended Songs:", recommendations)


