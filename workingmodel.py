import pandas as pd
import numpy as np
from indicnlp.tokenize import indic_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

df = pd.read_csv("tamil_movie_reviews_train.csv")

def enhanced_preprocessing(df):
    df = df.drop(columns=['ReviewId'], errors='ignore')
    
    def extract_movie_name(text):
        text = text.replace("<NEWLINE>", " ").replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        
        patterns = [
            r'[“"]([^"”]+)[”"]\s*படத்த(ின்|ை)',
            r'படம(்|்)\s*[“"]([^"”]+)[”"]',
            r'[“"]([^"”]+)[”"]\s*திரைப்படம்',
            r'(?:திரைப்பட|பட)மான\s*[“"]([^"”]+)[”"]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                movie_name = next((g for g in match.groups() if g), None)
                if movie_name and movie_name != "NEWLINE":
                    return movie_name
        
        words = indic_tokenize.trivial_tokenize(text, lang='ta')
        proper_nouns = [word for word in words if len(word) > 3 and word[0].isupper()]
        if proper_nouns:
            for word, _ in Counter(proper_nouns).most_common():
                if word != "NEWLINE":
                    return word
        return "Unknown"
    
    df['MovieName'] = df['ReviewInTamil'].apply(extract_movie_name)
    
    tamil_stopwords = set(["மற்றும்", "ஒரு", "என்று", "போன்ற", "இது", "அது"])
    
    def clean_text(text):
        text = re.sub(r'[^\w\s\u0B80-\u0BFF]', ' ', text)
        tokens = indic_tokenize.trivial_tokenize(text, lang='ta')
        tokens = [token for token in tokens if token not in tamil_stopwords and len(token) > 2]
        return ' '.join(tokens)
    
    df['processed_text'] = df['ReviewInTamil'].apply(clean_text)
    return df

df = enhanced_preprocessing(df)

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
similarity_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(query, top_n=5):
    if query in df['MovieName'].values:
        idx = df[df['MovieName'] == query].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
    else:
        query_processed = ' '.join(indic_tokenize.trivial_tokenize(
            re.sub(r'[^\w\s\u0B80-\u0BFF]', ' ', query), 
            lang='ta'))
        query_vec = vectorizer.transform([query_processed])
        sim_scores = list(enumerate(cosine_similarity(query_vec, tfidf_matrix)[0]))
    
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n+1]
    
    results = []
    for i, score in sorted_scores:
        if df.iloc[i]['MovieName'] != query:  # Don't recommend the same movie
            results.append(f"{df.iloc[i]['MovieName']} (score: {score:.2f})")
    
    return results[:top_n]

print("Tamil Movie Recommender")
print("----------------------")
while True:
    user_input = input("\nEnter a movie name or keywords (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break
    
    recommendations = get_recommendations(user_input)
    
    if not recommendations:
        print("No recommendations found. Try different keywords.")
    else:
        print("\nRecommended Movies:")
        for movie in recommendations:
            print(f"- {movie}")