import logging

from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.utils import load_object

app = Flask(__name__)
df = pd.read_csv('src/pipelines/artifacts/data.csv')
# Function to recommend books based on genre and description
def recommend_books(genre, description, num_recommendations=5):
    tfidf_vectorizer= load_object('src/pipelines/artifacts/verctorizer.pkl')
    knn_model = load_object('src/pipelines/artifacts/knn_model.pkl')
    # Transform input description into TF-IDF vector
    input_vector = tfidf_vectorizer.transform([description])

    # Find the nearest neighbors based on cosine similarity
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=num_recommendations)

    # Get recommended book titles
    recommended_books = df.iloc[indices[0]]['Title'].tolist()

    return recommended_books

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    genre = request.form['genre']
    description = request.form['description']
    recommendations = recommend_books(genre, description)
    return render_template('recommendation.html',recomendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
