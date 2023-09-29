from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import recommend_books

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    genre = request.form['genre']
    description = request.form['description']
    recommendations = recommend_books.predict(genre, description)
    return render_template('recommendation.html',recomendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
