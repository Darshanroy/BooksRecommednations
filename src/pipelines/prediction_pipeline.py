import pandas as pd
from src.utils import load_object
from src.exception import CustomException
import sys
from src.logger import logging


class recommend_books:
    def __init__(self):
        pass
        logging.info("Predction Pipeline")

    def predict(genre, description, num_recommendations=5):

        try:
            df = pd.read_csv('src/pipelines/artifacts/data.csv')
            logging.info("Readed the data from local machine ")

            tfidf_vectorizer= load_object('src/pipelines/artifacts/verctorizer.pkl')
            knn_model = load_object('src/pipelines/artifacts/knn_model.pkl')
            logging.info("Loaded both the model from pickel file")

            # Transform input description into TF-IDF vector
            input_vector = tfidf_vectorizer.transform([description])
            # Find the nearest neighbors based on cosine similarity
            distances, indices = knn_model.kneighbors(input_vector, n_neighbors=num_recommendations)
            # Get recommended book titles
            recommended_books = df.iloc[indices[0]]['Title'].tolist()

            logging.info("Predction Completed")

            return recommended_books

        except Exception as e:
            raise CustomException(e,sys)

