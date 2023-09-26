import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','knn_model.pkl')
    trained_vectorizer_model_file_path = os.path.join('artifacts', 'verctorizer.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, df_path):
        try:

            df = pd.read_csv(df_path)
            logging.info("model training process started")
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

            # Build the KNN model
            knn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
            knn_model.fit(tfidf_matrix)
            logging.info("Model Training completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=knn_model
            )
            save_object(
                file_path=self.model_trainer_config.trained_vectorizer_model_file_path,
                obj=tfidf_vectorizer
            )


            return (
                self.model_trainer_config.trained_model_file_path,
                self.model_trainer_config.trained_vectorizer_model_file_path
            )

            logging.info("Model Training Complted")

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
















