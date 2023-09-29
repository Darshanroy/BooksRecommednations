import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import sys
import os

logging.info("All Required Lib for Model Trianing Loaded succesfully.")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','knn_model.pkl')
    trained_vectorizer_model_file_path = os.path.join('artifacts', 'verctorizer.pkl')

    logging.info("The path for Model&vectorizer created sucesfully.")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("The paths of Model&vectorizer initiated (ModelTrainer class also intiated ")

    def initate_model_training(self, df_path):
        try:
            df = pd.read_csv(df_path)
            logging.info("Loading the Data from the local machine for model preprocessing and Training.")
            logging.info("model training process started")
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
            logging.info("The TFIDF-Vectorizer Preprocessor Created sucessfully")

            # Build the KNN model
            knn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
            knn_model.fit(tfidf_matrix)
            logging.info("knn Model Training completed sucessfully.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=knn_model
            )
            save_object(
                file_path=self.model_trainer_config.trained_vectorizer_model_file_path,
                obj=tfidf_vectorizer
            )
            logging.info("saving both the preprocessor(vectorizer)&model in pickel formate")

            logging.info("ModelTraing class returns the paths for model&vectorizer(preprocessor).")
            return (
                self.model_trainer_config.trained_model_file_path,
                self.model_trainer_config.trained_vectorizer_model_file_path
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
















