import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    data_path = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')

        try:
            """
                Here we are reading the data from the local files and saving them in the folders,
                        this is because it looks like pulling the data from the database 
            """

            books_df = pd.read_csv('data/preprocessed_file.csv')
            logging.info("Books Dataread Sucessfully")
            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)
            books_df.to_csv(self.ingestion_config.data_path,index=False,header=True)


            return str(self.ingestion_config.data_path),


        except Exception as e:
            logging.info('Error occured in Data ingestion')
            raise CustomException(sys,e)

