import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

logging.info("All Required Lib for Data Ingestion Loaded succesfully.")

@dataclass
class DataIngestionConfig:
    data_path = os.path.join('artifacts','data.csv')
    logging.info("DataIngestionConfig Path created Succesfully.")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        logging.info("DataIngestion class Activated and Intiated the DataIngestionConfig Sucessfully.")

    def initiate_data_ingestion(self):
        logging.info('In DataIngestion class data ingestion Initiated Sucessfully. ')

        try:
            """
                Here we are reading the data from the local files and saving them in the folders,
                        this is because it looks like pulling the data from the database 
            """

            books_df = pd.read_csv('data/preprocessed_file.csv')

            logging.info("The DataSet from the database readed sucessfully.")

            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)
            books_df.to_csv(self.ingestion_config.data_path,index=False,header=True)

            logging.info(f"The read Data from Database is now saving to local folder path: {self.ingestion_config.data_path}")

            logging.info("DataIngestion class returns the Data of that we have saved in the local Computer.")
            return self.ingestion_config.data_path

        except Exception as e:
            logging.info('Error occured in Data ingestion')
            raise CustomException(sys,e)

