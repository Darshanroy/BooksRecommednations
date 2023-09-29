import logging

from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer
from src.logger import logging



if __name__ == '__main__':
    logging.info("Training Pipelin")
    obj = DataIngestion()
    path=obj.initiate_data_ingestion()
    print(path)

    obj2 =  ModelTrainer()
    model_path = obj2.initate_model_training(path)
    print(model_path)