from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer




if __name__ == '__main__':
    obj = DataIngestion()
    path=obj.initiate_data_ingestion()
    print(path)

    obj2 =  ModelTrainer()
    model_path = obj2.initate_model_training(path[0])
    print(model_path)