from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
from logger import logging



if __name__ == '__main':
    # Initiate data ingestion
    obj_dataingestion=DataIngestion()
    train_path,test_path=obj_dataingestion.ingest_data()

    # initiate data transformation
    obj_datatransformation=DataTransformation()
    train_arr,test_arr,_=obj_datatransformation.initiate_data_transformation(train_path,test_path)

    # initiate model training
    obj_model=ModelTrainer()
    obj_model.initiate_model_trainer(train_arr,test_arr)
    logging.info("All components of the machine learning pipeline have been successfully executed.")