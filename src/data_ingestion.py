import os 
import sys  
from logger import logging
from exception import CustomException
import pandas as pd  
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass
from data_transformation import DataTransformation
from model_trainer import ModelTrainer




@dataclass
class DataIngestionConfig:
    train_data_path :str= os.path.join('artifacts', 'train_data.csv')
    test_data_path :str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path :str = os.path.join('artifacts', 'raw_data.csv')
    


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
     
    def ingest_data(self):
        logging.info("Cycle started")
        logging.info("Starting  the  data ingestion process") 
        try:
            # Read raw data  
            df=pd.read_csv("/Users/wasima/Desktop/Machine_Learning_Projects/Machine_Learning/notebook/data/data.csv")
            logging.info("Reading data from datasets ") 
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            logging.info("Created directory for saving raw data in artifacts ") 
            df.to_csv(self.config.raw_data_path, index=False,header=True)
            logging.info("saved raw data to csv in artifacts")
            # Split data into training and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Spliting  data into training and test sets")
            train_df.to_csv(self.config.train_data_path, index=False, header=True)  
            logging.info("saved training data to csv in artifacts")        
            test_df.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("saved test data to csv in artifacts")
        
            logging.info("Data ingestion completed successfully and return input train and test data parh for data transformation ")
    # Return paths to training and test data
            return (self.config.train_data_path, self.config.test_data_path)
           
        except Exception as e:
            raise CustomException(e,sys)


# Initiate data ingestion
obj_dataingestion=DataIngestion()
train_path,test_path=obj_dataingestion.ingest_data()

# initiate data transformation
obj_datatransformation=DataTransformation()
train_arr,test_arr,_=obj_datatransformation.initiate_data_transformation(train_path,test_path)

# initiate model training
obj_model=ModelTrainer()
obj_model.initiate_model_trainer(train_arr,test_arr)

   
    
