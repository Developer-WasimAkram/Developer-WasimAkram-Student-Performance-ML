import os 
import sys 
from exception import CustomException
from logger import logging
from utils import save_object
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    logging.info("Initializing DataTransformationConfig for preprocessor path")
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        
    def get_transformation_object(self):
        
       try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info("Splitting numerical and categorical data in group for data transformation input")
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            logging.info("Creating numerical pipeline for data imputation  Simple imputer and standard scalaer")
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            logging.info("Creating catogorical pipeline for data imputation simpleImputer and one Hot Encoder and standard scalaer")

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )
            logging.info("Creating ColumnTransformer for data transformation")
            return preprocessor
       except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Initaing data transformation process")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(" Read train and test data ")
            target_column_name = "math_score"
            logging.info("Dropping target column from training and test data")
            X_train = train_df.drop(columns=[target_column_name],axis=1)    
            y_train = train_df[target_column_name]   
            
            
            X_test = test_df.drop(columns=[target_column_name],axis=1)
            y_test = test_df[target_column_name]
            logging.info(" creating data preprocessor object for  data transformation")
            preprocessing_obj= self.get_transformation_object()  
          
            X_train_array = preprocessing_obj.fit_transform(X_train)
            logging.info("Fitted and transformed X_training data ")
            X_test_array = preprocessing_obj.transform(X_test)
            logging.info("transformed X_test  data ")
            train_array=np.c_[X_train_array, np.array(y_train)]
            logging.info("After transformation X_train data and y_train data concatenated in numpy arrays")
            test_array = np.c_[X_test_array,np.array(y_test)]
            logging.info("After transformation X_test data and y_test data concatenated in numpy arrays")
            logging.info("data transformation completed successfully")
            save_object(self.config.preprocessor_obj_file_path,preprocessing_obj)
            logging.info("Saved preprocessor object for future use")
            return (train_array, test_array, self.config.preprocessor_obj_file_path,)           
            
        except Exception as e:
            raise CustomException(e,sys)
     