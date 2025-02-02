import pickle 
import sys 
import os


from dataclasses import dataclass
import pandas as pd  
from utils import load_object
from logger import logging
from exception import CustomException


class PredictionPipeline:
    def __init__(self) :
        pass    
    
    def predict(self,feature):
        try:
            logging.info("Prediction pipeline initiated")
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocess_path = os.path.join('artifacts', 'proprocessor.pkl')
            model = load_object(model_path)
            logging.info("Model loaded successfully")
            preprocessor=load_object(preprocess_path)
            logging.info("Preprocessor loaded successfully")
            logging.info("Preprocessor data sacling started ")
            data_scaled=preprocessor.transform(feature)
            prediction = model.predict(data_scaled)
            logging.info("Prediction completed successfully")
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,gender:str,race_ethnicity: str,parental_level_of_education,lunch: str,
                 test_preparation_course: str,reading_score: int,writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict ={
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
            }
            
            return pd.DataFrame(custom_data_input_dict)  # Convert to DataFrame
            
        except Exception as e:
            raise CustomException(e,sys)





