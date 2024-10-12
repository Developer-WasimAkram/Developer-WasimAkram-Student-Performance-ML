import os
import sys
import pickle
import dill
from exception import CustomException
from logger import logging
from data_transformation import DataTransformation
from dataclasses import dataclass
from utils import evaluate_models,save_object
#Modeling impport
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
@dataclass
class ModelTrainingConfig:
    model_object_path :str = os.path.join("artifacts",'model.pkl')
    #Provide path to save trained model pickle file 
    
class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info("model training initiated ")
        try:
            logging.info("Spliting training and test data for model training")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],       
                )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("Model parameters loaded")
            model_report :dict =evaluate_models(X_train, y_train, X_test, y_test,models,params)
             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            
            logging.info(f"Model saved successfully at {self.model_trainer_config}")
            predicted=best_model.predict(X_test)
            logging.info(f"Best Model prediction completed successfully and best model are {best_model}")
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model is {r2_square}")
            
            save_object(
                self.model_trainer_config.model_object_path,
                best_model
            )
            logging.info("Cycle completed")
            return r2_square
                       
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
    
