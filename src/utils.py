import pickle
from exception import CustomException
import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(filename :str, object):
    """
    Save an object to a file using pickle.

    Parameters:
    filename (str): The path and name of the file to save the object to.
    object (any): The object to be saved.

    Returns:
    None

    Raises:
    CustomException: If an error occurs while saving the object.
    """
    try:
        dirpath=os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(object, file)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)






def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param =params[list(models.keys())[i]]

            gs= GridSearchCV(model, param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred =model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]]=test_r2
        return report   
    except Exception as e:
        raise CustomException(e,sys)