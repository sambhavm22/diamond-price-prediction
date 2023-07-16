from dataclasses import dataclass

import numpy as np
import pandas as pd
import os, sys

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        logging.info("initiating the model trainer")
        try:
            logging.info("splitting the data into train and test")
            X_train, y_train, X_test, y_test = (
                                                train_array[:, : -1], 
                                                train_array[:,-1], 
                                                test_array[:,:-1], 
                                                test_array[:,-1]
                                                )
            logging.info("training the data with different models")
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "DecisionTree" : DecisionTreeRegressor(),
                "RandomForest" : RandomForestRegressor(),
                "KNeighbors" : KNeighborsRegressor()
            }

            logging.info("getting the best model and best model score out")
            model_report:dict = evaluate_model(
                                                X_train = X_train, 
                                                y_train = y_train, 
                                                X_test = X_test, 
                                                y_test = y_test, 
                                                models = models
                                                )
            
            logging.info("getting best model score from dict")
            best_model_score = max(sorted(model_report.values()))

            logging.info("getting best model name from dict")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info("setting a threshold for best model")
            if best_model_score < 0.6:
                raise CustomException("No best model is found", sys)
            logging.info(f"best model found on dataset")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            logging.info("saving the best model")
            save_object(
                        file_path=self.model_trainer_config.model_trainer_file_path,
                        obj=best_model
                        )
            
            logging.info("predicting the data using best_model")
            y_pred = best_model.predict(X_test)

            logging.info("finding the r2 score with best model")
            r2_sqaure = r2_score(y_test, y_pred)
            logging.info("model training completed")
            return r2_sqaure
            
        except Exception as e:
            raise CustomException(e, sys) 
