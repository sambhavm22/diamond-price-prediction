import os, sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def prediction_data(self, features):
        try:
            preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
            model_file_path = os.path.join("artifacts", "model.pkl")
            logging.info("before loading")
            preprocessor = load_object(file_path=preprocessor_file_path)
            model = load_object(file_path=model_file_path)
            logging.info("after loading")

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred at Prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                carat:float, 
                cut:str,
                color:str,
                clarity:str,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float
                ):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z]
    }

            return pd.DataFrame(custom_data_input_dict)
        

        except Exception as e:
            logging.info("Error occurred in Prediction Pipeline")
            raise CustomException(e, sys)