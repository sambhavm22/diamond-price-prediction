import os, sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw_data.csv")
    train_data_path:str = os.path.join("artifacts", "train_data.csv")
    test_data_path:str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:

            logging.info("reading the data as data frame")
            df = pd.read_csv("notebook/gemstone.csv")

            logging.info("saving the data as data frame")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("splitting data into train and test")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            logging.info("saving train data")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            logging.info("saving train data")
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("data ingestion completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise(e, sys)    
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()        

