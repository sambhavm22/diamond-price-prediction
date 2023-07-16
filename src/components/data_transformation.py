import os, sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  

    def get_data_transformed(self):
        
        try:

            logging.info("separating numerical and categorical features")
            numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_features = ['cut', 'color', 'clarity']
            
            logging.info("define the custom ranking for each ordinal variable")
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("creating pipeline")
            num_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy='median')),
                    ("scaler" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("ordinal_encoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("initiating data transformation")

        try:
            
            logging.info("read train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformed()

            target_column = 'price'
            drop_column = [target_column,'id']

            logging.info("separating data into dependent and independent variable")
            input_feature_train_df = train_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[target_column]
            print("________________------")
            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            print("________________")

            logging.info("concating input array and target")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            print("________________7877778867896789")
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            print(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            print("_____________________________________________")
            logging.info("saving preprocessor.pkl file")
            print(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            print("_____________________________________________")
            save_object(
                        file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )
            
            print(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            print("___________________________________")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
            
        except Exception as e:
            raise CustomException(e,sys)   

