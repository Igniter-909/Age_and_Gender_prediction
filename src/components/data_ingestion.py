import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_extraction import Data_Extraction
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
from src.components.model_training import ModelTrainer
import numpy as np

@dataclass
class Data_Ingestion:
    def __init__(self,X,df):
        self.df = df
        self.X = X

    def data_ingestion(self):
        try:
            X = self.X/255.0
            y_gender = np.array(self.df["Gender"])
            y_age = np.array(self.df["Age"])

            logging.info("Converted all features into numpy arrays")
            return X, y_gender, y_age


        except Exception as e:
            logging.error("An error occurred during data ingestion", exc_info=True)
            raise CustomException(e,sys)



if __name__ == '__main__':
    zipfile_file_path = "data/archive (4).zip"
    extract_to_zipfile = "data/files"
    BASE_DIR = "data/files/UTKFace"
    extraction = Data_Extraction(zipfile_file_path=zipfile_file_path,extract_to_zipfile=extract_to_zipfile)
    extraction.data_extraction()
    obj1 = DataTransformation(BASE_DIR)
    X,df = obj1.data_transformer()
    obj = Data_Ingestion(X,df)
    X,y_gender,y_age = obj.data_ingestion()
    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(X,y_gender,y_age)

