import os
import sys
from src.exception import CustomException
from src.logger import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

@dataclass
class DataTransformation:
    def __init__(self,BASE_DIR):
        self.BASE_DIR = BASE_DIR
        self.image_paths = []
        self.age_labels = []
        self.gender_labels = []


    def extract_features(self,images):  
        features = []

        for image in tqdm(images):
            img = load_img(image,color_mode="grayscale")
            img = img.resize((128,128),Image.Resampling.LANCZOS)
            img = np.array(img)
            features.append(img)

        features = np.array(features)
        features = features.reshape(len(features),128,128,1)
        return features

    def data_transformer(self):

        try:
            for filename in tqdm(os.listdir(self.BASE_DIR)):
                image_path = os.path.join(self.BASE_DIR, filename)
                temp = filename.split('_')
                age_class = temp[0]
                gender_class = temp[1]
                self.image_paths.append(image_path)
                self.age_labels.append(int(age_class))
                self.gender_labels.append(int(gender_class))

            logging.info("Created image_paths, age_labels and gender_labels list from given dataset")

        except Exception as e:
            logging.error("Error occurred while getting data transformation object", exc_info=True)
            raise CustomException(e, sys)
        
        logging.info(f"Length of the lists is :{len(self.image_paths)}")

        df = pd.DataFrame()
        df['Image'],df["Age"],df["Gender"] = self.image_paths,self.age_labels,self.gender_labels

        logging.info("Data added as a dataframe")

        X = self.extract_features(df['Image'])
        logging.info("Extracted features from images")

        return X,df

