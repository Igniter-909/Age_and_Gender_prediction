import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import numpy as np

class predictPipeline:
    def __init__(self):
        self.model = load_object(file_path="artifacts/model.pkl")
    def predict(self,images):
        gender_dict = {0: "Male", 1: "Female"}
        features = []

        for image in tqdm(images):
            img = load_img(image,color_mode="grayscale")
            img = img.resize((128,128),Image.LANCZOS)
            img = np.array(img)
            features.append(img)

        features = np.array(features)
        features = features.reshape(len(features),128,128,1)
        
        X = features/255.0

        pred = self.model.predict(np.expand_dims(X[0],axis=0))
        pred_gender = gender_dict[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])
        
        return pred_gender, pred_age
    

if __name__=="__main__":
    try:
        pipeline = predictPipeline()
        images = ["artifacts/one.jpg"]
        pred_gender,pred_age = pipeline.predict(images)
        print(f"Predicted gender: {pred_gender} and age is {pred_age}")
    except Exception as e:
        raise CustomException(e,sys)

