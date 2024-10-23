import os
import sys

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Input,Dense,Dropout
from tensorflow.keras.models import Model,Sequential
from src.utils import save_object



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerConfig()

    def initiate_model_trainer(self,X,y_gender,y_age):
        try:
            logging.info("Received data in model trainer")
            
            input_shape = (128,128,1)
            inputs = Input((input_shape))
            conv1 = Conv2D(32,kernel_size=(3,3),activation='relu')(inputs)
            maxp_1 = MaxPooling2D(pool_size=(2,2))(conv1)
            conv2 = Conv2D(64,kernel_size=(3,3),activation='relu')(maxp_1)
            maxp_2 = MaxPooling2D(pool_size=(2,2))(conv2)
            conv3 = Conv2D(128,kernel_size=(3,3),activation='relu')(maxp_2)
            maxp_3 = MaxPooling2D(pool_size=(2,2))(conv3)
            conv4 = Conv2D(256,kernel_size=(3,3),activation='relu')(maxp_3)
            maxp_4 = MaxPooling2D(pool_size=(2,2))(conv4)

            flatten = Flatten()(maxp_4)

            #fully connected layers
            dense_1 = Dense(256,activation='relu')(flatten)
            dense_2 = Dense(256,activation='relu')(flatten)

            dropout_1 = Dropout(0.4)(dense_1)
            dropout_2 = Dropout(0.4)(dense_2)

            output_1 = Dense(1,activation="sigmoid",name="gender_out")(dropout_1) 
            output_2 = Dense(1,activation="linear",name="age_out")(dropout_2)

            model = Model(inputs=[inputs],outputs=[output_1,output_2])

            model.compile(loss=["binary_crossentropy","mse"],optimizer="adam",metrics=["accuracy","mse"])

            logging.info("Model architecture created successfully")

            history = model.fit(x=X,y=[y_gender,y_age],epochs=30,batch_size=32,validation_split=0.2)

            logging.info("Successfully run the model with 30 epochs of training")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Successfully saved the model")

            acc = history.history['gender_out_accuracy']
            val_acc = history.history['val_gender_out_accuracy']
            epochs = range(len(acc))

            plt.plot(epochs,acc,'b',label="Training Accuracy")
            plt.plot(epochs,val_acc,'r',label="Validation Accuracy")
            plt.title("Accuracy Graph for Gender")
            plt.legend(loc="upper left")
            plt.savefig("artifacts/results/gender_out_accuracy.png")

            loss = history.history['gender_out_loss']
            val_loss = history.history['val_gender_out_loss']

            plt.plot(epochs,loss,'b',label="Training Loss")
            plt.plot(epochs,val_loss,'r',label="Validation Loss")
            plt.title("Loss Graph for Gender")
            plt.legend(loc="upper left")
            plt.savefig("artifacts/results/val_gender_out_loss.png")

            logging.info("Saved the results of gender")

            loss = history.history['age_out_mse']
            val_loss = history.history['val_age_out_mse']
            epochs = range(len(loss))

            plt.plot(epochs,loss,'b',label="Training Loss")
            plt.plot(epochs,val_loss,'r',label="Validation Loss")
            plt.title("Loss Graph for Age")
            plt.legend(loc="upper left")
            plt.savefig("artifacts/results/loss_age_out.png")
            logging.info("Saved the results of age")
        

        except Exception as e:
            raise CustomException(e,sys)
            