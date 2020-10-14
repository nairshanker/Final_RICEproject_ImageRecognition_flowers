
import keras
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras import backend as K


class PredictFlower:

    def __init__(self):

        self.img_width = 180
        self.img_height = 180

        df_category = pd.read_csv("category.csv")
        df_category = df_category.sort_values(by=['Index'])
        self.category = list(df_category['Category'])
        self.num_category = len(self.category)
        K.clear_session()
        self.load_model()

        self.model_final._make_predict_function()

    def load_model(self):

        self.load("model_name.h5")

    

        return predictions
