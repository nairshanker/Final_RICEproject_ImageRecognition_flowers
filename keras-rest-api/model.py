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

        self.load("C:/ShankersDocs/EDUCATION/RICE_Bootcamp_DataAnalytics/FinalProject_Img_Recognition_Flowers/Final_RICEproject_ImageRecognition_flowers/mobilenet_model_trained_80_20")

    def call_predict(self, images, folder):

        predictions = []

        for image_name in images:
            image_path = folder + "/" + image_name
            print(f"imagepath: {image_path}")
            test_image = keras.preprocessing.image.load_img(
                image_path, target_size=(180, 180), grayscale=False)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = preprocess_input(test_image)

            predict = self.model_final.predict(test_image)

            zip_pred = zip(predict[0], self.category)

            match_found = False
            for pred_value, pred in zip_pred:
                if (pred_value > 0.8):
                    match_found = True
                    predictions.append((image_name, pred))

            if(not(match_found)):
                predictions.append((image_name, ""))

        return predictions