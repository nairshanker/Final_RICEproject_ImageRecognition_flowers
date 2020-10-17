import os
import keras
from keras.models import Model
from keras import applications, optimizers
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
# import tensorflow_hub as hub
from keras import backend as K
from keras.models import model_from_json

class PredictFlower:

    def __init__(self):

        self.img_width = 224
        self.img_height = 224

        df_category = pd.read_csv("category.csv")
        df_category = df_category.sort_values(by=['Index'])
        self.category = list(df_category['Category'])
        self.num_category = len(self.category)
        K.clear_session()
        self.loader()

        self.model3.make_predict_function()

    def loader(self):
        # pixels = 224
        # IMAGE_SIZE = (pixels, pixels)
        # do_fine_tuning = False

        # train_ds = tf.keras.preprocessing.image_dataset_from_directory('flower_photos',
        #             validation_split=0.2, subset="training", seed=123, image_size=(224, 224),
        #             batch_size=32)
        # class_names = train_ds.class_names
        # num_classes = len(class_names)

        # module_selection = ("mobilenet_v2_100_224", 224) 
        # handle_base, pixels = module_selection
        # MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
        # model3 = tf.keras.Sequential([
        # tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        # hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        # tf.keras.layers.Dropout(rate=0.2),
        # tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))])
        # model3.build((None,)+IMAGE_SIZE+(3,))
        # model3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
        #                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                metrics=['accuracy'])


        self.model3 = keras.models.load_model("mobilenet_model_trained_80_20")
        self.model3.load_weights("mobilenet_model_80_20.hdf5")
        
        

    def predictor(self, images, folder):

        predictions = []

        for image_name in images:
            image_path = folder + "/" + image_name
            print(f"imagepath: {image_path}")
            test_image = keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224), grayscale=False)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = preprocess_input(test_image)

            predict = self.model3.predict(test_image)

            zip_pred = zip(predict[0], self.category)

            match_found = False
            for pred_value, pred in zip_pred:
                if (pred_value > 0.8):
                    match_found = True
                    predictions.append((image_name, pred))

            if(not(match_found)):
                predictions.append((image_name, ""))

        return predictions
