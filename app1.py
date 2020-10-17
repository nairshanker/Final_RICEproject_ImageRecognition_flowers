from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras import backend
from keras.models import load_model
from tensorflow.python.framework import ops

import os

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()


def read_image(filename):
    # Load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 28, 28, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('./template/index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                img = read_image(file_path)


                with graph.as_default():
                    model1 = load_model('mobilenet_model_80_20.hdf5')
                    class_prediction = model1.predict_classes(img)
                    print(class_prediction)


                if class_prediction[0] == 0:
                  product = "daisy"
                elif class_prediction[0] == 1:
                  product = "dandelion"
                elif class_prediction[0] == 2:
                  product = "roses"
                elif class_prediction[0] == 3:
                  product = "sunflowers"
                else:
                  product = "tulips"
                return render_template('./template/word_search.html', product = product, user_image = file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('./template/word_search.html')

if __name__ == "__main__":
    init()
    app.run()