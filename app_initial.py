
from flask import Flask, jsonify, render_template, url_for, request, redirect
import os
import h5py
import glob
from PIL import Image
from models import PredictFlower
import pandas as pd
from random import shuffle


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
my_pred = PredictFlower()

@app.route("/",  methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/predict",  methods=['GET', 'POST'])
def predict():

    print("upload click")
    if request.method == 'POST':
        if request.files.get('file'):
    
            images = request.files.getlist("file")
            print(f"Images: {images}")

            files = glob.glob(app.config['UPLOAD_FOLDER']+'/*')

            for f in files:
                os.remove(f)

            filenames = []

            for image in images:

                filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(filepath)
                filenames.append(image.filename)
                print(filenames)

            predictions = my_pred.predictor(filenames, app.config['UPLOAD_FOLDER'])

    # return render_template('word_search.html', predictions=predictions, user_image=files )    

            print(predictions)
        return jsonify({'result': 'success', 'predictions': predictions})
    return render_template('word_search.html', predictions= predictions)



if __name__ == '__main__':
    app.run(debug=True)
