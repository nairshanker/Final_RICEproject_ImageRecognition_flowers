from flask import Flask, jsonify, render_template, url_for, request, redirect
import json
import os
import glob
from model import PredictFlower
import pandas as pd
from random import shuffle


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
my_pred = PredictFlower()


@app.route("/",  methods=['GET', 'POST'])
def index():
    print("upload click")
    if request.method == 'POST':
        if request.files.get('file'):
#
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

            predictions = my_pred.call_predict(filenames, app.config['UPLOAD_FOLDER'])


    return render_template('index.html')


@app.route("/flower_category", methods=['POST'])
def find_category():
    data = {"success": False}
    if request.method == 'POST':
        print("flower_category is ")
        print("-------------------------------------")
        data = request.get_json()
        

        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)