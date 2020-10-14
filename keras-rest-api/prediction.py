# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "C:/ShankersDocs/EDUCATION/RICE_Bootcamp_DataAnalytics/FinalProject_Img_Recognition_Flowers/Final_RICEproject_ImageRecognition_flowers/Prediction_images/dandelion_test1.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	print(r['prediction'])
	
# otherwise, the request failed
else:
	print("Request failed")