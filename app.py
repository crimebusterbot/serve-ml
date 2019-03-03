from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import json
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/3mar-weights.66-0.99.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')

def resize_with_pad(image, width, height):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # first we resize the whole image to the desired width using the right ratio
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # the resized image heigth and width
    (h, w) = resized.shape[:2]

    # short images need to match the desired heigth
    if h < height:
        BLACK = [0, 0, 0]
        resized = cv2.copyMakeBorder(resized, 0 , height - h, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else: 
        resized = resized[0:0+height, 0:0+width]

    return resized

def model_predict(img_path, model):
    img = cv2.imread(img_path)

    # Resize image
    img = resize_with_pad(img, 455, 700) 

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)

    return preds

@app.route('/test', methods=['GET'])
def test():
    return "TEST"

@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
           'uploads', secure_filename(f.filename))

        # Using Flask to save image
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        prediction = preds.tolist()

        # Remove the image from the temp uploads folder
        if os.path.exists(file_path):
            os.remove(file_path)

        return json.dumps({
            'fake': round(prediction[0][0], 3),
            'normal': round(prediction[0][1], 3),
            'good': round(prediction[0][2], 3),
        })
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    # To run local:
    # app.run(port=8080)
