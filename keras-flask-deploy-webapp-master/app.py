from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#image processing libs
from skimage.transform import resize
from skimage import io

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/bloodsmear_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(100, 100))
    img = io.imread(img_path)

    # Preprocessing the image
    # x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    #BOC Inserted Custom Logic
    img_resized = resize(img, (100, 100), anti_aliasing=True)
    preds = model.predict(img_resized[np.newaxis,: ])
    #EOC 


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    #preds = model.predict(x)
    #print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    final_class = ""
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
##        pred_class = np.max(preds) #decode_predictions(preds, top=1)   # ImageNet Decode
##        print(preds)
##        print(pred_class)
##        pred = preds[0]
##        res = np.where(pred == pred_class)
##        print(res)
##        result = pred_class #str(pred_class[0][0][1])               # Convert to string'
        if preds[0][0] > preds[0][1]:
##            final_class = "WBC is MonoNuclear with Probability Confidence {} %".format(str(preds[0][0]*100))
            final_class = "WBC is MonoNuclear with Probability Confidence {0:.2f} %".format(preds[0][0]*100)
        else:
            final_class = "WBC is PolyNuclear with Probability Confidence {0:.2f} %".format(preds[0][1]*100)
        return final_class
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
