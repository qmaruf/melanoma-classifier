import sys
sys.path.insert(0, '../src/')
import os
from flask import Flask
from flask import render_template
from flask import request
from utils import augmentations
from model import MelanomaDetector
import torch
from time import time
import cv2
import torch.nn as nn
import logging
from flask_debugtoolbar import DebugToolbarExtension
import numpy as np


op_sigmoid = nn.Sigmoid()
app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = '<replace with a secret key>'
toolbar = DebugToolbarExtension(app)
UPLOAD_FOLDER = './static/'
MODEL_PATH = '../src/melanoma-epoch=4-step=8284.ckpt'

@app.route('/', methods=['GET', 'POST'])
def upload():
    def get_prediction(image_location):
        image = cv2.imread(image_location)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentations['valid'](image=image)['image']
        images = torch.stack([image]*10)
        probas = op_sigmoid(MODEL(images))
        probas = probas.data.cpu().numpy()
        logging.warning(probas)
        proba_mean = np.mean(probas)
        proba_var = np.var(probas)
        prediction = 'positive' if proba_mean > 0.5 else 'negative'
        prediction = '%s(uncertainty %f)'%(prediction, proba_var)
        return prediction

    prediction = 'negative'
    image_location = None

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, str(time()) + image_file.filename)
            image_file.save(image_location)
            prediction = get_prediction(image_location)
            return render_template('index.html', prediction=prediction, image_location=image_location)
    else:
        return render_template('index.html', prediction='none', image_location=None)

if __name__ == '__main__':
    MODEL = MelanomaDetector.load_from_checkpoint(MODEL_PATH)
    # MODEL.eval()
    app.run(port=9991, host='0.0.0.0', debug=True)