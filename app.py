from fastai.vision.all import *
from fastai.vision.widgets import *
import urllib.request
from fastai.imports import *
from fastai import *
from fastai.vision.data import ImageDataLoaders
from flask import Flask,render_template,request
from PIL import Image

import os
import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

def get_model():
    global model
    model = load_learner('export.pkl')
    print("Model loaded!")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def prediction(img_path):
    #new_image = load_image(img_path)
    pred = model.predict(img_path)
    
    return(pred[0])

get_model()

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static/', filename)
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)  


if __name__ == "__main__":
    app.run()
