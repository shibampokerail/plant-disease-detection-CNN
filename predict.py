import random

import tensorflow as tf
from train import class_names, test_ds, plt, model
import cv2
import numpy as np
from PIL import Image
from numpy import asarray

plants = ["apple", "corn", "potato", "tomato"]


def convert_input_image(path):
    # IMAGE = Image.open('apple.JPG')
    image_array = cv2.imread(path)
    img_shape = cv2.resize(image_array, (256, 256))

    return img_shape.reshape(1, 256, 256, 3)


model = tf.keras.models.load_model("Leaves-2convo32-4convo64-2dense64.h5")


def predict_disease(image_name):
    prediction = model.predict([convert_input_image(image_name)]).round()
    return prediction




