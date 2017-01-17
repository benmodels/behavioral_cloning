import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
HEIGHT = 80
WIDTH = 160
CHANNEL = 3
#from preprocess import image_preprocess

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
sequence_num = 0


@sio.on('telemetry')
def telemetry(sid, data):
    global sequence_num 
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    # Uncomment to save the images before feeding them to the NN. Useful for debugging or visualizations
    #image.save("debug/{0:0>10}.jpg".format(sequence_num))
    
    sequence_num += 1
    #image_array = image #mage_preprocess(np.asarray(image)).reshape((-1, 16, 32, 1))
    image_array = cv2.resize(np.asarray(image),(WIDTH,HEIGHT))/255.0 * 2.0 - 1.0
    image_array = image_array.reshape((1,HEIGHT,WIDTH,CHANNEL))
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(image_array, batch_size=1)[0])
    print("---------",steering_angle)
    # Very basic way to keep a constant speed
    if float(speed) < 15:
        throttle = 0.5
    else:
        throttle = 0.05
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
#    with open(args.model, 'r') as jfile:
    with open('model.json', 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
#    model.load_weights(weights_file)
    model.load_weights('model.h5')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
