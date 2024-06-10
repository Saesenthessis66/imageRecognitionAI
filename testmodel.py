import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('ai_generated_detector.h5')

img_path = 'notejaj.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'notejaj2.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'notejaj3.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'notejaj4.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'notejaj5.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'ejaj.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'ejaj2.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'ejaj3.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'ejaj4.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'ejaj5.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

img_path = 'michal.png'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The michal image is AI-generated.")
else:
    print("The michal image is real.")