import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# data_dir = 'train'
# valid_data_dir = 'valid'

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary')

# validation_generator = validation_datagen.flow_from_directory(
#     valid_data_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary')

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=15,
#     validation_data=validation_generator,
#     validation_steps=50)


# val_loss, val_acc = model.evaluate(validation_generator, steps=50)
# print(f"Validation loss: {val_loss}")
# print(f"Validation accuracy: {val_acc}")

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# model.save('ai_generated_detector.h5')

from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model('ai_generated_detector.h5')

# Predict on a new image
img_path = 'notejaj.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("The image is AI-generated.")
else:
    print("The image is real.")

