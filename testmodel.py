import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import PySimpleGUI as sg
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('ai_generated_detector.h5')


file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
def main():
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
    ]
    window = sg.Window("Image Viewer", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                i = Image.open(values["-FILE-"])
                i.thumbnail((400, 400))
                bio = io.BytesIO()
                i.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

                img_path = filename
                img = image.load_img(img_path, target_size=(150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                prediction = model.predict(img_array)
                if prediction[0] < 0.5:
                    print("The image is AI-generated.")
                else:
                    print("The image is real.")

    window.close()
if __name__ == "__main__":
    main()







