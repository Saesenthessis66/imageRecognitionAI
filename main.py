import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
import cv2

data_dir = 'train'
valid_data_dir = 'valid'

batch_size = 16
epochs = 20

data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=batch_size)
valid_data =  tf.keras.utils.image_dataset_from_directory(valid_data_dir, batch_size=batch_size)

data = data.map(lambda x,y: (x/255,y))
valid_data = valid_data.map(lambda x,y: (x/255,y))

train_size = len(data)
val_size = int(len(valid_data)*.65) #val_size = len(valid_data)
test_size = int(len(valid_data)*.35)

train = data.take(train_size)
val = valid_data.take(val_size)
test = valid_data.skip(val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(8,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
model.summary() # good for sprawko

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train,epochs=epochs, validation_data = val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=20)
plt.legend(loc='upper left')
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    X, y =batch
    yhat = model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)

print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')

#for i in len(test) 
# for batch in test.as_numpy_iterator():
#     X, y =batch
#     yhat = model.predict(X)
#     pre.update_state(y,yhat)
#     re.update_state(y,yhat)
#     acc.update_state(y,yhat)
#     test.as_numpy_iterator().next()
# print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')

img = cv2.imread('ejaj.jpg') # hello read me
resize = tf.image.resize(img,(256,256))

yhat = model.predict(np.expand_dims(resize/255,0))

if yhat >0.5:
    print(f'Predicted image is AI generated')
else:
    print(f'Predicted image is NOT AI generated!')

model.save(os.path.join('models','imageclassifier.h5'))

new_model = load_model(os.path.join('models','imageclassifier.h5'))

yhatnew =  new_model.predict(np.expand_dims(resize/255,0))

if yhatnew >0.5:
    print(f'Predicted image is AI generated')
else:
    print(f'Predicted image is NOT AI generated!')