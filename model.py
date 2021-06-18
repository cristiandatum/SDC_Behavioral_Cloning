import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import ndimage

from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers import ELU



#read in each row/line from driving_log.csv
def log_reader(path):
    lines = []
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def data_augmentation (images, measurements):
    #augmenting images by flipping them horizontally
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)        
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements

def network_model(row,col,chan):
    model = Sequential()    
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, chan), output_shape=(row, col, chan)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5)) #remove the dropout - added again
    model.add(Dense(units=100, activation='relu'))    
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    return model

def generator(data, batch_size):
    num_samples = len(data)
    correction = 0
    while True: # Loop forever so the generator never terminates
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples:
                for i in range(3): #iterate to get images: center, left, right
                    orig_image = cv2.imread(sample[i])
                    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                    images.append(orig_image)
                    measurement = float(sample[3])
                    if i==0: #center image 
                        correction = 0
                    elif i==1: #left image
                        correction = 0.2
                    elif i==2: #right image
                        correction = -0.2
                    measurement = float(sample[3])+correction
                    measurements.append(measurement)
            images, measurements = data_augmentation (images, measurements)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

path = '/home/workspace/CarND-Behavioral-Cloning-P3/training_data'
data = log_reader(path)

### split data into training and validation
train_data, validation_data = train_test_split(data, test_size=0.2)

### train and validation generator functions
batch_size = 64
train_generator = generator(train_data, batch_size)
validation_generator = generator(validation_data, batch_size)

model = network_model(row=160, col=320, chan=3)
model.summary()

model.compile(loss='mse', optimizer = 'adam')

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_data)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_data)/batch_size), 
            epochs=7, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss function.png')