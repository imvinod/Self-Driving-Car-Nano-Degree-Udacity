import os
import csv
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
Given steering angle, plot th angles on a histogram with 75 bins
"""
def visualize(train_steering):
    train_steering = [float(x[3]) for x in train_data]
    plt.figure(figsize=(12, 8))
    plt.title('Steer Angle distribution')
    plt.hist(train_steering, 75, normed=0, facecolor='green', alpha=0.75)
    plt.ylabel('Images'), plt.xlabel('Angle')
    plt.show()

"""
Randomly perturb the brightness of th image. Convert from RGB2HSV before processing and convert back to RGB
"""
def purturb_brightness(img):
    image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    bright = 0.4*(2*np.random.uniform()-1.0)    
    image[:,:,2] = image[:,:,2]*(0.7 + bright)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

"""
Read and resize the image.
Remove top 50 and bottom 20 pixels from the image
Induce a Gaussian blur on the image
Perturb brighness of the image
resize the image to 66x200x3
"""
def process_img(path, flip = False):
    image = plt.imread(path)
    image = image[50:140, :]
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = purturb_brightness(image)
    if flip:
        image = cv2.flip(image,1)
    return image

"""
Pick each line in the data array, read the image, crop and resize the image
yield the image when size matches the batch size.
"""
def generate_val_data(data, path_prefix='', batch_size=128):
    while 1:
        batch_x, batch_y = [], []
        for line in data:
            image = plt.imread(line[0])
            image = image[50:140, :]
            image = cv2.resize(image, (200,66))

            steering_angle = float(line[3])
            batch_x.append(np.reshape(image, (1,66,200,3)))
            batch_y.append(np.array([[steering_angle]]))

            if len(batch_x) == batch_size:
                yield (np.vstack(batch_x), np.vstack(batch_y))
                batch_x, batch_y = [], []

"""
Pick each line in the data array, read the image, compensate the angle for the
left and right camera images, process the image (blur, crop, brightness and resize)
yield the image when size matches the batch size.
"""
def generate_train_data(data, path_prefix='', batch_size=128):
    
    while 1:
        batch_x, batch_y = [], []
        for line in data:
            rand = np.random.randint(0,3)
            if (rand == 0):
                steering_angle = float(line[3])
            elif rand == 1:
                steering_angle = float(line[3]) + 0.25
            else:
                steering_angle = float(line[3]) - 0.25            
            
            image = process_img(line[rand])
            batch_x.append(np.reshape(image, (1,66,200,3)))
            batch_y.append(np.array([[steering_angle]]))
            
            if len(batch_x) == batch_size:
                yield (np.vstack(batch_x), np.vstack(batch_y))
                batch_x, batch_y = [], []
"""
Create Nvidia end to end model
"""
def nvidia():
    keep_prob = 0.5    
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(66, 200, 3), output_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform', W_regularizer=l2(0.01)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
    
if __name__ == '__main__':
    csv_udacity_data = './data/driving_log.csv'
    csv_mouse_recovery = './recovery_mouse/driving_log_path_corrected.csv'
    
    # Read data from recovery dataset captured with a mouse (not keyboard)
    with open(csv_mouse_recovery, 'r') as file:
        reader = csv.reader(file)
        recovery = [row for row in reader][1:]
    
    # Read data from Udacity dataset
    with open(csv_udacity_data, 'r') as file:
        reader = csv.reader(file)
        udacity = [row for row in reader][1:]

    driving_data = udacity + recovery
    driving_data = shuffle(driving_data)        

    # Split dataset into Train and Test dataset
    train_data, val_data = train_test_split(driving_data, test_size=0.2, random_state=1)
    print('Trainging set:'+ str(len(train_data)) + 'Validation set: '+ str(len(val_data)))
    #visualize(train_steering)
    
    # Validation loss reached a platue at 7 epochs generally
    EPOCHS = 7
    model = nvidia()
    
    # Early stopping in case the losses stay the same or increase
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')
    
    # Fit model with data generators.
    model.fit_generator(
        generate_train_data(train_data, batch_size=128),
        samples_per_epoch=len(train_data)*4, nb_epoch=EPOCHS,
        validation_data =generate_val_data(val_data, batch_size=128),
        nb_val_samples=len(val_data)*3,
        callbacks=[earlystop]
    )
    
    # Save the model
    model.save('./model.h5')
    print("Model saved")