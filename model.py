from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Dense, Flatten, Conv2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import sklearn.utils as utls
import imageio as imgio
import numpy as np
import cv2
import csv

path_img1 = 'data/IMG/'
path_data1 = 'data/driving_log.csv'
path_img2 = 'IMG/'
path_data2 = 'driving_log.csv'

# A method for merging the Udacity sample data with some of the corner cases driven by me in the simulation
def merge_driving_data(data, output_samples):
    with open(data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # Some of the images were corrupted - here they will be ignored
            if line[0] == 'center' or line == '/home/workspace/CarND-Behavioral-Cloning-P3/IMG/center_2020_02_26_22_01_15_785.jpg' or line == '/home/workspace/CarND-Behavioral-Cloning-P3/IMG/left_2020_02_26_22_01_15_785.jpg' or line == '/home/workspace/CarND-Behavioral-Cloning-P3/IMG/right_2020_02_26_22_01_15_785.jpg':   
                continue
            output_samples.append(line)
    return output_samples

# Combine the Udacity sample images with the images from the corner cases driven in the simulation
def copy_img(path1, path2): 
    for item in os.listdir(path2):
        s = os.path.join(path2, item)
        d = os.path.join(path1, item)
        if os.path.isdir(s):
            print(item)
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
# Python generator for data selection for each batch
def generator(samples, path_image, steering_correction, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        utls.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = path_image+'/'+batch_sample[0].split('/')[-1]
                name_left = path_image+'/'+batch_sample[1].split('/')[-1]
                name_right = path_image+'/'+batch_sample[2].split('/')[-1]
                loaded_center_image = imgio.imread(name_center)
                loaded_left_image = imgio.imread(name_left)
                loaded_right_image = imgio.imread(name_right)
                center_image = loaded_center_image
                center_angle = float(batch_sample[3])
                left_image = loaded_left_image
                left_angle = float(batch_sample[3]) + steering_correction
                right_image = loaded_right_image
                right_angle = float(batch_sample[3]) - steering_correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield utls.shuffle(X_train, y_train)

batch_size = 32
steering_correction = 0.05 # Steering angle will be corrected by this number for the side camera images

# Combine the driving data from 2 sources
samples_new = []
samples_first_source = merge_driving_data(path_data1, samples_new)
samples_final = merge_driving_data(path_data2, samples_first_source)

# Combine the images from 2 sources
copy_img(path_img1, path_img2)
        
# Split the training and validation data
train_samples, validation_samples = train_test_split(samples_final, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, path_img1, steering_correction, batch_size=batch_size)
validation_generator = generator(validation_samples, path_img1, steering_correction, batch_size=batch_size)

ch, row, col = 3, 160, 320
# Cropping parameters for rows and columns - this way the image will only represent the area relevant for driving (without the sky and car hood pixels
cropping_rows_top = 50
cropping_rows_bottom = 20
ch_cropped, row_cropped, col_cropped = ch, row-(cropping_rows_top+cropping_rows_bottom), col

## Create a CNN model. We are taking the Nvidia's End-to-End Deep Learning model
model = Sequential()
# Cropping
model.add(Cropping2D(cropping=((cropping_rows_top, cropping_rows_bottom), (0,0)), input_shape=(row,col,ch)))
# Normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row_cropped,col_cropped,ch_cropped), output_shape=(row_cropped,col_cropped,ch_cropped)))
# Convolutional layer 1
model.add(Conv2D(24, (5,5), subsample=(2,2), activation = 'relu', padding='valid'))
# Convolutional layer 2
model.add(Conv2D(36, (5,5), subsample=(2,2), activation = 'relu', padding='valid'))
# Convolutional layer 3
model.add(Conv2D(48, (5,5), subsample=(2,2), activation = 'relu', padding='valid'))
# Convolutional layer 4
model.add(Conv2D(64, (3,3), activation = 'relu'))
# Convolutional layer 5
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Flatten())
# Dropout layer
model.add(Dropout(0.5))
# Fully connected layers
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')
# Train the model
history_object = model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)

# Save the model
model.save('model.h5')