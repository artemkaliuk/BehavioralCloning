# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: Images/nn.jpg "Modified Nvidia model architecture"
[image2]: Images/architecture.png "Layer description"
[image3]: Images/2020_03_01_15_33_15_760.jpg "Original image - center driving"
[image4]: Images/2020_03_01_15_33_15_760_cropped.jpg "Cropped image - center driving"
[image5]: Images/center_2020_03_01_14_18_26_534.jpg "Recovery driving - roadside"
[image6]: Images/center_2020_03_01_14_18_30_966.jpg "Recovery driving - lane center"
[image7]: Images/loss.jpg "Training and validation loss"


## Rubric Points
### Here the [rubric points](https://review.udacity.com/#!/rubrics/432/view) will be considered individually, the implementation of the each point will be described

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 (model.h5.zip due to a large size of the model) containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used represents the Nvidia's End-to-End Deep Learning pipeline for autonomous driving: https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

This model has proven to provide robust performance in terms of learning steering angles based on the video data. The model is designed as a sequence of five convolutional layers with ReLu activations followed by three fully-connected layers. To avoid overfitting, an extra dropout layer with a dropout probability of 50% was added right after the flatten layer.


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 116). 

The model was trained and validated on two different data sets (model.py lines 86, 89, 90, 127)

#### 3. Model parameter tuning

The model used the Adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

At first, the training data provided by Udacity was utilized. However, as the model was not able to handle sharp turns in the simulation, data augmentation techniques were used - more on this in the next section!


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The modified Nvidia model was powerful enough to allow for a smooth and robust vehicle control. In contrast to the approach by Nvidia, the input images were not resized to 66x200, but cropped by 70 pixel rows in order to reduce the amount of noninformative pixels (car hood and sky) and at the same time speed up the training. The image width stayed unchanged at 320 pixels. Thus, the input format of the image is 90x320x3.

#### 2. Final Model Architecture

The final model architecture (model.py lines 101-122) can be seen here:
![alt text][image2]

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For a good quality center lane driving the Udacity sample data was used. Here are the examples of the original and cropped images (center lane driving):

![alt text][image3]![alt_text][image4]

As the model was still not able to keep the car on the track during sharp turns, recovery driving technique was used in order to make the model learn larger steering angles in cases when the car is getting too close to the road edge.

![alt text][image5]
![alt text][image6]

As the captured data was still biased in terms of centered driving (steering angles around 0°), more diversity in the data was needed. Thus, recovery driving scenarios were recorded several times for each of the sharp curves.

The data was randomly shuffled, split into the training and validation data set and put into training batches. Based on recommendations of the fellow Udacity students, 5 training epochs with Adam optimizer were used (thus no tuning of the learning rate needed). The resulting loss can be seen here:
![alt_text][image7]

As a result, the car drives smoothly around the track, showing an impressive lane centering behavior and confidently tackling the sharp curves. The video can be found in the root folder for this project (video.mp4).