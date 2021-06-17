
## Behavioral Cloning Project
by Cristian Alberch submitted on June 2021 as part of Udacity Self Driving Car Engineer 

The goals / steps of this project are the following:

#### Use the simulator to collect data of good driving behavior.
#### Build, a convolution neural network in Keras that predicts steering angles from images.
#### Train and validate the model with a training and validation set.
#### Test that the model successfully drives around track one without leaving the road.
#### Summarize the results with a written report.

1. My project includes the following files:

- model.py containing the script to create and train the model.
- drive.py for driving the car in autonomous mode.
- model.h5 containing a trained convolution neural network.
- writeup_report.md this report summarizing the results.
- run1.mp4 showing autonomous driving.

## 1. Training model pipeline
The Python script to train the model is model.py which contains the pipeline to train the neural network. 


#### 1.1 Reading the data
The csv file containing the image file path and corresponding steering angle information is read using function log_reader. 
```
def log_reader(path):
    lines = []
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines
```

#### 1.2 Training and Validation data
The data contents are split into training and validation datasets (80-20% split) to ensure the model does not overfit during training and maintains a high accuracy when using test data it has never seen before.

```
train_data, validation_data = train_test_split(data, test_size=0.2)
```

#### 1.3 Breaking the data in batches 
The training and validation data is broken into batches of size 64 and fed into a generator function that iterates through the training and validation datasets iteratively until the entire dataset is fed into the training function. This is useful for memory management and also helps avoid overfitting.

```
train_generator = generator(train_data, batch_size = 64)
validation_generator = generator(validation_data, batch_size = 64)
```

#### 1.4 Data from multiple Cameras
The training data includes images taken from center, left and right cameras. The measurement steering angle is compensated depending on the image (left image is corrected by -ve 0.2, and right image by +ve 0.2).

```
for i in range(3): #iterate to get images: center, left, right
    image = cv2.imread(sample[i])
    images.append(image)
    measurement = float(sample[3])
    if i==0: #center image 
        correction = 0
    elif i==1: #left image
        correction = 0.2
    elif i==2: #right image
        correction = -0.2
    measurement = float(sample[3])+correction
    measurements.append(measurement)
```

#### 1.5 Data Augmentation
As the majority of the training race track is counter-clockwise, the training data has a left bias. In order to augment the training data, the training images and measurements are flipped along the vertical axis to simulate reverse direction and create double the amount of training images.
```
def data_augmentation (images, measurements):
    #augmenting images by flipping them horizontally
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)        
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements
```

## 2. Model Architecture

#### 2.1 Model Hyper-parameters
The neural network is a modified version of LeNet architecture and the following parameters modified to obtain the model with the highest validation accuracy improvement:
- Learning rate: Adam optimizer was used.
- Batch size: 64. Other batch sizes evaluted: 32, 128. 64 was found to be optimal in terms of validation error and training time.
- Activation layer "RELU". "ELU" with alpha = 0.1 was tried but did not produce improved training/validation accuracy.
- Dropout layer with dropout probability between the convolution and fully connected network parts. Varios configurations were tried with dropout layers in between layers with different dropout rates. The selected configuration provides a very fast training time.
- Learning rate: uses an Adam optimizer.
- Number of training epochs: 7 was selected as the validation plateaus after this epoch.

After multiple, only partially successful attempts at creating a customized neural network, the NVIDIA model architecture was selected as it provided the best output with lower training time per epoch and lower number of epochs to achieve a high accuracy.

The neural network model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160 x 320 x 3 RGB image                       |
| Normalize image     	| converts values from 0-255 to 0-1 range       |
|.......................|...............................................| 
| Convolution           | 2x2 stride, 'VALID' padding, outputs 24x24x5 	|
| Activation			| RELU          								|
|.......................|...............................................| 
| Convolution           | 2x2 stride, 'VALID' padding, outputs 36x5x5 	|
| Activation			| RELU			            					|
|.......................|...............................................| 
| Convolution           | 2x2 stride, 'VALID' padding, outputs 48x48x5 	|
| Activation			| ELU, alpha = 0.1								|
|.......................|...............................................| 
| Convolution           | outputs 64x3x3                             	|
| Activation			| RELU          								|
|.......................|...............................................| 
| Convolution           | outputs 64x3x3 	                            |
| Activation			| RELU          								|
|.......................|...............................................| 
| Flatten   			|                 								|
| Dropout               | 0.5 probability                               |
|.......................|...............................................| 
| Fully Connected     	| outputs 100 x 1	                            |
|.......................|...............................................| 
| Fully Connected     	| outputs 50 x 1	                            |
|.......................|...............................................| 
| Fully Connected     	| outputs 10 x 1                              	|
|.......................|...............................................| 
| Fully Connected     	| outputs 1 labels                            	|
|.......................|...............................................|



#### 2.2 Validating the model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model mean squared loss plot shows the validation accuracy declining with increasing number of epochs which plateaus at around epoch number 7.
![Training and validation loss](loss_function.png)

## 3. Recording Training Data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


#### 3.1 Normal Driving

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


Left Camera image             |  Center Camera image          |  Right Camera image
:-------------------------:|:-------------------------:|:-------------------------:
![](left_image.jpg)  |  ![](center_image.jpg)|  ![](right_image.jpg)



#### 3.2 Abnormal Driving

In order to train the model on corrective action in case the vehicle steers away from the road, I recorded the vehicle recovering from the left side and right sides of the road back to center.

#### 3.3 Compiling Training Data

The size of the final training set is 21,679 images.


## 4. Further Work
Areas for further work:

1. Increasing the training dataset may improve the model but will result in increased training time.

2. The training data can be further enhanced using image processing.

3. Multiple neural networks could be run in parallel with increased number of hyperparameter selections.

4. The neural network architecture could be modified to include 'LSTM' to provide a model that can interpret behavior considering consecurive frames.


Authors: Cristian Alberch https://github.com/cristiandatum

License: This project is licensed under the MIT License.