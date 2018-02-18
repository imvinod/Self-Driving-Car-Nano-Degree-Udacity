# **Behavioral Cloning**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior and Recovery behavior.
* Build, a Nvidia end-end convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---
### Introduction

Behavioral cloning refers to the process of capturing a skill/action from a human being and be able to reproduce it by a computer program. In the context of this project, A human being (Me) drives a car around a simulator track. Three cameras record pictures which are then used by a computer program to process and learn to mimic the driving skills observed on the track. Our task is to build a deep learning model that can drive on the simulator track without falling off the track.

This project requires you to have a decent understanding of Deep learning principals (Optimization, Regularization, Common deep learning Architectures, Identify bias, over-fitting etc.), Data processing and working knowledge of deep learning framework Keras.

### 1. Collecting data
This is arguably the most crucial part of this project. I found that tuning the model with different techniques did very little to improve the track performance with data that did not well represent a good driving behavior.

Having not played a video game ever since NFS, I immediately started driving around on the simulator with my keyboard. I recorded driving in the forward as well as opposite direction so that the model will have data to better generalize to different circumstances. I also collected Recovery data. Recovery is a situation where your car is on the edge and you bring it to the center of the track where it should ideally drive along. If the model finds itself being on the edge, it will have an opportunity to lear how to get back to the center from the Recovery data.

<center>Recovery Image</center>
<p style="text-align:center;">
<img src="./img/recovery.jpg?raw=true"center=True width="300px"></p>

However, the data I collected performed poorly on the same model that worked well with Udacity data. It took many hours to come to this realization. I tried driving around the track again with my mouse. This time it was better but not as good as Udacity data.

I decided to go with Udacity data combined with recovery data. This is how the data distribution looks like.

<center>Udacity data and recovery data combined distribution</center>
<p style="text-align:center;">
<img src="./img/udacity_and_mouse_recovery.png?raw=true"center=True width="400px"></p>

It can be observed that about half of the data is of steering angle 0.0. This is because a vast stretch of the track is straight or of little curvature. Our model can get influenced disproportionately to driving straight and may not do very well on the curves. We may choose to balance the data by discarding some percentage of it or adding data to compensate for this bias, as we will see later.

We could also collect data from track 2 but I wanted to see for myself if we would be able to tweak the model to generalize to track 2 without having any data samples from it but so far I have not been successful at that.

#### Processing the Data

There are three cameras, center, left and right. Steering angle is provided only for the front camera.  However, As mentioned in the course, we can use the left and right camera images by compensating for the angle by +0.25 or -0.25 for left and right camera images respectively. This is help us with additional data. We can also flip the camera and it's respective steering angle for additional data but I found this to worsen the performance in my experiments. In principal it should be helpful.

Images are of size 320x160. The top portion is just trees and rocks that provide no beneficial information but only distracts the model. The bonnet of the car is also visible. So we remove top 50 an bottom 20 pixels from the image. We also resize it to 200x66 as required by the Nvidia model.

<center>Orignal images</center>
<p style="text-align:center;">
<img src="./img/orignal.png?raw=true" width="900px"></p>

<center>Cropped images</center>
<p style="text-align:center;">
<img src="./img/cropped.png?raw=true" width="900px"></p>

In order to reduce the model's tendency to over-fit the data, we try to perturb the images. The first option I used was to randomly change the brightness of the image. This helps model be robust to changes in lighting conditions. The second option is to use Gaussian blur. This helps model to be robust against the minute details but focus on the structure.

<center>Random Brightness</center>
<p style="text-align:center;">
<img src="./img/brightness1.png?raw=true" width="900px"></p>

<center>Random Blurring</center>
<p style="text-align:center;">
<img src="./img/blur1.png?raw=true" width="900px"></p>

### 2. Model

I decided to go with Nvidia model straight away as suggested in the course. The model architecture is as shown below.

<center>Nvidia model</center>
<p style="text-align:center;">
<img src="./img/nvidia.png?raw=true" width="400px"></p>

The first layer is the normalization layer. Normalization helps optimizer converge to the minima faster. This is followed by three Convolution layers with 5x5 filters with glorot_uniform (Also called Xavier Initialization).Three Convolution layers with 3x3 filters follow. The output of the convolution layer is flattened and fully connected with dropout before each layer for Regularization. The final layer is a single neuron that outputs single value estimate of the steering angle. We play around with keep_probablity later. We use ELU for activation function and Adam for optimizer. The loss function is MSE (Mean squared error) best suited for regression problems. Nvidia paper suggests converting your RGB image to YUV color space but I found it to deteriorate performance.

### 3. Training and Validation

An important part of training is generating data into batches on demand. Keras generator is a convenient function for our purpose. Our Training set data generator function accepts array of information from CSV file that contains driving log information like steering angle and image location path. The generator function compensates for the angle when it randomly picks one of the images among left, center and right cameras. These images are then processed, resized and batched. When the batch equals the batch_size,  we yield the batch. Note that all the above process run in a while loop. For validation data generator function we skip the image processing part.

We use a batch size of 128. We use EPOCH values of between 3-20 based on whether we would like to tackle bias or over-fitting. We also use a great handy callback function EarlyStopping provided by Keras. We configure EarlyStopping callback to watch validation loss and exit training epochs if it stagnates.

We save the model in .h5 format. The model file is passed as argument to drive script provided by Udacity to run simulator in autonomous mode controlled by your model.

Below is a table of a handful of my trails from my training efforts.

| Data                                                                                                        | Cameras            | Fliped          | Blur          | Random <br> Brightness | Dropout | Epoch | Result                                                                                                           |
|-------------------------------------------------------------------------------------------------------------|--------------------|-----------------|---------------|------------------------|---------|-------|------------------------------------------------------------------------------------------------------------------|
| - Keyboard, forward and opposite - Train: 3000 Val:766                                                      | Front              | No              | No            | No                     | 0.5     | 3     | 9mph Smooth, dives into the river                                                                                |
| - Keyboard- Train: 3000 Val:766                                                                             | Front              | No              | No            | No                     | 0.5     | 7     | 9mph Wobbly, into the river                                                                                      |
| - Keyboard- Train: 3000 Val:766                                                                             | Front              | No              | No            | No                     | 0.5     | 10    | Wobbly, makes it until bridge                                                                                    |
| - Keyboard- Train: 3000 Val:766 - Cropped top 50 and lower 20                                               | Front              | No              | No            | No                     | 0.5     | 15    | Into the river                                                                                                   |
| - Keyboard-forward and opposite - Train: 3000 Val:766- Cropped top 50 and lower 20                          | Left, Front, Right | No              | No            | No                     | 0.5     | 15    | Into the river                                                                                                   |
| - Keyboard-forward and opposite - Train: 3000 Val:766 Cropped top 50 and lower 20                           | Left, Front, Right | Yes, Random 50% | No            | No                     | 0.5     | 7     | Worse, Left biased, Into the river                                                                               |
| - Keyboard-forward and opposite - Train: 3000 Val:766 Cropped top 50 and lower 20                           | Left, Front, Right | Yes, Random 50% | No            | No                     | 0.5     | 15    | Worse than prior, Into the river                                                                                 |
| - Udacity data- Cropped top 50 and lower 20 - Train:6428 Val: 1608                                          | Left, Front, Right | No              | No            | No                     | 0.5     | 3     | Hits the bridge                                                                                                  |
| - Udacity data- Cropped top 50 and lower 20 - Kept 20% of data with angle 0.0 - Train:4428 Val: 1107        | Left, Front, Right | No              | No            | No                     | 0.5     | 3     | Hits bridge but Reaches goal                                                                                     |
| - Udacity data and Keyboard data - Cropped top 50 and lower 20 - Train 7400 Val: 1900                       | Left, Front, Right | No              | No            | No                     | 0.5     | 3     | Worse than just alone with Udacity data                                                                          |
| - Udacity data and Keyboard Recovery data- Cropped top 50 and lower 20- Train 7400 Val: 1900                | Left, Front, Right | No              | No            | No                     | 0.5     | 3     | Reaches goal but not smooth                                                                                      |
| - Udacity data and Keyboard Recovery data- Cropped top 50 and lower 20- Train 7400 Val: 1900                | Left, Front, Right | No              | No            | Yes                    | 0.5     | 3     | Wobbly, Fell into the ditch                                                                                      |
| - Udacity data and Keyboard Recovery data- Cropped top 50 and lower 20- Train 7400 Val: 1900                | Left, Front, Right | No              | No            | Yes                    | 0.5     | 20    | Wobbly, Fell into the ditch                                                                                      |
| - Udacity data and Keyboard Recovery data- Cropped top 50 and lower 20- Train 7400 Val: 1900                | Left, Front, Right | No              | Guassian blur | No                     | 0.5     | 3     | Wobbly, Fell into the ditch                                                                                      |
| -Data captured with mouse-Cropped top 50 and lower 20-Train 6000 Val: 1500                                  | Left, Front, Right | No              | No            | No                     | 0.5     | 3     | Left biased, Fell into the ditch                                                                                 |
| -Data captured with mouse-Cropped top 50 and lower 20-Train 6000 Val: 1500                                  | Left, Front, Right | Yes             | No            | No                     | 0.5     | 7     | Right biased, Fell into the ditch                                                                                |
| - Udacity data and Recovery data captured with mouse. - Cropped top 50 and lower 20 - Train: 7648 Val: 1912 | Left, Front, Right | No              | No            | No                     | 0.5     | 7     | Track 1: Reaches goal smoothly at 9mph and 20 mph Track 2: Left biased, travels a bit, hits a rock wall.   |
| - Udacity data and Recovery data captured with mouse.- Cropped top 50 and lower 20- Train: 7648 Val: 1912   | Left, Front, Right | No              | Yes           | No                     | 0.5     | 7     | Track 1: Reaches goal smoothly at 9mph - Track 2: Left biased, travels a bit, hits a rock wall.            |
| - Udacity data and Recovery data captured with mouse.- Cropped top 50 and lower 20- Train: 7648 Val: 1912   | Left, Front, Right | No              | Yes           | Yes                    | 0.5     | 7     | Track 1: Reaches goal smoothly at 9mph- Track 2: Left biased, little better, hits a rock wall.                   |
| - Udacity data and Recovery data captured with mouse.- Cropped top 50 and lower 20- Train: 7648 Val: 1912   | Left, Front, Right | No              | Yes           | Yes                    | 0.2     | 7     | Track 1: Reaches goal but bit wobbly at 9mph- Track 2: Left biased, almost same as before, hits a rock wall.     |
| - Udacity data and Recovery data captured with mouse.- Cropped top 50 and lower 20- Train: 7648 Val: 1912   | Left, Front, Right | Yes             | Yes           | Yes                    | 0.2     | 7     | Track 1: Dives into the lake - Track 2: Left biased, Worse than earlier.                                         |
| - Udacity data and Recovery data captured with mouse.- Cropped top 50 and lower 20- Train: 7648 Val: 1912   | Left, Front, Right | No              | Yes           | Yes                    | 0.5     | 7     | Track 1: Reaches goal smoothly at 9mph and 20 Mph - Track 2: Left biased, hits a rock wall. |


### 4. Test and results

Although, there is a lot of improvements we can make to the model on various fronts, the model successfully completes Track 1 for several laps at 20 mph. It reaches a distance on track 2 before crashing into the wall.

#### Scope for improvement
- The major take away from this project is that I appreciate an appropriate amount the importance of a well balanced data. A model can only be tuned so much. Without a dataset of a balanced distribution, the model would not perform to its best potential. I found that the data collected from my keyboard performed bad with the same model that works great on Udacity data. Data collected using mouse performed better than keyboard collected data. Studying data distribution is a promising avenue. If data points are concentrated disproportionately on one direction, the model tends to favor more towards it even when it isn't the appropriate action. However, tailoring the data distribution to work well on a particular track will result in model performing worse on a different track. A mask of finely balanced proportion for each steering angles that works well for both the tracks may come handy during data collection. Instead of blindly discarding a chunk of data, it may be useful to get more data to compensate for the bias in the proportion. Multiplying this mask with any dataset should tell us the gaps in the dataset proportions.
- Once we have a balanced data, the model will be in position to be tweaked. Particularly the regularization techniques like dropout, l2 regularization, image processing like cropping, jitter, blurring, sheering and other augmentation techniques can be used to generalize the model.
- The keyboard method of driving the car doesn't perfectly replicate the real steering control while driving a real car. Actual steering is much more smooth and continuous than abrupt like with the keyboard. Being a hardware guy, I am excited by the idea of setting up a simple steering mechanism that feeds steering angle to the simulator of the exact angle it currently is in. Either an encoder or a variable resistance connected to an Arduino should be able to read angles off of the steering. These values can be transmitted to the simulator via USB. It is to be seen how easy it is to hack the simulator module that takes in key board inputs. Although, this exercise doesn't augment skills in deep learning, it is something that might be cool.
- I use AWS for my projects. It is cumbersome to transfer files to and fro. It would be convenient to have a GPU set up locally and scripts to automate the repeated no-brainer tasks may also be the need of the hour.


### Conclusion

The project was an excellent exercise in understanding the importance of data collection, augmentation and processing. It was also a fun exercise that gave a small peek into magnitude of challenges a self driving car engineer would face in the real world. I am grateful for Udacity SDC team for such a great simulator and exercise in general. I hope to come back to this exercise and implement the improvements those of which I have noted down.
