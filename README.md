# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualizations/Label_Frequency.png "Label Frequency"
[image2]: ./visualizations/Traffic_Sign_Grayscale.png "Grayscale and Normalized"
[image3]: ./visualizations/Training_Visualization_Images.png "Training Images with Labels"
[image4]: ./German_Signs/German1.jpg "Traffic Sign 1"
[image5]: ./German_Signs/German2.jpg "Traffic Sign 2"
[image6]: ./German_Signs/German3.jpg "Traffic Sign 3"
[image7]: ./German_Signs/German4.jpg "Traffic Sign 4"
[image8]: ./German_Signs/German5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and built in python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34,799 images**
* The size of the validation set is **4,410 images**
* The size of test set is **12,630 images**
* The shape of a traffic sign image is **(32px, 32px, 3 channels)**
* The number of unique classes/labels in the data set is **43**

#### 2. Visualization of Dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the frequency of the labels for the training set of data.  In the training set we can see that there are varying frequencies from under 250 to well over 1750.  This can lead to a bias towards specific types of traffic signs (shape, color, or symbology)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the consistency of the test image's color and lighting was skewing the training of the neural network and adding a fair amount of processing time.

As a last step, I normalized the image data because it allowed for the network to train faster along with removing any outlying pixel values for intensity in the image.

Here is an example of a traffic sign image before and after grayscaling with normalization.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale and Normalized image  		| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				| Output 400									|
| Dropout				| keep prob = 0.5								|
| Fully Connected	    | Output 120    								|
| L2 Regularization		| B = 0.001								        |
| RELU					|												|
| Dropout				| keep prob = 0.5								|
| Fully Connected	    | Output 84     								|
| L2 Regularization		| B = 0.001								        |
| RELU					|												|
| Dropout				| keep prob = 0.5								|
| Fully Connected	    | Output 43     								|
| L2 Regularization		| B = 0.001								        |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
* batch size: 128
* number of epochs: 100
* learning rate: 0.0005
* keep probability for dropout layers: 0.5
* L2 regularization for fully connected layers: B * l2_loss(fully connected layer weights), where B = 0.001
* Variables were initialized with mu = 0.0 and signma = 0.1
* Optimizer: Adam

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99%**
* validation set accuracy of **95.7%**
* test set accuracy of **94%**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because it is a combination of the training set images and not directly in the image set that was trained with.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


