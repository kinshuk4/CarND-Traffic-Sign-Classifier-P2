#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/image1.jpg "Traffic Sign 1"
[image5]: ./test/image2.jpg "Traffic Sign 2"
[image6]: ./test/image3.jpg "Traffic Sign 3"
[image7]: ./test/image4.jpg "Traffic Sign 4"
[image8]: ./test/image5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kinshuk4/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43. All these are numbered from 0 to 42 in [signnames.csv](/signnames.csv)

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will reduce the length of the input from 3 to 1 channel and the classifier perform better working with only one parameter. Also, for the traffic signs, we generally don't need colors.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it improves the model as it gets all the input images have same treatment.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6  				    |
| Convolution 3x3	    | 1x1 strinde, outputs 10x10x16        		    |
| Max pooling	      	| 2x2 stride, outputs 5x5x16  				    |
| Flatten               | outputs 400                                   |
| Fully connected		| output = 120     								|
| RELU					|												|
| Fully connected		| output = 84     								|
| RELU					|												|
| Fully connected		| output = 43     								|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 16 epochs, batch size of 128 and the hyper parameter Learning Rate of 0.01.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.922 
* test set accuracy of 0.896

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

1. To begin with, I used the LeNet Model. The number of epochs were 16, learning rate was 0.01. The model was overfitted as the training accuracy was 98% and test accuracy was 89.6%. 

2. To overcome this, I used early termination while training the model. That reduced overfitting, and now the model has training accuracy of 95.4 and test accuracy of 86.9%. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h       		| 50 km/h   									| 
| No Entry     			| No Entry           							|
| Stop					| Stop											|
| 100 km/h	      		| Bumpy Road					 				|
| General Caution		| Traffic Signals      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares badly to the accuracy on the test set of 5 images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 60 km/h (probability of 0.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99998093e-01         			| 50 km/h   						| 
| 9.87646558e-07     				| 60 km/h 							|
| 8.65907339e-07					| 80 km/h							|
| 2.56162691e-09	      			| 30 km/h							|
| 3.86504266e-11				    | Priority Road      				|

For the second image, the model is relatively sure that this is a No Entry (probability of 0.5), and the image does contain a No Entry. The top five soft max probabilities were
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00]

       [17,  0,  1,  2,  3],
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| No Entry   						| 
| 0.00000000e+00     				| 20 km/h							|
| 0.00000000e+00					| 30 km/h							|
| 0.00000000e+00	      			| 50 km/h			 				|
| 0.00000000e+00				    | 60 km/h      						|

For the third image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were
       [  1.00000000e+00,   6.54085618e-13,   1.01315152e-14,
          5.45550247e-15,   1.75676182e-16],
                 [14, 33, 34,  5, 17],

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Stop sign							| 
| 6.54085618e-13     				| Turn right side 					|
| 1.01315152e-14					| Turn left side					|
| 5.45550247e-15	      			| 80 km/h			 				|
| 1.75676182e-16				    | No Entry  	    				|

For the fourth image, the model is relatively sure that this is a 100km/h sign (probability of 0.0), and the image does contain a stop sign. The top five soft max probabilities were

       [  3.75983685e-01,   2.99720675e-01,   1.11472495e-01,
          8.64256024e-02,   5.53182103e-02]
                 [22,  3, 25, 15,  5],
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.75983685e-01         			| Dangerous curve to the right 		| 
| 2.99720675e-01     				| 60 km/h 							|
| 1.11472495e-01					| Road work							|
| 8.64256024e-02	      			| No vehicles						|
| 5.53182103e-02				    | 80 km/h      			    		|

For the fifth image, the model is relatively sure that this is a General Caution (probability of 0.0), and the image does contain a stop sign. The top five soft max probabilities were
9.99829531e-01,   1.67821679e-04,   2.60027196e-06,
          2.59870486e-10,   2.50130437e-11

          [26, 29, 18, 30, 22]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99829531e-01         			| Traffic Signals   				| 
| 1.67821679e-04     				| Bicycles crossing 				|
| 2.60027196e-06					| General caution					|
| 2.59870486e-10	      			| Beware of ice/snow				|
| 2.50130437e-11				    | Bumpy road      					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


