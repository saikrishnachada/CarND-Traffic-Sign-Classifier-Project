# **Project3 : Traffic Sign Recognition**

### Objectives 

The objectives of this project are as follows:
* Load the pickled dataset consisting of the training, validation and testing images, obtained from [here](https://knowledge.udacity.com/questions/81091)
* Explore, summarize and visualize the data set 
* Perform pre-processing on the dataset and visualize the same
* Design a convolutional neural netowork architecture, train it and test the architecture on the provided data
* Use the model to make predictions on new images obtained from the internet
* Analyze the softmax probabilities of the new images
*  Document the results along with the necessary further improvements in a written report.   


[//]: # (Image References)

[image1]: ./examples/color_random_images.png "Random training set images with respective labels"
[image2]: ./examples/bar_plot.png "Bar plot explaining dataset"
[image3]: ./examples/gray_random_images.png "Grayscale images with respective labels"
[image4]: ./examples/internet_images.png "Images from internet"

---
### Load the data
I have used pickle module to load the dataset. The dataset consists of training, validation and testing images along with their respective labels.

---
### Data Set Summary & Exploration
#### 1. Summarize the dataset

I have used python length function (len()) to output the number of the test, validation and training examples. Additionally, the shape of the traffic sign images are found to be 32x32 with 3 color-channels. Finally, the number of unique labels have been obtained using set() method from python.   
The dataset can be summarized as follows...
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes in the dataset = 43

#### 2. Exploratory visualization of the dataset

Few images from the training dataset are randomly chosen and displayed in the below image. It can be seen from the figure that the images are color images and their respective labels are annotated on the images.    
![alt text][image1]

The visualization of the training dataset is showed using a bar plot, in which the counts (y-axis) of each unique labels (x-axis) are plotted below.  
![alt text][image2]

---

### Design and Test a Model Architecture

#### 1. Pre-processing the dataset

As a first step, I decided to convert the images to grayscale. The first reason being that the grayscale images contain less information for each pixel i.e. less parameters in compared to color images and can be easily trained using CNNs. Referring to Pierre Sermanet and Yann LeCun's paper on **Traffic Sign Recognition with Multi-Scale Convolutional Networks**, their results using grayscale traffic signal images have resulted in an increase in prediction accuracy. Here is an example of a random traffic sign images after grayscaling.

![alt text][image3]

In the next step, I have normalized the grayscale images by dividing the pixels by 255. Because, by normalizing the pixel values to range within [0-1] the convergence rate of CNN can be made faster. 

#### 2. Model Architecture

The proposed CNN model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution Layer 1    	| Output = 28x28x6 (1x1 stride, valid padding)|
| RELU	Activation function				|												|
| Max pooling	      	| Output = 14x14x6 (2x2 k-size, 2x2 stride, valid padding)			|
| Convolution Layer 2    	| Output = 10x10x16 (1x1 stride, valid padding)|
| RELU	Activation function				|	
| Max pooling	      	| Output = 5x5x16 (2x2 k-size, 2x2 stride, valid padding)			|
| Flatten				|	Output = 400
| Fully connected layer 1	    | Output = 300		|
| Fully connected layer 2	    | Output = 150		|  				| Fully connected layer 3	    | Output = 120		|
| Fully connected layer 4	    | Output = 43		|        				

#### 3. Training the model 
First of all, I have chosen the adam optimizer for the optimization after referring [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). Adam optimizer has been regarded as an efficient optimizer for deep learning because it is computationally efficient and can compute individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.  
There are various hyperparameters that have been tuned in the iterative way. The initial batch size introduced in the tutorial has been reduced to more than a half (BATCH_SIZE=55) to improve the ability of the learning process. The number of epochs has been set to 10, because after 10 epochs the model accuracy has not improved. The learning rate is maintained as 0.001, as it is already smaller. The weights are initialized randomly by choosing zero mean and variance (sigma=0.16). 

#### 4. Approach taken to improve the validation accuracy to be atleast 93% 

* **What was the first architecture that was tried and why was it chosen?**  
The classical convolutional neural network architecture of Yan LeCun discussed in the course work is the first architecture chosen. In general, the reason for choosing CNNs for traffic signs detection is inspired from the paper [(Yan LeCun)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The paper proved amazing prediction results by using CNNs. 
* **What were some problems with the initial architecture?**  
The classical CNN architecture discussed in the LeNet Lab has been trained on the handwritten digits. In the first attempt, testing the traffic sign predictions on the same model has resulted in 88% accuracy, which is great! However, to improve the accuracy further on the validation dataset, the architecture is improved further. 
* **How was the architecture adjusted and why was it adjusted?**   
The reasoning for choosing the hyperparameters has been discussed in the previous section. One of the major factor observed that lead to increase in the validation accuracy is by adding additional fully connected layers (2 & 3) to the CNN architecture. The activation function is chosen as ReLU.


My final model results were:
* Validation set accuracy of 93.4% 
* Test set accuracy of 91.3%
 

### Test Analysis on New Images

#### 1. Chosen Images

Here are five German traffic signs that I found on the web:  
![alt text][image2]  
My first impression was that it would be difficult for the model to predict 3rd and 4th images as they seem distorted.  

#### 2. Prediction results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Turn right ahead									| 
| Yield     			| Yield 										|
| Speed limit (50km/h)					| No entry										|
| Stop	      		| Stop					 				|
| Right-of-way at the next intersection		| Right-of-way at the next intersection	      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The wrongly predicted image was 3 (50km/hr speed limit sign). This has done comparatively fair job in predicting the images which it has never seen before.

#### 3. Softmax probabilities

The code for making predictions on my final model is located in the 104th cell of the Ipython notebook.

For the first image, the model was 100% sure that the image was a turn right ahead sign (probability of 1.0). For the second image, the model was 100% sure that the image was a yield sign (probability of 1.0). For the third image, the predictions were wrong. The model predicted with 68% accuracy that the image was a No entry sign and with 30% accuracy that the image was a 30km/hr speed limit sign. However, the original image was a 50km/hr speed limit sign. This suggests that the model has poorly learned its features. For the 4th and 5th images, the model has predicted with 99.8% and 100% accuracy that the images are stop sign and right-of-the-way at the next junction sign respectively. 

The summary of the top 5 softmax probabilities can be found in the below table. 

|    Original    | 1st best |   2nd best    | 3rd best   |   4th best    | 5th best   |
|----------------------|---------------|--------------------|------------|--------------------|------------|
|Turn right ahead                | 100%(Turn right ahead)          | 0%              |0%       | 0%              |0%       |
| Yield                   | 100%(Yield)          | 0%              |0%       | 0%              |0%       |
|  Speed limit (50km/h)                  |68%(No entry)      | 30%(30km/h)               |~ 0%       | ~ 0%             |~ 0%      |
| Stop                  | 99.8%(Stop)         | ~ 0%             | ~ 0%     |~ 0%             |~ 0%      |
| Right-of-way at the next intersection	                  | 100%(Right-of-way)         |0%              | 0% | 0%              |0%       |

---
### Possible improvements to my model

The prediction accuracy on the validation and test datasets can further be improved by trying out the **Suggestions to Make Your Project Stand Out!** section in the rubrix cube. 
