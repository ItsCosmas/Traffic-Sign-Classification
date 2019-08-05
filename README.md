# Traffic Sign Classifier

A project that stretched my understanding of Neural Networks especially CNN’s and their successful use in Computer Vision. I built the network on Keras framework running on top of Tensorflow backend, trained on Google Colab using thousands of images in the training set and attained an accuracy of 97.8% against the test set. I further tested the model on some independent external data and it performed well.

The training source code is available in a notebook file containing all the source code, visualizations and notes: https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/Traffic_Sign_Classification.ipynb
The trained model file is a `.h5` file named `my_model.h5`

  - Keras
  - Python

### Python Libraries

Dillinger uses a number of open source projects to work properly:

* Open CV
* Numpy
* Pandas
* Matplotlib
* and more ...

##### Training
I trained this model in an online cloud instance on [Google Colab](https://colab.research.google.com/)

##### Key points
I made use of Histogram Equalization technique to standardize lighting in all our images.
I used an ImageGenerator to help show different angles and views of the same data set to the model for it to better identify the features.

###### Building the Network
I used and tweaked a leNet model to attain desired results:
```
def better_model():
  model = Sequential()
  # add the convolutional layer
  #filters, size of filters,input_shape,activation_function
  model.add(Conv2D(60,(5,5), input_shape= (32,32,1), activation = 'relu'))
  model.add(Conv2D(60,(5,5), input_shape= (32,32,1), activation = 'relu'))
  #pooling layer
  model.add(MaxPooling2D(pool_size = (2,2)))
  # add another convolutional layer
  model.add(Conv2D(30, (3, 3) , activation = 'relu'))
  model.add(Conv2D(30, (3, 3) , activation = 'relu'))
  # pooling layer
  model.add(MaxPooling2D(pool_size = (2,2)))
  
  #model.add(Dropout(0.5))
  
  #Flatten the image to 1 dimensional array
  model.add(Flatten())
  #add a dense layer : amount of nodes, activation
  model.add(Dense(500, activation = 'relu'))
  # place a dropout layer
  #0.5 drop out rate is recommended, half input nodes will be dropped at each update
  model.add(Dropout(0.5))
  # defining the ouput layer of our network
  model.add(Dense(num_classes, activation = 'softmax'))
  
  
  #Compile Model
  # we use Adam optimizer with a learning rate 0f 0.01
  # A categorical_crossentropy'
  
  model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
  return model
 ```
 ##### Overfitting and Underfitting
![Overfitting and Underfitting curve](https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/overfitting.png) <br />
![Overfitting and Underfitting curve](https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/underfitting.png) <br />

### Testing the model
If you wish to test this model locally, please refer to this jupyter notebook file:
https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/Test%20Model%20Using%20local%20image%20in%20OpenCV.ipynb

If you wish to test this model with images from the internet, please refer to this  notebook:
https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/Test%20Model.ipynb
