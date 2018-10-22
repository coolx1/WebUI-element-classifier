# WebUI-element-classifier
# WebUI-element-classifier
This project is a part of an end to end software tool to assist wireframing. Wireframing is the process of creating a blueprint to a website. 
The blueprint is created either by hand or by using tools like Balsamiq, Axure etc.As a part of developing a tool to make wireframing easy and interactive
so that even a layman can do it, we were aked to develop a neural network model that can recognize hand drawn webUI elements. 
This project includes developing a CNN to classify hand drawn WebUI elements and a Flask server to run the trained CNN model.
Run the flaskServer.py and upload any image from the test_dataset folder to see the model at work.

## segmentation.py
Contains the code used for preprocessing the images

## training.py
Contains the code to create the CNN and training it

## running.py
Contains the code to get predictions from the CNN

## templates
Contains the HTML file that is displayed when the flask server is run.

## training_set and testing_set
Contains images for training and validation of CNN.

## model.json and model.h5
Contains the CNN model and the trained weights respectively.
