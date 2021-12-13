# Feature Extraction and classification on Sun397 dataset using Places365 weights 

## Project Description
The Sun397 dataset contains 108,753 images of 397 categories, used in the Scene UNderstanding (SUN) benchmark. The number of images varies across categories, but there are at least 100 images per category. For this experiment, 50 images per class are taken randomly for the training set, that is 19,850 images for training and 50 images per class for the testing set with the image size of 224x224x3. The train and test data together is 13.4 GB in total, after preprocessing.

Initially, the images in the training set are arguments using the ImageDataGenerator from TensorFlow to train the network with several variations in the image. Deep features are extracted using the two DCNN models, ResNet-101 with  ImageNet weights and VGG-16 with places365 weights on both the train and test sets. The extracted deep features are then combined resulting in a feature with the shape of 29775 x 2560. Finally, the Support Vector Machine(SVM) classifier is used to classify the images on the extracted deep features. The average accuracy from 3 runs is 64.77%

## Configuration Instructions
The [Project](https://github.com/sowmi06/Feature-Extraction-and-classification-using-Places365-weights-on-Sun397-dataset-.git) requires the following tools and libraries to run the source code.
### System Requirements 
- [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/install/)
    - Python version 3.6.0 – 3.9.0
    - pip 19.0 or later 
    - Ubuntu 16.04 or later (64-bit)
    - macOS 10.12.6 (Sierra) or later (64-bit)
    - Windows 7 or later (64-bit) 
 
- Python IDE (to run ".py" file)
    - [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), [Spyder](https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/spyder/doc/installation.html) or [VS code](https://code.visualstudio.com/download)

### Tools and Library Requirements 
- [TensorFlow](https://www.tensorflow.org/install/pip)
    
- [Numpy](https://numpy.org/install/)

- [Scikit-learn](https://scikit-learn.org/stable/install.html) 
  
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)



