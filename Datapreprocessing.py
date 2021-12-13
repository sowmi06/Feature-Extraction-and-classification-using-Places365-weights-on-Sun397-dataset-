from sklearn.preprocessing import LabelEncoder
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import random
import numpy as np
import os


# loading Sun397 dataset
def Sun397_dataset(root):
    x = []
    y = []
    # reading file from parent path
    for parentPath, subdirs, files in os.walk(root):
        for subdir in subdirs:
            path = parentPath + "/" + subdir
            label = subdir.split(".")
            datafile = os.listdir(path)
            for file_name in datafile:
                imgPath = path + '/' + file_name
                # loading the images
                img = image.load_img(imgPath, target_size=(224, 224))
                # converting images to array
                fileContent = image.img_to_array(img)
                # preprocessing the sample
                fileContent = preprocess_input(fileContent)
                x.append(fileContent)
                y.append(label[0])
                # printing the image shape and its labels
                print(fileContent.shape)
                print(label[0])
    x = np.asarray(x)
    y = np.asarray(y)
    # using label encoder for the y_train and y_test
    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)
    # Shuffle the whole dataset
    total_data = list(zip(x, y))
    random.shuffle(total_data)
    X, Y = zip(*total_data)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y







