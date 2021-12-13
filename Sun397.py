from Datapreprocessing import Sun397_dataset
from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import time


# image data augmentation
def image_augmentation(x_train, y_train):
    random_img = []
    img_labels = []
    # taking 25 images per class for the image augmentation
    unique_labels = np.unique(y_train)
    for i in unique_labels:
        y_index = np.where(y_train == i)
        y_index = y_index[0]
        for j in range(25):
            y_inx = y_index[j]
            x_img = x_train[y_inx]
            y_values = y_train[y_inx]
            random_img.append(x_img)
            img_labels.append(y_values)
    random_img = np.asarray(random_img)
    img_labels = np.asarray(img_labels)

    # augmenting using ImageDataGenerator
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.2, horizontal_flip=True)
    train_generator = data_generator.flow(random_img, img_labels, batch_size=9925)
    # fetching the augmented data and its labels from train_generator
    random_img = train_generator[0][0]
    img_labels = train_generator[0][1]
    # combining the augmented image(25images per class(that is: 9925images)) with the original train dataset
    train_x = np.vstack((x_train, random_img)) # Total Number of samples in train dataset with argumented image = 29775
    train_y = np.hstack((y_train, img_labels))

    return train_x, train_y


# defining the VGG-16 model with places365 weights
def places_VGG(train_x, train_y, x_test, y_test):
    # defining the places365 weights path file
    places365_weights = 'vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base = VGG16(input_shape=(224, 224, 3), weights=places365_weights, include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    model = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    vgg_train_deep_features = model.predict(train_x)
    vgg_test_deep_features = model.predict(x_test)
    # saving the deep features of x_train and x_test
    np.savetxt("./VGG_places/X_train_places.csv", vgg_train_deep_features, delimiter=",", fmt='%s')
    np.savetxt("./VGG_places/X_test_places.csv", vgg_test_deep_features, delimiter=",", fmt='%s')
    np.savetxt("./VGG_places/Y_train.csv", train_y, delimiter=",", fmt='%s')
    np.savetxt("./VGG_places/Y_test.csv", y_test, delimiter=",", fmt='%s')

    return vgg_train_deep_features, vgg_test_deep_features


# defining the ResNet-101 using imagenet model
def Imagenet_ResNet(train_x, x_test):
    # setting the weights = 'imagenet'
    base = ResNet101(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    model = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    resnet_train_deep_features = model.predict(train_x)
    resnet_test_deep_features = model.predict(x_test)
    # saving the deep features of x_train and x_test
    np.savetxt("./Resnet/X_train.csv", resnet_train_deep_features, delimiter=",", fmt='%s')
    np.savetxt("./Resnet/X_test.csv", resnet_test_deep_features, delimiter=",", fmt='%s')

    return resnet_train_deep_features, resnet_test_deep_features


# combining the deep features extracted from the two models(VGG-16 and ResNet101)
def feature_combination(vgg_train_deep_features, vgg_test_deep_features, resnet_train_deep_features, resnet_test_deep_features):
    # combining the train deep features
    x_train_deep_features = np.hstack((vgg_train_deep_features, resnet_train_deep_features))
    # combining the test deep features
    x_test_deep_features = np.hstack((vgg_test_deep_features, resnet_test_deep_features))

    return x_train_deep_features, x_test_deep_features


# classifying using SVM classifier
def Svm_Classifier(x_train_deep_features, x_test_deep_features, train_y, y_test, start_train):
    # initializing the svm classifier
    classifier = svm.SVC(kernel='rbf', C=0.9)
    # training the classifier
    print("-------------------------------")
    print(" Training using SVM ")

    # train module
    classifier.fit(x_train_deep_features, train_y)

    y_tr_pred = classifier.predict(x_train_deep_features)
    training_accuracy = accuracy_score(train_y, y_tr_pred)

    # end train time
    end_train = time.process_time()
    training_time = end_train - start_train
    print("-------------------------------")
    print("Training accuracy:", training_accuracy)
    print("Training Time:", training_time)
    print("-------------------------------")

    # testing the classifier
    print("-------------------------------")
    print(" Testing using SVM")

    # start time
    start_test = time.process_time()

    # test module
    y_pred = classifier.predict(x_test_deep_features)
    testing_accuracy = accuracy_score(y_test, y_pred)

    # end time
    end_test = time.process_time()
    testing_time = end_test - start_test
    print("-------------------------------")
    print("Testing accuracy:", testing_accuracy)
    print("Testing Time:", testing_time)
    print("-------------------------------")


def main():
    # defining the train and test dataset path
    train_path = "./SUN397_dataset/Train"
    test_path = "./SUN397_dataset/Test"

    # loading train dataset
    x_train, y_train = Sun397_dataset(train_path) # x_train_shape = (19850 x 224 x 224 x 3)

    # loading test dataset
    x_test, y_test = Sun397_dataset(test_path) # x_test_shape = (19850 x 224 x 224 x 3)

    # start total_training time
    start_train = time.process_time()

    # image data argumentation on the train dataset
    train_x, train_y = image_augmentation(x_train, y_train)

    # extracting deep features using VGG-16 places365 weights
    vgg_train_deep_features, vgg_test_deep_features = places_VGG(train_x, train_y, x_test, y_test)

    # extracting deep features using ResNet101 imagenet weights
    resnet_train_deep_features, resnet_test_deep_features = Imagenet_ResNet(train_x, x_test)

    # feature combination
    x_train_deep_features, x_test_deep_features = feature_combination(vgg_train_deep_features, vgg_test_deep_features, resnet_train_deep_features, resnet_test_deep_features)

    # predicting using SVM classifier
    Svm_Classifier(x_train_deep_features, x_test_deep_features, train_y, y_test, start_train)

if __name__ == "__main__":
    main()

