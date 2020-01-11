#!/usr/bin/python
"""
Created on Sun Aug 11 02:52:54 2019

@author: kevin
"""
import os
import sys
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import videoto3d

def loadsingle3d(video_path, vid3d, color, skip):   
    file_tree = video_path.split("/")
    file_tree_len = len(file_tree)
    filename = file_tree[file_tree_len - 1]
    label = file_tree[file_tree_len - 2]
    X = []
    X.append(vid3d.video3d(video_path, color=color, skip=skip))
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), label
    else:
        return np.array(X).transpose((0, 2, 3, 1)), label


def Main():
    print("Loading Models.")   
    acc_3d = 0
    acc_lstm = 0
    acc_both = 0
    #Dataset Location and Labels
    video_file_path = os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/very_large_data')
    #video_path = os.path.join(video_file_path + '/UCF-101/BandMarching/v_BandMarching_g01_c01.avi')
    ucf_labels = os.listdir(os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/very_large_data/UCF-101'))
    test_file = os.path.join(os.path.dirname(__file__), 'test_list.txt')
    test_list = [line.rstrip('\n') for line in open(test_file)]
    
    #LSTM Load
    sys.path.append(os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master'))
    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels
    
    vgg16_include_top = False
    model_dir_path = os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/models/UCF-101')
    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)
    
    #3DCNN Load
    color=True
    skip=True
    nb_classes = 101
    img_rows, img_cols, frames = 32, 32, 10
    channel = 3 if color else 1
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    #Allows the shape to be compatible with single files rather than a list of files.
    temp_path = os.path.join(video_file_path + '/UCF-101/BandMarching/v_BandMarching_g01_c01.avi')
    x, y = loadsingle3d(temp_path, vid3d, color, skip)
    x = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    # Define 3Dmodel, allows input shape to change depending on chosen settings.
    batch_size = 128
    model = Sequential(name="3dcnn")
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        x.shape[1:]), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    weights_path_3d = os.path.join(os.path.dirname(__file__), '3d_cnn_action_recognition_master/3dcnnresult/ucf101_3dcnnmodel-gpu.hd5')
    model.load_weights(weights_path_3d)
    
    
    print("Models loaded.")    
    
    acc_3d = 0
    acc_lstm = 0
    acc_both = 0
    print("Predicting.")
    labelFile = open('labels.txt', 'w')

    first_predict = True
    def predict(test_video, acc_3d, acc_lstm, acc_both):
        test_video_path = os.path.join(video_file_path + test_video)
        x, y = loadsingle3d(test_video_path, vid3d, color, skip)
        x = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        
        prediction_3d = np.squeeze(model.predict(x, steps=len(x)))
        label_3d = int(np.where(prediction_3d == np.amax(prediction_3d))[0])
        label_3d = str(ucf_labels[label_3d])
        if str(label_3d == y):
            acc_3d += 1
            
        
        prediction_lstm = predictor.predict_values(test_video_path)
        label_lstm = int(np.where(prediction_lstm == np.amax(prediction_lstm))[0])
        label_lstm = str(ucf_labels[label_lstm])
        if str(label_lstm == y):
            acc_lstm += 1
        

        prediction_both = np.squeeze(np.average([prediction_lstm, prediction_3d], axis = 0, weights=None))
        predicted_class = int(np.where(prediction_both == np.amax(prediction_both))[0])#Where in prediction_both is the highest softmax
        predicted_class = str(ucf_labels[predicted_class])
        print("Predicted class: " + predicted_class)
        print("Actual class: " + y)
        if predicted_class == y:
            acc_both += 1
        for softmax in prediction_both:
            if first_predict == False:
                pFile.write("\n")
                labelFile.write("\n")
            pFile.write(str(softmax) + ", ")
            #first_predict = False

        labelFile.write(y + " " + label_3d + " " + label_lstm + " " + predicted_class)
        return predicted_class, acc_3d, acc_lstm, acc_both


    for test in test_list:
        label_ensemble, acc_3d, acc_lstm, acc_both = predict(test, acc_3d, acc_lstm, acc_both)
    labelFile.close()
    acc_3d = acc_3d / len(test_list)
    acc_lstm = acc_lstm / len(test_list)
    acc_both = acc_both / len(test_list)
    print("Predictions completed and saved.")
    print("Accuracy of 3d: " + str(acc_3d))
    print("Accuracy of lstm: " + str(acc_lstm))
    print("Accuracy of ensemble: " + str(acc_both))
    
if __name__ == "__main__":
	Main()
