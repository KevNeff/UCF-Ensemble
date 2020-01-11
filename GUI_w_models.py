# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 03:54:31 2019

@author: kevin
"""
from tkinter import filedialog
from tkinter import *
import os
from PIL import ImageTk, Image
import cv2
import os
import sys
import matplotlib
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
    ucf_labels = os.listdir(os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/very_large_data/UCF-101'))

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
    dummy_path = os.path.join(video_file_path + '/UCF-101/BandMarching/v_BandMarching_g01_c01.avi')
    x, y = loadsingle3d(dummy_path, vid3d, color, skip)
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

    def test_model(test_video):
        test_video_path = os.path.join(video_file_path + test_video)
        x, y = loadsingle3d(test_video_path, vid3d, color, skip)
        x = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        prediction_3d = np.squeeze(model.predict(x, steps=len(x)))
        prediction_lstm = predictor.predict_values(test_video_path)
        prediction_both = np.squeeze(np.average([prediction_lstm, prediction_3d], axis = 0, weights=None))
        predicted_class = int(np.where(prediction_both == np.amax(prediction_both))[0])#Where in prediction_both is the highest softmax
        label_ensemble = str(ucf_labels[predicted_class])

        return label_ensemble

    print("Models loaded.")

    #Code for GUI initialization
    root = Tk()
    root.title("UCF Ensembler")
    background_image = PhotoImage(file="background.gif")
    background_label = Label(root, image=background_image, fg="blue")
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    w = background_image.width()
    h = background_image.height()
    root.geometry('%dx%d+0+0' % (w,h))
    font_type = "Arial"
    font_size = 12
    btn_width = 20

    #Code for the model selection row
    model_select_lbl = Label(root, bg="#bad9f2", anchor=W, font=(font_type, font_size))
    global filename#To be used by test_model in the model's code, not the GUI.
    filename = StringVar()
    filename = ''

    #code for Test and Result Row
    result_lbl = Label(root, bg="#bad9f2", anchor=W, font=(font_type, font_size))
    test_lbl = Label(root, bg="#bad9f2", anchor=W, font=(font_type, font_size))
    test_file = StringVar()
    test_file = ''
    def test_command():
        global img
        file_path = ''
        file_path = filedialog.askopenfilename(initialdir =
                                               os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/very_large_data/UCF-101'),
                                               title = "Select a test video",
                                               filetypes = (("videos","*.avi"),("all files","*.*")))
        file_tree = file_path.split("/")
        full_file_name = (os.path.split(file_path)[1])
        print("Testing: " + full_file_name)
        file_name = full_file_name.split('.')[0]
        file_ext = full_file_name.split('.')[1]

        if file_path == '':
            test_lbl.configure(text="No file selected.")
            result_lbl.configure(text="")
            img = ImageTk.PhotoImage(Image.open("blank.jpg"))
            image_lbl = Label(root, image = img)
            image_lbl.grid(column=1, row = 3)

        elif file_ext == 'avi':
            message = "Testing " + file_name
            test_lbl.configure(text=message)
            label = test_model("/UCF-101/" + file_tree[len(file_tree) - 2] + "/"+ full_file_name)
            result = "Label: " + label
            result_lbl.configure(text=result)

            vidcap = cv2.VideoCapture(file_path)
            vidcap.set(cv2.CAP_PROP_POS_MSEC,0)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite("video_render.jpg", image)

            img = ImageTk.PhotoImage(Image.open("video_render.jpg"))
            image_lbl = Label(root, image = img)
            image_lbl.grid(column=1, row = 3)
        else:
            test_lbl.configure(text="Invalid file selected.")
            result_lbl.configure(text="")
            img = ImageTk.PhotoImage(Image.open("blank.jpg"))
            image_lbl = Label(root, image = img)
            image_lbl.grid(column=1, row = 3)

    test_btn = Button(root, text = "Test from video sequence", command = test_command, bg="#9ed4ed", anchor=W, font=(font_type, font_size))


    #code for object placement


    test_btn.grid(column= 0, row=0)
    test_btn.config(width=btn_width)
    test_lbl.grid(column=1, row=1)
    result_lbl.grid(column=1, row=2)

    root.mainloop()

if __name__ == "__main__":
    Main()
