# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:06:55 2019

@author: kevin
"""
import os
from sklearn.model_selection import train_test_split
def Main():

    video_file_path = os.path.join(os.path.dirname(__file__), 'keras_video_classifier_master/demo/very_large_data')
    data_set_name = 'UCF-101'

    input_data_dir_path = video_file_path + '/' + data_set_name
    y_samples = []
    x_samples = []
    
    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + '/' + f
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                x = 0
                y = data_set_name + '/' + f + '/' + ff
                y_samples.append(y)
                x_samples.append(x)
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=.3,
                                                        random_state=43)
    trainFile = open('train_list_master.txt', 'w')
    count = 0
    y_length = len(Ytrain)
    for Y in Ytrain:
        trainFile.write(str(Y))
        if count < len(Ytrain) :
            trainFile.write('\n')
            count += 1
    trainFile.close()

    testFile = open('test_list_master.txt', 'w')
    count = 0
    y_length = len(Ytest)
    for Y in Ytest:
        testFile.write(str(Y))
        if count < len(Ytest) :
            testFile.write('\n')
            count += 1
    testFile.close()
    
if __name__ == "__main__":
	Main()
