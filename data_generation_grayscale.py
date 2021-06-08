# GRAYSCALE WITHOUT HU-MOMENTS

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import os
import tensorflow as tf
import pandas as pd

# Check the maximum image length for train set
mypath = "training/image_2"
frames = os.listdir(mypath)

max_size_train = 0

for f in frames :
    path = mypath + "/" + f
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.flatten()
    if len(im) > max_size_train :
        max_size_train = len(im)
    
max_size_train

# Check the maximum image length for test set
mypath = "testing/image_2"
frames = os.listdir(mypath)

max_size_test = 0

for f in frames :
    path = mypath + "/" + f
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.flatten()
    if len(im) > max_size_test :
        max_size_test = len(im)
    
max_size_test

im_size = min(max_size_test, max_size_train)

# Function to generate train set
def gen_grayscale_train() :
    
    X_train = open('new_X.csv', 'w')
    y_train = open('new_y.csv', 'w')
    
    mypath_X = "training/image_2"
    mypath_y = "training/calib"
    
    frames = os.listdir(mypath_X)
  
    for f in frames :
    
        path_1 = mypath_X + "/" + f
        f = f.replace(".png", ".txt" ) 
        path_2 = mypath_y + "/" + f
                
        # image pre-processing
        im = cv2.imread(path_1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im/255
        im = im.flatten()
        temp = np.zeros((1,im_size), dtype = "float32")
        if im_size > len(im) :
            temp[0, :len(im)] = im
            temp = temp[0]
        else:
            temp = np.array(im[:im_size])
        image_string = ','.join(['%.5f' % num for num in temp])
        
        # get label
        file = open(path_2, 'r')
        Lines = file.readlines()

        new_file_content = ""
        count = 0
        for line in Lines:
            count += 1
            a = line.replace(" ", ",")
            a = a.replace(":", "")
            new_file_content += a + "\n"

        file.close()

        file = open(path_2, 'w')
        file.write(new_file_content)
        file.close()
        
        lbls = np.array(pd.read_csv(path_2, names = range(0, 13, 1)).drop(0, axis = 1).fillna(0.0)).flatten()
        label_string = ','.join(['%.5f' % num for num in lbls])
        
        X_train.write(image_string)
        X_train.write("\n")
        y_train.write(label_string)
        y_train.write("\n")
        
    X_train.close()
    y_train.close()
    
# Function to generate test set
def gen_grayscale_test() :
    
    X_test = open('new_X.csv', 'w')
    y_test = open('new_y.csv', 'w')
    
    mypath_X = "testing/image_2"
    mypath_y = "testing/calib"
    
    frames = os.listdir(mypath_X)

    for f in frames :
    
        path_1 = mypath_X + "/" + f
        f = f.replace(".png", ".txt" ) 
        path_2 = mypath_y + "/" + f
        
        # image pre-processing
        im = cv2.imread(path_1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im/255
        im = im.flatten()
        temp = np.zeros((1,im_size), dtype = "float32")
        if im_size > len(im) :
            temp[0, :len(im)] = im
            temp = temp[0]
        else:
            temp = np.array(im[:im_size])
        image_string = ','.join(['%.5f' % num for num in temp])
        
        # get label
        file = open(path_2, 'r')
        Lines = file.readlines()

        new_file_content = ""
        count = 0
        for line in Lines:
            count += 1
            a = line.replace(" ", ",")
            a = a.replace(":", "")
            new_file_content += a + "\n"

        file.close()

        file = open(path_2, 'w')
        file.write(new_file_content)
        file.close()
        
        lbls = np.array(pd.read_csv(path_2, names = range(0, 13, 1)).drop(0, axis = 1).fillna(0.0)).flatten()
        label_string = ','.join(['%.5f' % num for num in lbls])
        
        X_test.write(image_string)
        X_test.write("\n")
        y_test.write(label_string)
        y_test.write("\n")
        
    X_test.close()
    y_test.close()
    
    
gen_grayscale_train()
gen_grayscale_test()
