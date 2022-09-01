#!/usr/bin/python3
# coding: UTF-8

import numpy as np
import argparse
import cv2 as cv
import pandas as pd
filenn="Sp4.csv"
# Scaling data2 to the [0,1] range
# y = (x - min) / (max / min)

df = pd.read_csv('Sp4.csv', delimiter=',', header=0, usecols=[1,2])
# print(df)
def scaling(series):
    minimum = np.amin(series)
    maximum = np.amax(series)
    new = np.zeros(len(series))
    for i in range(len(series)):
        new[i] = (series[i] - minimum)/(maximum - minimum)
    return new
    
def Mat2Image(matrix, fileName):
    minimun = np.amin(np.min(matrix))
    maximun = np.amax(np.amax(matrix))
    diff = maximun-minimun
    print("max= %.1f, min= %.1f"%(minimun,maximun))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = 255*((matrix[i,j]-minimun)/(diff))
    cv.imwrite(fileName, matrix)
#
#
# # Binarization
# def binarization(matrix, threshold):
#     matrix[matrix < threshold] = 0.0
#     matrix[matrix >= threshold] = 1.0
#     return matrix

# Recurrence (Distance) Plot 
def rplot(series, err, bin=0):
    dim = len(series)
    rp = np.zeros((dim,dim))
    for x in range(dim):
        for y in range(dim):
            rp[x,y] = abs(series[x] - series[y])
    # if (bin == 1):
    #     rp = binarization(rp, err)
    return rp
    
def loadingData_x1(fileName):
    # df = pd.read_csv('Sp4.csv', delimiter=',', header=0, usecols=[1, 2])
    dataframe1 = pd.read_csv('Sp4.csv', delimiter=',', header=0, usecols=[1], engine='python') #first line is read as header
    # dataset1 = dataframe1.values
    return dataframe1
def loadingData_x2(fileName):
    dataframe2 = pd.read_csv(fileName, usecols=[2], engine='python') #first line is read as header
    # dataset2 = dataframe2.values
    return dataframe2


#### Main
if __name__ == "__main__":
    import os
    win_lenth = 15
    half_len_win = int(win_lenth / 2)
    dis_array = [[], []]
    win = []
    windows = []
    path_array = []
    win_min = 0
    # print(www)
    kj = win_lenth
    print('win_length', win_lenth)
    number = 0
    # for i in range(half_len, len(x), half_len):
    for i in range(half_len_win, len(df), half_len_win):
        win = []

        counter = i - half_len_win
        while (counter < kj):
            dataframe1 = loadingData_x1(filenn)
            win1 = dataframe1.iloc[counter:kj]
            win1 = win1.reset_index().values.ravel()
            win.append(win1)
            win1 = win1[1::2]
            # win1 = np.delete(win1, np.arange(0, win1.size, 3))
            print('==============')
            # win1=[x for x in win1 if x % 2 == 0]
            # dataframe1 = loadingData_x1(filenn)
            dataframe2 = loadingData_x2(filenn)
            win2 = dataframe2.iloc[counter:kj]
            win2 = win2.reset_index().values.ravel()
            win.append(win2)
            win2 = win2[1::2]
            # print(win1)
            # print(win2)
            windows.extend(win1)
            windows.extend(win2)
            counter = counter + 1
            kj = kj + 1
            number = number + 1

            # win1 = df.iloc[counter:kj]
            # win1 = win1.reset_index().values.ravel()
            # win.append(win1)
            # # win1 = win1[1::3]
            # win1 = np.delete(win1, np.arange(0, win1.size, 3))
            # print('==============')
            # # win1=[x for x in win1 if x % 2 == 0]
            # print(win1)
            # counter = counter + 1
            # kj = kj + 1
            # number = number + 1
            #
            print('kj')
            print(kj)
            # print(windows)
            if kj >= 53624:
                break;

            # dataset = win1
            dataset = windows
            windows = []
        # dataset = [1,445,778,23,73,2,-54,7,-34]
            print(dataset)
            # y_label = df[]
            y_label = dataframe1.iloc[kj]
            label = y_label.reset_index().values.ravel()
            ylabel = label[1]

            print("label_y: "+str(ylabel))
            new_dataset = scaling(dataset)
            rp = []
            rp = rplot(new_dataset, err=0.03, bin=0)
            first = -861
            for index in range(-861,2055,15):
               # "c" + index = -861 + 15
                if index <= ylabel < (index+15):
                    yclass = "c"+str(index)
                    print("class:" + str(yclass))
                    path = "data/"+str(yclass)

                    try:
                        os.mkdir(path)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                    Mat2Image(rp, "data/"+str(yclass)+"/RP"+str(number)+","+ str(ylabel)+","+str(yclass) + ".jpg")
    print('finish win creation')

    
    

