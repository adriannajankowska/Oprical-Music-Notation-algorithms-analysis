import os
import pygame
import numpy as np
from pygame.locals import*
from PIL import Image
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import logging
# ==============================================================================================
def StringToInteger(labels): 
    result = []
    dictionary = []
    for lab in labels:
        if not lab in dictionary:
            dictionary.append(lab)

    for lab in labels:
        for i in range(len(dictionary)):
            if dictionary[i] == lab:
                result.append(i)
    return result
# ==============================================================================================
datasetPath = os.getcwd() + "\\HOMUS"
logging.basicConfig(filename='log_mainScript_easy.log', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
labels = []
valuesFromOneFile = []    # lista list
maxLength = 0
fileCounter = 0
for subdir, dirs, files in os.walk(datasetPath):
    for txt in files:
        sumOfObjects = 0
        values = []
        filepath = subdir + os.sep + txt
        if filepath.endswith(".txt"):
            filename = txt.replace(".txt", "")
            fileCounter += 1
            pointsTable = []     
            
            with open(filepath, 'r') as source:
                labels.append(source.readline().replace('\n', ""))
                for line in source:
                    if line.endswith('\n'):
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    pointsTable = line.split(";")
                    sumOfObjects += len(pointsTable)
                    for i in range(len(pointsTable)):
                        x, y = pointsTable[i].split(",")
                        values.append(x)
                        values.append(y)
            if sumOfObjects > maxLength:
                maxLength = sumOfObjects
                nameOfTheBiggest = filename
            if fileCounter%10 == 0:
                print(fileCounter, " files already")
        valuesFromOneFile.append(values)

# print("Najwięcej par wspolrzednych w pliku: ", nameOfTheBiggest, ". Ilość: ", maxLength)
# print("Ilość par wspolrzednych: ", len(valuesFromOneFile[1]))
# print("Ilosc odczytanych plików: ", fileCounter)
# # print(valuesFromOneFile[0])
# # print(valuesFromOneFile[1])
# print(valuesFromOneFile)

X = np.zeros((fileCounter, maxLength, 2), dtype=int)

i = 0
j = 0
k = 0

for oneList in valuesFromOneFile:
    if(i==fileCounter+1):
        break
    for value in oneList:
        if(i==fileCounter+1):
            break
        X[i][j][k] = value
        k += 1
        if(k == 2):
            k = 0
            j += 1
    i += 1
    j = 0
    k = 0

Y = StringToInteger(labels)
Y = np.asarray(Y)

np.save("matrices_2.npy", X)
np.save("labels_2.npy", Y)

# with open('X.txt','w') as f:
#     for tablica in X:
#         f.write(str(tablica))
#         f.write("==========================")
#         f.write("\n")
#     f.close()
# print(X)