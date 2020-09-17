import os
import pygame
import numpy as np
from pygame.locals import*
from PIL import Image
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import logging
#============================================================================
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
#============================================================================
def SaveTxtToPng(filepath, labels):
    pointsTable = []     
    screen = pygame.display.set_mode((200, 300))
    screen.fill((0, 0, 0))
    
    with open(filepath, 'r') as source:
        labels.append(source.readline().replace('\n', ""))
        for line in source:
            line = line[:-2]
            pointsTable = line.split(";")
            for i in range(0, len(pointsTable)-2):
                x1, y1 = pointsTable[i].split(",")
                x2, y2 = pointsTable[i+1].split(",")
                pygame.draw.line(screen, (255, 255, 255), (int(x1), int(y1)), (int(x2), int(y2)))            
    pygame.image.save(screen, subdir + os.sep + filename+".png")
#============================================================================
def PngToMatrix(subdir, filename):
    source = Image.open(subdir + os.sep + filename+".png").convert('L')
    matrix = np.array(source)

    for row in range(len(matrix)): 
        for col in range(len(matrix[0])):
            if not matrix[row][col]==0:
                matrix[row][col] = 1
    return matrix
#============================================================================                      
def FindBoundaries(matrix):
    left = len(matrix)
    right = 0
    top = len(matrix[0])
    bottom = 0

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 1:
                if col < left:
                    left = col
                if col > right:
                    right = col
                if row < top:
                    top = row
                if row > bottom:
                    bottom = row
    return left, right, top, bottom
#============================================================================
def WriteMatrixToFile(filename, matrix):
    with open(filename+'.txt','w') as f:
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                f.write(str(matrix[row][col]))
            f.write("\n")
    f.close()
#============================================================================

datasetPath = os.getcwd() + "\\HOMUS"

notes = []              #list matrices of notes
labels = []              #list of labels
maxWidthCropped = 0
maxHeightCropped = 0

counter = 0 

logging.basicConfig(filename='log_mainScript.log', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

for subdir, dirs, files in os.walk(datasetPath):
    for txt in files:
        filepath = subdir + os.sep + txt

        if filepath.endswith(".txt"):
            counter += 1
            filename = txt.replace(".txt", "")

            SaveTxtToPng(filepath, labels)
            matrix = PngToMatrix(subdir, filename)

            left, right, top, bottom = FindBoundaries(matrix)
            croppedMatrix = matrix[top:bottom+1, left:right+1]

            if len(croppedMatrix) > maxHeightCropped:
                maxHeightCropped = len(croppedMatrix)
            if len(croppedMatrix[0]) > maxWidthCropped:
                maxWidthCropped = len(croppedMatrix[0])

            notes.append(croppedMatrix)
            if counter%10 == 0:
                print(counter, " files already")


# uniqueLabels = list(dict.fromkeys(labels))
# with open('labelsss.txt', 'w') as f:
#     for item in uniqueLabels:
#         f.write("%s\n" % item)


print("Maksymalna wysokość: ", maxHeightCropped)
print("Maksymalna szerokość: ", maxWidthCropped)

Y = StringToInteger(labels)
Y = np.asarray(Y)

#wykres częstości elementów w datasecie
df = pd.DataFrame({'freq': labels})
df.groupby('freq', as_index=False).size().plot(kind='bar', colormap='ocean')
#plt.show()

X = np.zeros((len(notes), maxHeightCropped, maxWidthCropped), dtype=int)

for i in range(len(notes)):
    for j in range(len(notes[i])):
        for k in range(len(notes[i][j])):
            X[i][j][k] = notes[i][j][k]
    #WriteMatrixToFile("note_" + str(i+1), X[i])

np.save("matrices.npy", X)
np.save("labels.npy", Y)