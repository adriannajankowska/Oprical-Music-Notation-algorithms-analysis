from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import backend as K 
import matplotlib.pyplot as plt
import numpy as np
import time, logging

logging.basicConfig(filename='log_testYourOwnNeuralNetwork.log', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.warning(" .................. testYourOwnNeuralNetwork.py .................. ")
start = time.time()
Y = np.load("labels_2.npy")
X = np.load("matrices_2.npy", allow_pickle=True)

# Y = np.expand_dims(Y, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=True)
withoutDuplicates = list(dict.fromkeys(Y))

#print("Shape of X[0]: ", X[0].shape)

model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=X[0].shape),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(withoutDuplicates), activation='softmax')    #normalization
])


print("Before: ", X.shape)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print("After: ", X.shape)
a, b, c, d = X.shape
input_shape = (b, c, d)

logging.warning("Compiling the model")
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',    #inne to np. mean sqaure error...
              metrics=['accuracy']) 

logging.warning("Fitting the model")
history = model2.fit(X_train, Y_train, validation_split=0.10, epochs=200, batch_size=10, verbose=1)
print(history.history.keys())
predictions = model2.predict(X_test)
print(predictions)
listOfPredictions = []
i = -1

for pred in predictions:
    index = np.argmax(pred)
    listOfPredictions.append(index)


labels = ['12-8-Time','2-4-Time','Quarter-Note','Quarter-Rest','Sharp','Sixteenth-Note','Sixteenth-Rest','Sixty-Four-Rest','3-4-Time','Thirty-Two-Note','Thirty-Two-Rest','Whole-Half-Rest','Whole-Note','3-8-Time','4-4-Time','6-8-Time','9-8-Time','Barline','C-Clef','Common-Time','Cut-Time','Dot','2-2-Time','Double-Sharp','Eighth-Note','Eighth-Rest','F-Clef','Flat','G-Clef','Half-Note','Natural']
cm = confusion_matrix(Y_test, listOfPredictions)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=plt.cm.get_cmap("summer"))
fig.colorbar(cax)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xticks(np.arange(len(labels)), rotation='vertical')
plt.yticks(np.arange(len(labels)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.clf()



logging.warning("Plotting the history")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training set accuracy', 'validation set accuracy'], loc='upper left')
saveTime = time.time()
plt.savefig(fname = "Model_Accuracy_"+str(saveTime)+".png")
plt.show()
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training set loss', 'validation set loss'], loc='upper left')
saveTime = time.time()
plt.savefig(fname = "Model_Loss_"+str(saveTime)+".png")
plt.show()
end = time.time()
logging.warning("Elapsed time: %f seconds", end - start)