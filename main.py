import spectrogramsCreator as sC
from dataSet import createTrainingSet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from musicLabelBinarizer import fit_trasform
from musicFileInfo import getAllGenres
import numpy

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, activation='elu', input_shape=(128, 128, 1)))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=128, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=256, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=512, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.get_weights())
spectrogramsAvalaible = sC.getAvalaibleSpectrograms()
dataSetTrain = createTrainingSet(spectrogramsAvalaible, 200)
dataSetVal = createTrainingSet(spectrogramsAvalaible, 200)
dataSetTrainX, dataSetTrainY = zip(*dataSetTrain)
dataSetValX, dataSetValY = zip(*dataSetVal)


trainX = numpy.asarray(dataSetTrainX)
trainX = trainX.reshape([-1, 128, 128, 1])
trainY = fit_trasform(dataSetTrainY, getAllGenres())

validX = numpy.asarray(dataSetValX)
validX = validX.reshape([-1, 128, 128, 1])
validY = fit_trasform(dataSetValY, getAllGenres())

model.fit(trainX, trainY, validation_data=(validX, validY), epochs=1, batch_size=128, shuffle=True)
print(model.get_weights())