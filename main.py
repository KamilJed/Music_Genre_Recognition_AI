import spectrogramsCreator as sC
from dataSet import createTrainingSet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import LabelBinarizer
import numpy

# TODO Whole main
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, activation='elu', input_shape=(128, 128, 1), data_format='channels_last'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=128, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=256, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=512, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

spectrogramsAvalaible = sC.getAvalaibleSpectrograms()
dataSetTrain = createTrainingSet(spectrogramsAvalaible, 700)
dataSetVal = createTrainingSet(spectrogramsAvalaible, 200)
dataSetTrainX, dataSetTrainY = zip(*dataSetTrain)
dataSetValX, dataSetValY = zip(*dataSetVal)

encoder = LabelBinarizer()

trainX = numpy.asarray(dataSetTrainX)
trainX = trainX.reshape(trainX.shape[0], 128, 128, 1)
trainY = numpy.asarray(dataSetTrainY)
trainY = encoder.fit_transform(trainY)


validX = numpy.asarray(dataSetValX)
validX = validX.reshape(validX.shape[0], 128, 128, 1)
validY = numpy.asarray(dataSetValY)
validY = encoder.fit_transform(validY)

model.fit(trainX, trainY, validation_data=(validX, validY), epochs=3)