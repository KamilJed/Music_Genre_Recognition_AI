from tensorflow.python.keras.callbacks import ModelCheckpoint
import spectrogramsCreator as sC
from dataSet import createTrainingSet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.optimizers import rmsprop, Adam
from musicLabelBinarizer import fit_trasform
from musicFileInfo import getAllGenres
from tensorflow.python.keras.callbacks import TensorBoard
from config import dataForTestingPercent, dataForTrainingPercent, dataForValidatingPercent
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

model.compile(optimizer=Adam(epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])

spectrogramsAvalaible = sC.getAvalaibleSpectrograms()
dataSet = createTrainingSet(spectrogramsAvalaible, len(spectrogramsAvalaible))
dataSetX, dataSetY = zip(*dataSet)

dataSetTrainX = dataSetX[:int(dataForTrainingPercent*len(dataSetX))]
dataSetTrainY = dataSetY[:int(dataForTrainingPercent*len(dataSetY))]

dataSetValX = dataSetX[int(dataForTrainingPercent*len(dataSetX)):int((dataForTrainingPercent + dataForValidatingPercent)*len(dataSetX))]
dataSetValY = dataSetY[int(dataForTrainingPercent*len(dataSetY)):int((dataForTrainingPercent + dataForValidatingPercent)*len(dataSetY))]

dataSetTestX = dataSetX[int((dataForTrainingPercent + dataForValidatingPercent)*len(dataSetX)):]
dataSetTestY = dataSetY[int((dataForTrainingPercent + dataForValidatingPercent)*len(dataSetY)):]


trainX = numpy.asarray(dataSetTrainX)
trainX = trainX.reshape([-1, 128, 128, 1])
trainY = fit_trasform(dataSetTrainY, getAllGenres())

validX = numpy.asarray(dataSetValX)
validX = validX.reshape([-1, 128, 128, 1])
validY = fit_trasform(dataSetValY, getAllGenres())

testX = numpy.asarray(dataSetTestX)
testX = testX.reshape([-1, 128, 128, 1])
testY = fit_trasform(dataSetTestY, getAllGenres())

checkpoint = ModelCheckpoint("best_model_ever", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
callbacks_list = [checkpoint, tensorboard]

model.fit(trainX, trainY, validation_data=(validX, validY), epochs=20, batch_size=128, shuffle=True,
          callbacks=callbacks_list)

print("model succesfully trained")
model.save("model")

testAccu = model.evaluate(testX, testY)[0]
print(testAccu)