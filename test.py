from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.optimizers import Adam
from crop import crop
import numpy
import os
from musicFileInfo import getAllGenres
from spectrogramsCreator import _createSpectrogram
from config import spectrogramsDir

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

model.load_weights("best_model_ever_adam0")

#files = os.listdir("songs\\")
#for file in files:
#    _createSpectrogram("songs\\" + file, mode="test")

test_Spect= os.listdir(spectrogramsDir + "test\\")

genres = getAllGenres()

for spect in test_Spect:

    data = crop(spectrogramsDir + "test\\" + spect)

    testX = numpy.asarray(data)
    testX = testX.reshape([-1, 128, 128, 1])

    predictions = model.predict_classes(testX)

    classses = [0] * 10
    for c in predictions:
        classses[c] += 1

    if max(classses) > 0.35*len(predictions):
        print(os.path.splitext(os.path.basename(spect))[0] + ' is ' + genres[classses.index(max(classses))])
    else:
        print("I'm not sure what genre is " + os.path.splitext(os.path.basename(spect))[0])