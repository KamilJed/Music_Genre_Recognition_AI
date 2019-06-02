from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import tensorflow as tf
from crop import crop
import numpy
import spectrogramsCreator as sC
import os

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, activation='elu', input_shape=(128, 128, 1)))
model.add(Conv2D(filters=64, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=128, kernel_size=2, activation='elu'))
model.add(Conv2D(filters=128, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=256, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Conv2D(filters=512, kernel_size=2, activation='elu'))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(1024, activation='elu'))
#model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("model")

test_Spect = ["IStayinAlive.png",
              "SoWhat.png",
              "Echoes.png",
              "CzteryPoryRoku.png",
              "IShottheSheriff.png",
              "SweetChildOMine.png",
              "TheGreatGigInTheSky.png",
              "BoogieWonderland.png",
              "EnterSandman.png",
              "IWalkTheLine.png",
              "LoseYourself.png",
              "Roar.png",
              "RoundMidnight.png",
              "TheThrillIsGone.png",
              "AHardDay\'sNight.png"]

genres = ["Disco",
        "Jazz",
        "Rock",
        "Classical",
        "Reggae",
        "Rock",
        "Rock",
        "Disco",
        "Metal",
        "Country",
        "Hip-Hop",
        "Pop",
        "Jazz",
        "Blues",
        "Rock"]

k = 0
for spect in test_Spect:


    data = crop("spectrograms\\test\\" + spect)

    testX = numpy.asarray(data)
    testX = testX.reshape([-1, 128, 128, 1])

    predictions = model.predict(testX)
    score = []
    for i in range(0,10):
        score.append(0)

    for pred in predictions:
        j = numpy.argmax(pred)
        score[j] += 1

    print(genres[k] + ": " + spect.split('.')[0] + " " )
    print(score)
    k += 1


