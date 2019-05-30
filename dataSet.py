from crop import crop
import config
import musicFileInfo
import random

def createTrainingSet(fileList, trainingSetSize):
    random.shuffle(fileList)
    data = fileList[:trainingSetSize]
    fileList[:] = fileList[trainingSetSize:]

    dataSet = []
    for file in data:
        genre = musicFileInfo.getGenre(file)
        slices = crop(config.spectrogramsDir + genre + '\\' + file)
        for slice in slices:
            dataSet.append((slice, genre))

    random.shuffle(dataSet)
    return dataSet
