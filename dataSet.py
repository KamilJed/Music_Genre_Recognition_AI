import crop
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
        dataSet.append((crop.crop(config.spectrogramsDir + genre + '\\' + file), genre))

    return dataSet
