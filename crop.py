from PIL import Image
import numpy

def crop( inputPath):
    img = Image.open(inputPath).convert("L")
    desiredSize = 128

    width, height = img.size
    nbSamples = int(width / desiredSize)

    imagesList = []
    for i in range(nbSamples):
        startPixel = i * desiredSize
        imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, desiredSize + 1))
        imgTmp = imgTmp.resize((128, 128), resample=Image.ANTIALIAS)
        data = numpy.asarray(imgTmp).reshape((128, 128, 1))
        data = data / 255.
        imagesList.append(data)

    return imagesList


