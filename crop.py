from PIL import Image

def crop( inputPath):
    im = Image.open(inputPath).convert('L')
    pixels = list(im.getdata())
    imgwidth, imgheight = im.size
    #newwidth = imgwidth/10
    #k = 0
    imagesList = []
    image = []

    for i in range(0,imgwidth):
        if(i != 0 and i%128 == 0):
            imagesList.append(image)
            image = []
        col = []

        for j in range(i,imgwidth*imgheight - imgwidth + i + 1 ,imgwidth):
            col.append(pixels[j])

        image.append(col)
    if(imgwidth%128 == 0):
        imagesList.append(image)
    return imagesList
