import os
import config

def getGenre(fileName):
    return fileName.split('.')[0]

def getAllGenres():
    genres = []
    curpath = os.getcwd()
    curpath += '\\' + config.dataDir
    genres = os.listdir(curpath)
    return genres