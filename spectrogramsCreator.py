import config
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import gc


def createSpecrograms():
    print("Generating spectrograms started")
    curpath = os.getcwd()
    curpath += '\\' + config.dataDir
    genres = os.listdir(curpath)

    for genre in genres:
        path = curpath + genre
        print("Current genre: " + genre)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                if os.path.isfile(path + '\\' + file):
                    _createSpectrogram(path + '\\' + file, genre, "train")


def getAvalaibleSpectrograms():
    path = config.spectrogramsDir
    genres = os.listdir(path)
    avalaibleSpectrograms = []

    for genre in genres:
        genrePath = path + '\\' + genre
        if os.path.isdir(genrePath):
            files = map(os.path.basename, os.listdir(genrePath))
            avalaibleSpectrograms += list(files)

    return avalaibleSpectrograms


def _createSpectrogram(path, genre, mode):
    print("Generating spectrogram: " + os.path.splitext(os.path.basename(path))[0])
    x, sr = librosa.load(path, mono=True)
    spectr = librosa.stft(x)
    spectrDb = librosa.amplitude_to_db(abs(spectr))
    fig = plt.figure(figsize=(0.65*librosa.get_duration(x), 1.67))
    ax = plt.axes()
    ax.set_axis_off()
    librosa.display.specshow(spectrDb, sr=sr, cmap='gray')
    if mode == "train":
        path = config.spectrogramsDir + genre + '\\' + os.path.splitext(os.path.basename(path))[0] + ".png"
    else:
        path = config.spectrogramsDir + "test" + '\\' + os.path.splitext(os.path.basename(path))[0] + ".png"
    fig.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0.0)
    fig.clf()
    plt.close()
    gc.collect()

