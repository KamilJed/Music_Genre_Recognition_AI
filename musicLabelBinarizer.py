import numpy

def fit_trasform(labels, genres):
    bin_labels = []

    for label in labels:
        bin_labels.append([0] * len(genres))
        for i in range(0, len(genres)):
            if label == genres[i]:
                bin_labels[-1][i] = 1
                break

    return numpy.asarray(bin_labels)
