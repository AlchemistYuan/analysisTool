'''
A collection of functions to calculate histograms
'''

import numpy as np


def histogram_1d(data, nbins=50, density=True):
    hist, xedges = np.histogram(data, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    return hist.squeeze(), x.squeeze()

def histogram_2d(data_x, data_y, nbins=50, density=True):
    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    return hist, x, y

