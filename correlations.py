import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')
from typing import Tuple, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score 

from utils import *


'''
This is the collection of the functions that calculate different types of correlation coefficients.

1. Pearson correlation coefficient
2. Spearman correlation coefficient
3. Mutual information for a continuous variable
'''


def calc_correlation(universes: list, atoms: str='protein and name CA', start: int=0, stop: int=-1, stride: int=1, method: str='pearson', bins: int=100) -> np.ndarray:
    '''
    This function calculates the Pearson correlation coefficients between Cartesian coordinates of atoms in MD trajectories.

    Parameters
    ----------
    universes : list
        The list of MDAnalysis universe.
    atoms : str
        The atom selection string.
    start : int
        The first frame to read.
    stop : int
        The last frame to read.
    stride : int
        The interval to read frames. 
    method : str
        The method of correlation coefficient
    bins : int
        The number of bins. Required if method == 'mutual'

    Returns
    -------
    corr_mat : np.ndarray
        The correlation coefficients matrix.
    '''
    coors = []
    for u in universes:
        coordinates = u.trajectory.timeseries(u.select_atoms(atoms), start=start, stop=stop, step=stride, order='fac')
        #x = coordinates.ravel().reshape(coordinates.shape[0],3*coordinates.shape[1])
        coors.append(coordinates)
    x = np.concatenate(coors)
    print(x.shape)
    average_coor = np.average(x, axis=0)
    x -= average_coor

    if method == 'pearson':
        func = _covariance
    elif method == 'spearman':
        func = _spearman_rankcorr
    elif method == 'mutual':
        func = _mutual_information
    else:
        print('undefiend method') 
    corr_mat = func(x, bins)
    return corr_mat


def _covariance(x: np.ndarray, extra: int) -> np.ndarray:
    '''
    This function calculates the Pearson correlation coefficients between Cartesian coordinates of atoms in MD trajectories.

    Parameters
    ----------
    x : np.ndarray
        the coordinates set
    extra : int
        a redundant argument for a consistent api interface

    Returns
    -------
    corr_mat : np.ndarray
        The correlation coefficients matrix.
    '''
    corr_mat = np.zeros((x.shape[1],x.shape[1]))
    for i, frame in enumerate(x):
        corr_mat += np.corrcoef(frame)
    corr_mat /= x.shape[0]
    return corr_mat


def _spearman_rankcorr(x: np.ndarray, extra: int) -> np.ndarray:
    '''
    This function calculates the Spearson rank-order correlation coefficients between Cartesian coordinates of atoms in MD trajectories.

    Parameters
    ----------
    x : np.ndarray
        the coordinates set
    extra : int
        a redundant argument for a consistent api interface

    Returns
    -------
    corr_mat : np.ndarray
        The correlation coefficients matrix.
    '''
    corr_mat = np.zeros((x.shape[1],x.shape[1]))
    for i, frame in enumerate(x):
        #If axis=0 (default), then each column represents a variable, 
        #with observations in the rows. If axis=1, the relationship is transposed: 
        #each row represents a variable, while the columns contain observations. 
        #If axis=None, then both arrays will be raveled.
        correlation, _ = spearmanr(frame, axis=1)
        corr_mat += correlation
    corr_mat /= x.shape[0]
    return corr_mat


def _mutual_information(x: np.ndarray, bins: int=100) -> np.ndarray:
    '''
    This function calculates the Spearson rank-order correlation coefficients between Cartesian coordinates of atoms in MD trajectories.

    Parameters
    ----------
    x : np.ndarray
        the coordinates set
    bins : int
        The number of bins to calculate 2d histogram

    Returns
    -------
    corr_mat : np.ndarray
        The correlation coefficients matrix.
    '''
    corr_mat = np.zeros((x.shape[1],x.shape[1]))
    for i in range(x.shape[1]):
        data_x = x[:,i,:].flatten().squeeze()
        for j in range(x.shape[1]):
            data_y = x[:,j,:].flatten().squeeze()
            c_xy = np.histogram2d(data_x, data_y, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            corr_mat[i,j] += mi
    corr_mat /= corr_mat.max()
    return corr_mat
