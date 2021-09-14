from typing import Tuple, Union

import pyemma
import numpy as np


def kmeans_pyemma(feat: str, dcd: list, k: int=500, stride: int=1):
    '''
    This function performs kmeans clustering using pyemma package.

    Parameters
    ----------
    feat : str
        The pyemma featurizer.
    dcd : list
        The list of the MD trajectory files
    k : int
        The number of the clusters
    stride : int
        The step interval to read the MD snapshots
    
    Returns
    -------
    dtrajs : list of ndarray (T_i, d)
        The assignment of the each frame to each cluster.
    '''
    data = pyemma.coordinates.load(dcd, features=feat)
    cluster_kmeans = pyemma.coordinates.cluster_kmeans(data, k=k, stride=stride)
    dtrajs = cluster_kmeans.dtrajs
    maxlength = 0
    for i in range(len(dtrajs)):
        dtraj = dtrajs[i]
        if maxlength < len(dtraj):
            maxlength = len(dtraj)
    dtrajs_padded = np.ones((len(dtrajs), maxlength))
    dtrajs_padded = dtrajs_padded * -1
    for i in range(dtrajs_padded.shape[0]):
        dtraj = dtrajs[i]
        dtrajs_padded[i,:len(dtrajs)] = dtraj
    return dtrajs_padded

def tica_pyemma(files: list, top: str, lag: int=2, dim: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function performs time-lagged independent component analysis (TICA).

    Parameters
    ----------
    files : list
        The list of the trajectory files
    top : str
        The name of the topology file
    lag : int
        the lag time, in multiples of the input time step.
        default value is 2.
    dim : int
        the number of dimensions (independent components) to project onto.
        default value is 1.
    
    Returns
    -------
    output : np.ndarray
        The projection of the trajectories onto the tica components
    eigenvectors : np.ndarray
        The eigenvectors of TICA
    eigenvalues : np.ndarray
        The eigenvalues of TICA
    '''
    reader = pyemma.coordinates.source(files, top=top) # create reader
    runner = pyemma.coordinates.tica(reader, lag=lag, dim=dim)
    output = runner.get_output()
    eigvecs = runner.eigenvectors
    eigvals = runner.eigenvalues
    return (output, eigvecs, eigvals)


#def msm_pyemma(dtrajs: Union[list, np.ndarray], lags: list, nits: int, errors: str='bayes'):
#    its = pyemma.msm.its(dtrajs, lags=lags, nits=nits, errors=errors)
    
