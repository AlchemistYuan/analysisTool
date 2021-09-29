from typing import Tuple, Union

import pyemma
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def kmeans_scikit(data: np.array, k: int=250):
    '''
    This function performs kmeans clustering using sci-kit learn. 

    Parameters
    ----------
    data : list
        The array of input features of shape (nframe, nfeatures)
    k : int
        The number of the clusters
    
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels : ndarray of shape (n_samples,)
        The assignment of the each frame to each cluster.
    '''
    kmeans = MiniBatchKMeans(init="k-means++", n_clusters=k, batch_size=4096)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return (centers, labels)


def tica_pyemma(universes: list, lag: int=2, dim: int=1, atoms: str='name CA') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function performs time-lagged independent component analysis (TICA).

    Parameters
    ----------
    universes : list
        The list of the trajectory
    lag : int
        the lag time, in multiples of the input time step.
        default value is 2.
    dim : int
        the number of dimensions (independent components) to project onto.
        default value is 1.
    atoms : str
        The atoms used in TICA.

    Returns
    -------
    output : np.ndarray
        The projection of the trajectories onto the tica components
    eigenvectors : np.ndarray
        The eigenvectors of TICA
    eigenvalues : np.ndarray
        The eigenvalues of TICA
    '''
    print('starting TICA...')
    ntraj = []
    natoms = len(universes[0].select_atoms(atoms))
    for u in universes:
        ntraj.append(len(u.trajectory))
    ntraj = np.asarray(ntraj, dtype=int)
    coor = np.zeros((np.sum(ntraj), natoms, 3), dtype=np.float32)
    start = 0
    end = 0
    print('Reading coordinates...')
    for i, u in enumerate(universes):
        end += ntraj[i]
        coor[start:end,:,:] = u.trajectory.timeseries(u.select_atoms(atoms), order='fac')
        start += ntraj[i]

    offset = coor - np.mean(coor, axis=0)
    x = offset.ravel().reshape(offset.shape[0],3*offset.shape[1])
    x = x.astype('float32')

    runner = pyemma.coordinates.tica(x, lag=lag, dim=dim)
    output = runner.get_output()
    output = np.concatenate(output)
    print(output.shape)
    eigvecs = runner.eigenvectors
    eigvals = runner.eigenvalues
    timescales = runner.timescales
    return (output, eigvecs, eigvals, timescales)


#def msm_pyemma(dtrajs: Union[list, np.ndarray], lags: list, nits: int, errors: str='bayes'):
#    its = pyemma.msm.its(dtrajs, lags=lags, nits=nits, errors=errors)
    
