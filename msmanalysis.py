from typing import Tuple, Union

import pyemma
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def kmeans_scikit(data: np.array, k: int=250, batch_size: int=0) -> Tuple[np.ndarray, np.ndarray]:
    '''
    This function performs kmeans clustering using sci-kit learn. 

    Parameters
    ----------
    data : list
        The array of input features of shape (nframe, nfeatures)
    k : int
        The number of the clusters
    batch_size : int
        Batch size for MiniBatchKmeans. Default is 0, which means not to use MiniBatchKMeans.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels : ndarray of shape (n_samples,)
        The assignment of the each frame to each cluster.
    cluster_center_indices : ndarray of shape (n_clusters,)
        The frame indices cloesest to each cluster
    '''
    if batch_size > 0:
        kmeans = MiniBatchKMeans(init="k-means++", n_clusters=k, batch_size=batch_size)
    else:
        kmeans = KMeans(init="k-means++", n_clusters=k)
    distances = kmeans.fit_transform(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels = np.asarray(labels, dtype=int)
    cluster_center_indices = np.argmin(distances, axis=0)
    cluster_center_indices = np.asarray(cluster_center_indices, dtype=int)
    return (centers, labels, cluster_center_indices)


def tica_pyemma(universes: list, lag: int=2, dim: int=1, atoms: str='name CA', transform: Union[None, list]=None) -> Tuple[np.ndarray, Union[np.ndarray,None], np.ndarray, np.ndarray]:
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
    transform : list or None
        The list of dcd files to be transformed. The default is None.

    Returns
    -------
    output : np.ndarray
        The projection of the trajectories onto the tica components
    transformed : np.ndarray or None
        The projection of other trajectories onto the already-computed tica components
    eigenvectors : np.ndarray
        The eigenvectors of TICA
    eigenvalues : np.ndarray
        The eigenvalues of TICA
    timescales : np.ndarray
        The timescales of TICA
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
    if type(transform) != 'NoneType':
        coors_transform = []
        for i, u in enumerate(transform):
            coors = u.trajectory.timeseries(u.select_atoms(atoms), order='fac')
            if i in [0, 2]:
                coors = coors[::24,:,:]
            coors_transform.append(coors)
        coors_transform = np.concatenate(coors_transform)
        x = coors_transform.ravel().reshape(coors_transform.shape[0],3*coors_transform.shape[1])
        x = x.astype('float32')
        transformed = runner.transform(x)
    else:
        transformed = None
    return (output, transformed, eigvecs, eigvals, timescales)
