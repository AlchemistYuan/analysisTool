import pyemma
import numpy as np


def tica_pyemma(coor, lag=2):
    '''
    This function performs time-lagged independent component analysis (TICA).

    Parameters
    ----------
    coor : np.ndarray
        ndarray of shape (T, d) or list of ndarray of shape (T_i, d).
    lag : int
        the lag time, in multiples of the input time step.
        default value is 2.

    Returns
    -------
    
    '''
    tica_runner = pyemma.coordinates.tica(coor, lag=lag)
    tica_output = tica_runner.get_output()
    tica_eigvecs = tica_runner.eigenvectors
    tica_eigvals = tica_runner.eigenvalues
    return (tica_output, tica_eigvecs, tica_eigvals)
