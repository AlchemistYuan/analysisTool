from typing import Tuple, Union

import numpy as np
from scipy.spatial.distance import jensenshannon
import mdtraj as md

'''
This is the collection of the functions that can be used to transform or manipulate data.
'''

def compute_jensen_shannon_divergence(dataA: np.array, dataB: np.array, axis: int=0, nbins: int=50) -> Tuple[float, np.array]:
    '''
    This function computes the jensen shannon divergence.

    Parameters
    ----------
    dataA: np.array
        The data sampled from one probability distribution.
    dataB: np.array
        The data sampled from another probability distribution.
    nbins: int
        The number of bins in making histograms
    axis: int
        Axis along which the Jensen-Shannon distances are computed.
    Returns
    -------
    js: float or np.array
        The Jensen-Shannon distances between the two distributions.
    '''
    xmin = min(dataA.min(), dataB.min())
    xmax = min(dataA.max(), dataB.max())
    histA, xedges = np.histogram(dataA, range=(xmin, xmax), bins=nbins, density=True)
    histB, xedges = np.histogram(dataB, range=(xmin, xmax), bins=nbins, density=True)
    js_distance = jensenshannon(histA, histB, base=2, axis=axis)
    js = js_distance ** 2
    return js ** 2

def switching_function(r: np.array, r0: float=10.0, n: int=6, m: int=10) -> np.array:
    ratio = r / r0
    s = (1 - ratio ** n) / (1 - ratio ** m)
    return s

def convert_to_square_form(distances: np.ndarray, residue_pairs: np.ndarray, nres: int=404, cutoff: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    nframe = len(distances)
    contact_maps = np.zeros((distances.shape[0], nres, nres), dtype=distances.dtype)
    contact_maps[:, residue_pairs[:, 0], residue_pairs[:, 1]] = distances
    contact_maps[:, residue_pairs[:, 1], residue_pairs[:, 0]] = distances
    contact_maps_avg = np.mean(contact_maps, axis=0)
    nframe = contact_maps.shape[0]
    contact_probability = np.zeros_like(contact_maps_avg)
    for i in range(nframe):
        contact = contact_maps[i,:,:].squeeze()
        # In mdtraj, the distance is in nm by default
        contact_cut = np.where(contact < cutoff, 1, 0)
        contact_probability += contact_cut
    contact_probability = contact_probability / nframe
    # Set the diagonal and first and second diagonal elements to zero
    np.fill_diagonal(contact_probability, 0)
    # We only need the triagonal elements due to the symmetry.
    contact_probability = np.tril(contact_probability)
    contact_first_offdiag = np.diag(np.diag(contact_probability, -1), -1)
    contact_probability -= contact_first_offdiag
    contact_second_offdiag = np.diag(np.diag(contact_probability, -2), -2)
    contact_probability -= contact_second_offdiag
    return contact_probability, contact_maps_avg


def convert_to_nonsquare_form(distances: np.ndarray, convert: list, cutoff: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    contact_maps = np.zeros((distances.shape[0], convert[0], convert[1]), dtype=distances.dtype)
    contact_maps[:,:,:] = distances.reshape((distances.shape[0], convert[0], convert[1]))
    contact_maps_avg = np.mean(contact_maps, axis=0)
    nframe = contact_maps.shape[0]
    contact_probability = np.zeros_like(contact_maps_avg)
    for i in range(nframe):
        contact = contact_maps[i,:,:]
        # In mdtraj, the distance is in nm by default
        contact_cut = np.where(contact < cutoff, 1, 0)
        contact_probability += contact_cut
    contact_probability = contact_probability / nframe
    return contact_probability, contact_maps_avg


def find_resid(res: int) -> str:
    """
    Format the residue ID in the form of "100A", where the number is the residue ID
    and the letter represents the monomer ID.

    Parameters
    ----------
    res : int
        The residue ID

    Returns
    -------
    index : str
        The formatted residue ID
    """
    if res > 201:
        index = str(res - 200) + 'B'
    else:
        index = str(res + 2) + 'A'
    return index


def find_secstr(res: int, secstr: dict) -> str:
    if res > 201:
        index = res - 200
    else:
        index = res + 2
    for key in secstr.keys():
        if index in secstr[key]:
            return key

def process_dcdfiles(dcdlist: list, dcdfilelist: str) -> list:
    """
    Process the dcd files from either a list or a file with path to the dcd files.

    Parameters
    ----------
    dcdlist : list
        A list with all dcd files
    dcdfilelist : str
        A file with the path to the dcd files

    Returns
    -------
    dcds : list
        A list with all dcd files
    """
    dcds = []
    if not isinstance(dcdlist, type(None)):
        dcds = dcdlist
    elif not isinstance(dcdfilelist, type(None)):
        with open(dcdfilelist, 'r') as f:
            line = f.readline()
            while line:
                l = line.strip()
                dcds.append(l)
                line = f.readline()
    return dcds
