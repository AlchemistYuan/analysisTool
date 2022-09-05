import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')
from typing import Tuple, Union

import MDAnalysis as mda
import mdtraj as md
import numpy as np
import pyemma
from MDAnalysis.analysis import align, rms
from sklearn.decomposition import PCA

from utils import *


'''
This is the collection of the functions that can be used in post-processing.
'''

def align_traj(psfs: list, dcds: list, ref: mda.Universe, atoms: str) -> list:
    '''
    Align the trajectory against the reference structure.

    Parameters
    ----------
    psfs : list
        A list of the psf files.
    dcds : list
        A list of the dcd files
    ref : MDAnalysis.Universe
        An MDAnalysis.Universe object for the reference structure
    atoms : str
        The atom selection string.

    Returns
    -------
    universes : list
        A list of the aligned trajectory
    '''
    print('start align trajectories...')
    universes = []
    for i, psf in enumerate(psfs):
        dcd = dcds[i]
        u = mda.Universe(psf, dcd)
        align.AlignTraj(u, ref, select=atoms, in_memory=True).run()
        universes.append(u)
        print('one trajectory completed...')
    return universes

def pca_pyemma(universes: list, atoms: str, dim: int=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.array]:
    '''
    Perform PCA on a list of Universe.

    Parameters
    ----------
    universes : list
        A list of Universe to be used in PCA.
    atoms : str
        An atom selection string to be used in PCA.
    dim : int
        The dimension of the PCA projection

    Returns
    -------
    output : np.ndarray
        A np ndarray of shape (n_samples, n_components) to store the projection along certain PC axis.
    eigvecs : np.ndarray
        Principal axes in feature space of shape (n_components, n_features), representing the directions of maximum variance in the data.
    eigvals : np.ndarray
        The amount of variance explained by each of the selected components.
    mean_coor : np.ndarray
        The average coordinates to be subtracted in PCA.
    ntraj : np.array
        The length of each trajectory.
    '''
    print('starting pyemma PCA...')
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

    runner = pyemma.coordinates.pca(x, dim=dim)
    output = runner.get_output()
    output = np.concatenate(output)
    eigvecs = runner.eigenvectors
    eigvals = runner.eigenvalues
    mean_coor = runner.mean
    mean_coor = np.reshape(mean_coor, (-1,3))
    return output, eigvecs.T, eigvals, mean_coor, ntraj

def pca_scikit(universes: list, atoms: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.array]:
    '''
    Perform PCA on a list of Universe.

    Parameters
    ----------
    universes : list
        A list of Universe to be used in PCA.
    atoms : str
        An atom selection string to be used in PCA.

    Returns
    -------
    proj : np.ndarray
        A np ndarray of shape (n_samples, n_components) to store the projection along certain PC axis.
    components : np.ndarray
        Principal axes in feature space of shape (n_components, n_features), representing the directions of maximum variance in the data.
    variance : np.ndarray
        The amount of variance explained by each of the selected components.
    mean_coor : np.ndarray
        The average coordinates to be subtracted in PCA.
    ntraj : np.array
        The length of each trajectory.
    '''
    print('starting PCA...')
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
    print('initialize PCA analysis...')
    pca = PCA()  #(n_components=10)
    print('project coordiantes onto pc componnets...')
    proj = pca.fit_transform(x)
    print('retrieving necessary results...')
    proj = proj[:,:10]
    components = pca.components_
    variance = pca.explained_variance_
    mean_coor = np.mean(coor, axis=0)
    return proj, components, variance, mean_coor, ntraj

def projection(pcs: np.ndarray, ref_coor: np.ndarray, universe: mda.Universe, atoms: str) -> np.ndarray:
    '''
    Project a MD trajectory onto a set of pre-computed PCA axis.

    Parameters
    ----------
    pcs : np.ndarray
        A set of pre-computed PC coordinates
    ref_coor : np.ndarray
        The average structure to be subtracted from the trajectory
    universe : MDAnalysis.Universe
        The trajecotry to be used in the projection
    atoms : str
        An atom selection string to be used in PCA

    Returns
    -------
    proj : np.ndarray
        The projection of the trajectory onto the PCs.
    '''
    # Get all coordinates of each frame.
    coor = universe.trajectory.timeseries(universe.select_atoms(atoms), order='fac')
    # Center the trajectory
    offset = coor - ref_coor
    frame = offset[:,:,:]
    # Flatten the centered trajectory such that the shape becomes (nframe, natom*3)
    x = frame.ravel().reshape(frame.shape[0],3*frame.shape[1])
    # Compute the dot products
    proj = np.dot(x, pcs.T)
    if proj.shape[1] > 10:
        proj = proj[:,:10]
    return proj

def rmsd(universes: list, ref: mda.Universe, refatoms: str) -> np.ndarray:
    """
    Compute the root mean square deviation for all trajectories

    Parameters
    ----------
    universes : list
        The list of mda Universes
    ref : mda.Universe
        The reference structure
    refatoms : str
        The atoms to be aligned against

    Returns
    -------
    rmsd_all : np.ndarray
        The computed rmsd values
    """
    ntraj = count_frame(universes)
    rmsd_all = np.zeros((np.sum(ntraj), 2))
    start = 0
    end = 0
    print('Starting RMSD calculations...')
    for i, u in enumerate(universes):
        rmsder = rms.RMSD(u, ref, select=refatoms)
        rmsder.run()
        end += ntraj[i]
        rmsd_all[start:end, 0] = rmsder.rmsd[:, 2]
        rmsd_all[start:end, 1] = int(i)
        start += ntraj[i]
    return rmsd_all

def rmsf(universes: list, refatoms: str) -> np.ndarray:
    """
    Compute the root mean square fluctuation of each protein residue.

    Parameters
    ----------
    universes : list
        The list of mda Universes
    refatoms : str
        The atoms to be aligned against

    Returns
    -------
    rmsf_all : np.ndarray
        The computed rmsf values
    """
    coors_all = []
    for i, u in enumerate(universes):
        protein = u.select_atoms('protein')
        atoms = u.select_atoms(refatoms)
        coor = u.trajectory.timeseries(protein, order='fac')
        coors_all.append(coor)
    coors_all = np.concatenate(coors_all)
    protein = universes[0].select_atoms('protein')
    u_new = mda.Merge(protein).load_new(coors_all, order="fac")
    ref_avg = mda.Merge(protein).load_new(coors_all.mean(axis=0)[None, :, :], order="fac")
    align.AlignTraj(u_new, ref_avg, select=refatoms, in_memory=True).run()
    rmsfer = rms.RMSF(refatoms, verbose=True)
    rmsfer.run()
    rmsf_all = rmsfer.rmsf
    return rmsf_all

def count_frame(universes: list) -> list:
    """
    Count the number of frames for each trajectory

    Parameters
    ----------
    universes : list
        The list of mda Universes

    Returns
    -------
    ntraj : np.ndarray
        The array to store the number of frames
    """
    print('Counting number of frame...')
    ntraj = []
    for u in universes:
        ntraj.append(len(u.trajectory))
    ntraj = np.asarray(ntraj, dtype=int)
    return ntraj

def distanceDBD(psfs: list, dcds: list, stride: int=1) -> tuple[np.ndarray, list[int]]:
    """
    Compute the DNA-binding domain distance.
    This is specific for the TetR(B) so the atom selection is hard-coded.

    Parameters
    ----------
    psfs : list
        The list of the psf files
    dcds : list
        The list of the dcd files
    stride : int
        The step interval to read frames

    Returns
    -------
    distances : np.ndarray
        The DNA-binding domain distances
    ntraj : list
        The number of frames for each trajectory
    """
    universes = []
    ntraj = []
    for i in range(len(psfs)):
        u = mda.Universe(psfs[i], dcds[i])
        universes.append(u)
        ntraj.append(len(u.trajectory[::stride]))
    distances = np.zeros(np.sum(ntraj))
    start, end = 0, 0
    for j, u in enumerate(universes):
        dbda = u.select_atoms('name CA and segid PROA and resid 37:44')
        dbdb = u.select_atoms('name CA and segid PROB and resid 37:44')
        dbda_position = u.trajectory.timeseries(dbda, order='fca')[::stride, :, :]
        dbdb_position = u.trajectory.timeseries(dbdb, order='fca')[::stride, :, :]
        dbda_com = np.mean(dbda_position, axis=2)
        dbdb_com = np.mean(dbdb_position, axis=2)
        diff = dbda_com - dbdb_com
        distance = np.linalg.norm(diff, axis=1)
        end += ntraj[j]
        distances[start:end] = distance
        start += ntraj[j]
    return distances, ntraj

def angle_a4(universes: list, dna_a4: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Compute the angle of the alpha helix 4 relative to the
    Parameters
    ----------
    universes : list
        The list of mda Universes
    dna_a4 : np.ndarray
        The reference position of the alpha helix 4 in the DNA-bound state reference structure

    Returns
    -------
    angles : np.ndarray
        The alpha helix 4 angles
    ntraj : list
        The number of frames for each trajectory
    """
    ntraj = []
    for u in universes:
        ntraj.append(len(u.trajectory))
    total_frame = np.sum(ntraj)
    angles = np.zeros((total_frame, 2))
    start = 0
    end = 0
    for i, u in enumerate(universes):
        res48a = u.select_atoms('backbone and segid PROA and resid 48')
        res48b = u.select_atoms('backbone and segid PROB and resid 48')
        res63a = u.select_atoms('backbone and segid PROA and resid 63')
        res63b = u.select_atoms('backbone and segid PROB and resid 63')
        res48a_position = np.mean(u.trajectory.timeseries(res48a, order='fac'), axis=1)
        res48b_position = np.mean(u.trajectory.timeseries(res48b, order='fac'), axis=1)
        res63a_position = np.mean(u.trajectory.timeseries(res63a, order='fac'), axis=1)
        res63b_position = np.mean(u.trajectory.timeseries(res63b, order='fac'), axis=1)
        a4a_vec = res48a_position - res63a_position
        a4b_vec = res48b_position - res63b_position
        a4a_dot = np.dot(a4a_vec, dna_a4[0, :])
        a4b_dot = np.dot(a4b_vec, dna_a4[1, :])
        a4a_norm = np.linalg.norm(a4a_vec, axis=1)
        a4b_norm = np.linalg.norm(a4b_vec, axis=1)
        a4a_cos = a4a_dot / (np.linalg.norm(dna_a4, axis=1)[0] * a4a_norm)
        a4b_cos = a4b_dot / (np.linalg.norm(dna_a4, axis=1)[1] * a4b_norm)
        cos = np.concatenate((a4a_cos.reshape(-1, 1), a4b_cos.reshape(-1, 1)), axis=1)
        angle = 180 * np.arccos(cos) / np.pi
        end += ntraj[i]
        angles[start:end, :] = angle
        start += ntraj[i]
    return angles, ntraj

def calc_contact_map(psf: str, dcd: list, pairs: list, chunksize: int = 100, stride: int = 1, cutoff: float=0.5,
                     convert: Union[str, list] = 'square') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes in a psf and dcd file and then creates a mdtraj trajectory.
    The trajectory is loaded into an iterator.
    For each chunk, the pair-wise residue distances are calculated
    as specified in the pairs variable.
    The distances are reshaped to a square form for visualization purpose.
    The distances in each frame which are within a cutoff distance are recognized as a contact and denoted as 1.
    Finally, the diagonal and first and second off-diagonal elements are set to zero and the contact probability is calculated as the sum of ones divided by the number of frames.
    Equivalently, the distances with self and i+1, i+2 residues are excluded.
    """
    trajs = []
    for d in dcd:
        u = md.iterload(d, top=psf, chunk=chunksize, stride=stride)
        trajs.append(u)
    distances = np.empty((0, len(pairs)), dtype=np.float32)
    print('Iterating trajectory...')
    for u in trajs:
        for chunk in u:
            distance, residue_pairs = md.compute_contacts(chunk, contacts=pairs)
            distances = np.append(distances, distance, axis=0)
    del trajs
    #u = mda.Universe(psf, dcd)
    #prot = u.select_atoms('protein and not hydrogen')
    #distances = np.zeros((len(u.trajectory[::stride]), len(pairs)), dtype=np.float32)
    #for i, ts in enumerate(u.trajectory[::stride]):
    #    distances[i,:] = self_distance_array(prot.positions)
    print('Iteration done')
    #distances = np.concatenate(distances)
    #residue_pairs = pairs
    if isinstance(convert, str) and convert == 'square':
        contact_probability, contact_maps_avg = convert_to_square_form(distances, residue_pairs, cutoff=cutoff)
        return contact_probability, contact_maps_avg, distances
    elif isinstance(convert, list) and len(convert) == 2:
        contact_probability, contact_maps_avg = convert_to_nonsquare_form(distances, convert, cutoff=cutoff)
        return contact_probability, contact_maps_avg, distances
    else:
        raise ValueError(f'{convert} is not supported')
