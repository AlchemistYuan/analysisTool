import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import eig
from sklearn.decomposition import PCA
from trajanalysis import *
import argparse


def read_argument() -> argparse.Namespace:
    '''
    This function reads all command line arguments and return the Namespace object.
    
    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
        The Namespace object to store all arguments
    '''
    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='psf', nargs='+', help='Lisf of psf files', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files', required=True)
    parser.add_argument('-p', dest='refpdb', help='pdb file of ref', required=True)
    parser.add_argument('-a', dest='atoms', help='atoms used in alignment', default='protein and backbone')
    parser.add_argument('-pa', dest='pcaatoms', help='atoms used in PCA', default='protein and name CA')
    parser.add_argument('-o', dest='outflag', nargs='+', help='output file flag', default=['out'])
    args = parser.parse_args()
    return args


def main() -> int:
    '''
    The main function to run the script.
    '''
    psfs = args.psf
    dcds = args.dcd
    rpdb = args.refpdb
    atoms = args.atoms
    pcaatoms = args.pcaatoms
    outflag = args.outflag
 
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
    proj_all, pcs, eigvals, ref_coor, ntraj = pca_scikit(universes, pcaatoms)
    ref_atoms = ref.select_atoms(pcaatoms)
    ref_atoms.positions = ref_coor
    np.savetxt('ref_coor_heavy_4ac0_all_heavy.txt', ref_coor)
    heavy_only_u.atoms.write('ref_coor_heavy_4ac0_all_heavy.pdb')
    
    ntraj = np.asarray(ntraj, dtype=int)
    cum_ntraj = np.cumsum(ntraj)[:-1]
    proj_ligand_bound = proj_all[:cum_ntraj[0],:]
    proj_dna_bound = proj_all[cum_ntraj[-1]:,:]
    proj_apo_state = proj_all[cum_ntraj[0]:cum_ntraj[-1],:]
    
    np.savetxt("projection_4ac0_all_heavy.txt", proj_apo_state)
    np.savetxt("projection_4ac0_ligand_bound_heavy.txt", proj_ligand_bound)
    np.savetxt("projection_4ac0_dna_bound_heavy.txt", proj_dna_bound)
    np.savetxt("pcs_4ac0_all_heavy.txt", pcs)
    np.savetxt("variance_4ac0_all_heavy.txt", eigvals)
