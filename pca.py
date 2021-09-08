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
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files', default=None)
    parser.add_argument('-dl', dest='dcdfile', type=str, help='File containing the names of dcd files', default=None)
    parser.add_argument('-p', dest='refpdb', help='pdb file of ref', required=True)
    parser.add_argument('-a', dest='atoms', help='atoms used in alignment', default='protein and backbone')
    parser.add_argument('-pa', dest='pcaatoms', help='atoms used in PCA', default='protein and name CA')
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    args = parser.parse_args()
    return args


def main() -> int:
    '''
    The main function to run the script.
    '''
    args = read_argument()
    psfs = args.psf
    if args.dcd != None:
        dcds = args.dcd
    elif args.dcdfile != None:
        dcdfile = args.dcdfile
        dcds = []
        with open(dcdfile, 'r') as f:
            line = f.readline()
            while line:
                l = line.strip()
                dcds.append(l)
                line = f.readline()
    rpdb = args.refpdb
    atoms = args.atoms
    pcaatoms = args.pcaatoms
    outflag = args.outflag
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
    proj_all, pcs, eigvals, ref_coor, ntraj = pca_scikit(universes, pcaatoms)
    ref_atoms = ref.select_atoms(pcaatoms)
    ref_atoms.positions = ref_coor
    np.savetxt('avg_coor_pca.txt', ref_coor)
    ref.atoms.write('avg_coor_pca.pdb')
    
    ntraj = np.asarray(ntraj, dtype=int)
    cum_ntraj = np.cumsum(ntraj)[:-1]
    
    np.savetxt("pcs_"+outflag+'.txt', pcs)
    np.savetxt("variance_"+outflag+'.txt', eigvals)
    np.savetxt("projection_"+outflag+'.txt', proj_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
