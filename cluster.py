import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')

import argparse
import pyemma
import MDAnalysis as mda
import mdtraj as md

from msmanalysis import *
from trajanalysis import *

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
    parser.add_argument('-f', dest='feat', help='The feature file', default=None)
    parser.add_argument('-p', dest='psf', nargs='+', help='The psf file', default=None)
    parser.add_argument('-d', dest='dcd', nargs='+', help='The dcd file', default=None)
    parser.add_argument('-r', dest='ref', help='The reference pdb file', default=None)
    parser.add_argument('-a', dest='atom', type=str, help='The atom selection string', default='name CA')
    parser.add_argument('-k', dest='k', type=int, help='The number of clusters', default=250)
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    parser.add_argument('-c', dest='convert', help='Whether convert the center coordiantes to pdb files', default=None)
    args = parser.parse_args()
    return args


def convert_centers_to_pdb(u: mda.Universe, atoms: str, centers: np.ndarray, outflag: str) -> None:
    '''
    This function takes in a trajectory and the frame indices cloesest ot kmeans centers
    and then save the corresponding frame to individual pdb files.

    Parameters
    ----------
    u : mda.Universe
        The trajectory.
    atoms : str
        The atom selection string.
    centers : np.ndarray
        The frame indices closest to each cluster.
    outflag : str
        The output file name patterns.

    Returns
    -------
    None
    '''
    ncenter = len(centers)
    atomgroup = u.select_atoms(atoms)
    for i in range(ncenter):
        center = centers[i]
        outname = 'centroid_structure_{0:d}_{1:s}.pdb'.format(i, outflag)
        with mda.Writer(outname, n_atoms=atomgroup.n_atoms, multiframe=False) as W:
            for ts in u.trajectory[center:center+1]:
                W.write(atomgroup)


def main() -> int:
    '''
    The main function to run the script.
    '''
    args = read_argument() 
    atoms = args.atom
    ref = mda.Universe(args.ref)
    coors = []
    universe = []
    if not args.feat:
        if args.psf and args.dcd:
            psfs = args.psf
            dcds = args.dcd
            if len(psfs) == 1:
                psfs = psfs * len(dcds)
            for i in range(len(dcds)):
                u = mda.Universe(psfs[i], dcds[i])
                alignment = align.AlignTraj(u, ref, select=atoms, in_memory=True).run()
                coor = u.trajectory.timeseries(u.select_atoms(atoms), order='fac')
                coors.append(coor.ravel().reshape(coor.shape[0],3*coor.shape[1]))
                universe.append(u)
            feat = np.concatenate(coors)
            print(feat.shape)
        else:
            print('psf file and/or dcd file are missing!')
    else:       
        feat = np.load(args.feat)

    k = args.k
    outflag = args.outflag
    centers, labels, cluster_center_indices = kmeans_scikit(feat, k=k) 
    np.savetxt("kmeans_centers_"+outflag+'.txt', centers)
    np.savetxt("kmeans_labels_"+outflag+'.txt', labels, fmt='%d')
    np.savetxt('kmeans_cluster_center_indices_' + outflag + '.txt', cluster_center_indices, fmt='%d')
 
    if isinstance(args.convert, str):
        convert = args.convert
        protein = ref.select_atoms(convert)
        coors_prot = []
        for u in universe:
            coor = u.trajectory.timeseries(u.select_atoms(convert), order='fac')
            coors_prot.append(coor)
        coors_prot = np.concatenate(coors_prot)
        u_new = mda.Merge(protein).load_new(coors_prot, order="fac")
        convert_centers_to_pdb(u_new, convert, cluster_center_indices, outflag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
