import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
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
    parser.add_argument('-a', dest='atoms', help='atoms used in alignment', default='protein and backbone and resid 48:63')
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
    if len(psfs) != len(dcds):
        psfs = [psfs[0]] * len(dcds)

    rpdb = args.refpdb
    atoms = args.atoms
    outflag = args.outflag
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
   
    # Find the a4 helix vector for the DNA-bound crystal
    res48a = ref.select_atoms('backbone and segid PROA and resid 48').positions
    res48b = ref.select_atoms('backbone and segid PROB and resid 48').positions
    res63a = ref.select_atoms('backbone and segid PROA and resid 63').positions
    res63b = ref.select_atoms('backbone and segid PROB and resid 63').positions
    a4a_vec = np.mean(res48a, axis=0) - np.mean(res63a, axis=0)
    a4b_vec = np.mean(res48b, axis=0) - np.mean(res63b, axis=0)
    dna_a4 = np.concatenate((a4a_vec.reshape(1,-1), a4b_vec.reshape(1,-1)), axis=0)
 
    angles, ntraj = angle_a4(universes, dna_a4) 
    
    np.savetxt("angle_a4_"+outflag+'.txt', angles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
