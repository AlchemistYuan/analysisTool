import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis.distances import self_distance_array
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
    parser.add_argument('-a', dest='atoms', help='atoms used in calculation', default='protein and name CA')
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

    atoms = args.atoms
    outflag = args.outflag
    
    prota_selection = 'segid PROA and ' + atoms
    protb_selection = 'segid PROB and ' + atoms

    u = mda.Universe(psf, dcds)
    distances_a = calc_pair_wise_distance(u, atoms=prota_selection)
    distances_b = calc_pair_wise_distance(u, atoms=protb_selection)
    distances = np.concatenate((distances_a, distances_b), axis=1)
    np.savetxt('pair_wise_distances_{0:s}.txt'.format(outflag), distances)
    return 0


if __name__ == "__main__":
    sys.exit(main())

