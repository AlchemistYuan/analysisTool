import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
from trajanalysis import *
import argparse

from correlations import *
from utils import *


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
    parser.add_argument('-a', dest='atoms', help='atoms used in alignment', default='protein and name CA')
    parser.add_argument('-b', dest='begin', type=int, help='first frame', default=0)
    parser.add_argument('-e', dest='end', type=int, help='last frame', default=-1)
    parser.add_argument('-s', dest='stride', type=int, help='stride to read frames', default=1)
    parser.add_argument('-m', dest='method', help='type of correlation', default='pearson')
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
    begin = args.begin
    stop = args.end
    stride = args.stride
    outflag = args.outflag
    method = args.method
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
    corr_mat = calc_correlation(universes, atoms, begin, stop, stride, method)
    
    if method == 'pearson':
        np.savetxt("corr_"+outflag+'.txt', corr_mat)
    elif method == 'spearson':
        np.savetxt("spearmancorr_"+outflag+'.txt', corr_mat)
    elif method == 'mutual':
        np.savetxt("mutualinfo_"+outflag+'.txt', corr_mat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
