import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')

import argparse
import pyemma
import MDAnalysis as mda

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
    parser.add_argument('-f', dest='psf', help='The psf file', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files to be transformed', default=None)
    parser.add_argument('-dl', dest='dcdfile', type=str, help='File containing the names of dcd files', default=None)
    parser.add_argument('-p', dest='ref', help='pdb file of ref', required=True)
    parser.add_argument('-a', dest='atoms', help='atoms used in alignment', default='name CA')
    parser.add_argument('-l', dest='lag', type=int, help='lag time of TICA', default=100)
    parser.add_argument('-n', dest='dim', type=int, help='dimension of the projection', default=10)
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    args = parser.parse_args()
    return args

def main() -> int:
    '''
    The main function to run the script.
    '''
    args = read_argument()
    psfs = args.psf
    if args.dcdfile != None:
        dcdfile = args.dcdfile
        dcds = []
        with open(dcdfile, 'r') as f:
            line = f.readline()
            while line:
                l = line.strip()
                dcds.append(l)
                line = f.readline()
    psfs = [psfs] * len(dcds)
    outflag = args.outflag
    lag = args.lag
    dim = args.dim
    refpdb = args.ref
    atoms = args.atoms
    ref = mda.Universe(refpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
    if args.dcd:
        psfs_transform = [psfs[0]] * len(args.dcd)
        transform = align_traj(psfs_transform, args.dcd, ref, atoms)
    else:
        transform = None
    output, transformed, eigvecs, eigvals, timescales = tica_pyemma(universes, lag=lag, dim=dim, atoms=atoms, transform=transform)
    np.savetxt("tica_eigvec_"+outflag+'.txt', eigvecs)
    np.savetxt("tica_eigval_"+outflag+'.txt', eigvals)
    np.savetxt("tica_proj_"+outflag+'.txt', output)
    np.savetxt("tica_timescale_"+outflag+".txt", timescales)
    if type(transformed) != 'NoneType':
        np.savetxt("tica_transformed_"+outflag+".txt", transformed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
