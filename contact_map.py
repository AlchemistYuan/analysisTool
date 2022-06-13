import sys
import argparse
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')

import numpy as np
import mdtraj as md

from trajanalysis import *
from msmanalysis import *
from distributions import *


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
    parser.add_argument('-f', dest='psf', help='Lisf of psf files', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files', required=True)
    parser.add_argument('-c', dest='cut', type=float, help='Contacts cutoff in nm', default=0.5)
    parser.add_argument('-b', dest='convert', help='Convert to square form', default=None)
    parser.add_argument('-k', dest='chunk', type=int, help='Chunk size', default=100)
    parser.add_argument('-s', dest='stride', type=int, help='Strie to read frames', default=1)
    parser.add_argument('-p', dest='pair', help='List of atom pair', default=None)
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    args = parser.parse_args()
    return args


def main() -> int:
    args = read_argument()
    psf = args.psf
    dcd = args.dcd
   
    # The residue pairs between which the distances to be calculated are hard-coded here. 
    pairs = []
    if args.pair == None:
        for i in range(404):
            for j in range(404):
                pair = [i,j]
                pairs.append(pair)
    else:
        with open(args.pair, 'r') as f:
            pair_a = f.readline().split()
            pair_b = f.readline().split()
        pair_a = np.asarray(pair_a, dtype=int)
        pair_b = np.asarray(pair_b, dtype=int)
        for i in pair_a:
            for j in pair_b:
                pair = [i,j]
                pairs.append(pair)
    pairs = np.asarray(pairs, dtype=int)
    if isinstance(args.convert, type(None)):
        convert = 'square'
    elif isinstance(args.convert, str):
        convert = args.convert.split(' ')
        convert = np.asarray(convert, dtype=int).tolist()

    contact_probability, contact_maps_avg, distances = calc_contact_map(psf, dcd, pairs, chunksize=args.chunk, convert=convert, stride=args.stride, cutoff=args.cut)
    np.savetxt(f'contact_probability_{args.outflag}.txt', contact_probability, fmt='%1.3f')
    np.savetxt(f'contact_distance_avg_{args.outflag}.txt', contact_maps_avg)
    np.save(f'contact_distance_all_{args.outflag}.npy', distances)
    return 0


if __name__ == "__main__":
    sys.exit(main())
