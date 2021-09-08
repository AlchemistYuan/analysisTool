import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import argparse
import numpy as np
import MDAnalysis as mda
from trajanalysis import *


def read_argument() -> argparse.Namespace:
    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='psf', help='A file with the names of psf files', required=True)
    parser.add_argument('-d', dest='dcd', help='A file with the names of dcd files', required=True)
    parser.add_argument('-i', dest='id', nargs='+', help='Frame id to be written out', required=True)
    parser.add_argument('-p', dest='ref', help='The reference structure', required=True)
    parser.add_argument('-a', dest='atoms', help='Atoms used in alignment', default='name CA')
    parser.add_argument('-l', dest='labels', nargs='+', help='Labels for the output files', required=True)
    parser.add_argument('-o', dest='outflag', help='output file flag', required=True)
    args = parser.parse_args()
    return args

def main() -> int:
    args = read_argument()
    psffile = args.psf
    dcdfile = args.dcd
    ref = mda.Universe(args.ref)
    atoms = args.atoms
    id_all = args.id
    labels = args.labels
    outflag = args.outflag
 
    # Read the psf and dcd names from the input files
    psfs, dcds = [], []
    with open(dcdfile, 'r') as f:
        line = f.readline()
        while line:
            l = line.strip()
            dcds.append(l)
            line = f.readline()

    with open(psffile, 'r') as f:
        line = f.readline()
        while line:
            l = line.strip()
            psfs.append(l)
            line = f.readline()
    
    # Align all trajectories agains the same reference strucure
    universes = align_traj(psfs, dcds, ref, atoms)

    for i in range(len(labels)):
        u = universes[i] 
        prot = u.select_atoms('protein')
        value = int(id_all[i])
        for ts in u.trajectory[value:value+1]:
            prot.atoms.write('{0:s}_{1:s}.pdb'.format(outflag, labels[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
