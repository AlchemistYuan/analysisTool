import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
import MDAnalysis.analysis.rms
import numpy as np
from MDAnalysis.analysis import align
import argparse
from trajanalysis import *


def read_argument() -> argparse.Namespace:
    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='psf', nargs='+', help='Lisf of psf files', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files', required=True)
    parser.add_argument('-p', dest='refpdb', help='pdb file of ref', required=True)
    parser.add_argument('-ad', dest='rmsdatoms', help='rmsd atoms selection string', default='backbone')
    parser.add_argument('-af', dest='rmsfatoms', help='rmsf atoms selection string', default='name CA')
    parser.add_argument('-o', dest='outflag', help='output file flag', required=True)
    args = parser.parse_args()
    return args


def main() -> int:
    # Read the command line arguments
    args = read_argument()
    psfs = args.psf
    dcds = args.dcd
    rpdb = args.refpdb
    outflag = args.outflag
    
    if len(psfs) != len(dcds):
        psfs = [psfs[0]] * len(dcds)

    # Load psf and dcd files, then select atoms and align traj
    rmsdatoms = args.rmsdatoms
    rmsfatoms = args.rmsfatoms
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, rmsdatoms)
    
    # Calculate RMSD
    rmsd_all = rmsd(universes, ref, rmsdatoms)
    
    # Calculate RMSF
    #universes = align_traj(psfs, dcds, ref, rmsfatoms)
    #rmsf_all = rmsf(universes, rmsfatoms)
    
    # Save files
    #np.savetxt('rmsf_'+outflag+'.txt', rmsf_all)
    np.savetxt('rmsd_'+outflag+'.txt', rmsd_all)
    

if __name__ == "__main__":
    sys.exit(main())
