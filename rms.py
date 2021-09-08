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
    
    # Load psf and dcd files, then select atoms and align traj
    atoms = 'protein and not name H*'
    rmsdatoms = 'protein and backbone'
    rmsfatoms = 'protein and name CA'
    ref = mda.Universe(rpdb)
    universes = align_traj(psfs, dcds, ref, atoms)
    
    # Calculate RMSD
    rmsd_all = rmsd(universes, ref, rmsdatoms)
    
    # Calculate RMSF
    rmsf_all = rmsf(universes, rmsfatoms)
    
    # Save files
    np.savetxt('rmsf_'+outflag+'.txt', rmsf_all)
    np.savetxt('rmsd_'+outflag+'.txt', rmsd_all)
    
    universes = align_traj(psfs, dcds, ref, "protein and backbone and not resid 156:164")
    
    # Calculate RMSD
    rmsd_all = rmsd(universes, ref, "protein and backbone and not resid 156:164")
    
    np.savetxt("rmsd_"+outflag+"_noloop.txt", rmsd_all)


if __name__ == "__main__":
    sys.exit(main())
