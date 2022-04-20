import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
from trajanalysis import *
import argparse


def read_argument() -> argparse.Namespace:
    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='psf', nargs='+', help='Lisf of psf files', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='List of dcd files', default=None)
    parser.add_argument('-l', dest='dcdfile', help='File with the names of dcd files', default=None)
    parser.add_argument('-p', dest='refpdb', help='pdb file of ref', required=True)
    parser.add_argument('-o', dest='outflag', help='output file flag', required=True)
    parser.add_argument('-a', dest='atoms', help='atoms used in PCA', default='name CA')
    parser.add_argument('-ipc', dest='pcs', help='file of PC coordinates', required=True)
    parser.add_argument('-npc', dest='npcs', help='number of PC coordinates', default=10)
    parser.add_argument('-avg', dest='avg', help='average structure of pca', required=True)
    args = parser.parse_args()
    return args


def main() -> int:
    args = read_argument()
    psfs = args.psf
    dcd = args.dcd
    dcdfile = args.dcdfile
    print('Reading dcd files...')
    if dcd != None and dcdfile != None:
        print('WARNING! Cannot provide dcd and dcdfile simultaneously.')
    elif dcd != None and dcdfile == None:
        dcds = dcd
    elif dcdfile != None and dcd == None:
        dcds = []
        with open(dcdfile, 'r') as f:
            line = f.readline()
            while line:
                dcds.append(line.strip())
                line = f.readline()
    if len(psfs) != len(dcds):
        psfs = [psfs[0]] * len(dcds)

    rpdb = args.refpdb
    outflag = args.outflag
    atoms = args.atoms
    pcfile = args.pcs
    npcs = int(args.npcs)
    avg = mda.Universe(args.avg)
    avg_pos = avg.select_atoms(atoms).positions
    pc = np.loadtxt(pcfile, max_rows=int(npcs))
    
    ref = mda.Universe(rpdb)
    print('Aligning trajs...')
    universes = align_traj(psfs, dcds, ref, atoms)
    refatoms = ref.select_atoms(atoms)
    ref_coor = refatoms.positions

    ntraj = []
    for u in universes:
        ntraj.append(len(u.trajectory))
    total_traj = np.sum(ntraj)
    proj = np.zeros((total_traj, npcs))
    current_traj = 0
    print('Calculating projections...')
    for i in range(len(universes)):
        uni = universes[i]
        proj[current_traj:current_traj+ntraj[i],:] = projection(pc, avg_pos, uni, atoms) 
        current_traj += ntraj[i]

    np.savetxt("projection_{0:s}.txt".format(outflag), proj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
