import sys
sys.path.append('../scripts/')

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
from trajanalysis import *
import argparse


# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='psf', help='Lisf of psf files', required=True)
parser.add_argument('-d', dest='dcd', help='List of dcd files', required=True)
parser.add_argument('-p', dest='refpdb', help='pdb file of ref', required=True)
parser.add_argument('-o', dest='outflag', help='output file flag', required=True)
parser.add_argument('-a', dest='atoms', help='atoms used in PCA', required=True)
parser.add_argument('-ipc', dest='pcs', help='file of PC coordinates', required=True)
parser.add_argument('-npc', dest='npcs', help='number of PC coordinates', default=2, required=True)
parser.add_argument('-avg', dest='avg', help='average structure of pca', required=True)
args = parser.parse_args()

psf = args.psf
dcd = args.dcd
rpdb = args.refpdb
outflag = args.outflag
atoms = args.atoms
pcfile = args.pcs
npcs = int(args.npcs)
avg = mda.Universe(args.avg)
avg_pos = avg.select_atoms(atoms).positions.flatten()
pc = np.loadtxt(pcfile, max_rows=int(npcs))

ref = mda.Universe(rpdb)
u = mda.Universe(psf, dcd)
alignment = align.AlignTraj(u, ref, select=atoms, in_memory=True).run()
refatoms = ref.select_atoms(atoms)

ntraj = len(u.trajectory)
proj = np.zeros((ntraj, npcs))
coor = u.trajectory.timeseries(u.select_atoms(atoms), order='fac')
for i in range(ntraj):
    frame = coor[i,:,:]
    frame = frame.flatten()
    frame -= avg_pos
    dotproducts = np.dot(pc, frame)
    proj[i,:] = dotproducts

np.savetxt("projection_{0:s}.txt".format(outflag), proj)
