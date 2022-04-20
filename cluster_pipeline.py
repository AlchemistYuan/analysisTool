import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')

import argparse
import MDAnalysis as mda
from MDAnalysis.analysis import align
from trajanalysis import *

import pyemma
import numpy as np
import matplotlib.pyplot as plt


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
    parser.add_argument('-p', dest='psf', help='The psf file', required=True)
    parser.add_argument('-al', dest='lig', help='The average structure of ligand bound state', required=True)
    parser.add_argument('-ad', dest='dna', help='The average structure of DNA bound state', required=True)
    parser.add_argument('-d', dest='dcd', nargs='+', help='The dcd file', required=True)
    parser.add_argument('-r', dest='ref', help='The reference pdb file', required=True)
    parser.add_argument('-a', dest='atom', help='atom selection', default='name CA')
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    parser.add_argument('-or', dest='rmsdout', help='combined rmsd output file', default='combined_rmsd.txt')
    parser.add_argument('-ob', dest='basinout', help='basin label output file flag', default='basin_out')
    parser.add_argument('-ol', dest='basinlig', help='ligand basin dcd file', default='basin_lig.dcd')
    parser.add_argument('-od', dest='basindna', help='dna basin dcd file', default='basin_dna.dcd')
    parser.add_argument('-l', dest='label', help='cluster label file', required=True)
    parser.add_argument('-c', dest='center', help='cluster center file', required=True)
    args = parser.parse_args()
    return args

def generate_dcd(u, coors, atom_selection='name CA', out='traj.dcd'):
    atoms = u.select_atoms(atom_selection)
    prot = u.select_atoms('protein')
    with mda.Writer(out, n_atoms=prot.n_atoms, multiframe=True) as W:
        for i in range(coors.shape[0]):
            coor = coors[i,:]
            coor = np.reshape(coor, (-1,3))
            atoms.positions = coor
            W.write(prot)

def find_basin_structure(u, basins, atom_selection='name CA'):
    atoms = u.select_atoms(atom_selection)

    nframe = len(u.trajectory)
    basin_structures = []

    for i in range(basins.shape[0]):
        coor = basins[i,:,:]
        coor = coor.flatten()
        minimal_rmsd = 9999.0
        frame_with_minimal_rmsd = 0
        for j, ts in enumerate(u.trajectory):
            rmsd = np.sqrt(np.sum((atoms.positions.flatten() - coor) ** 2) / atoms.n_atoms)
            if rmsd < minimal_rmsd:
                minimal_rmsd = rmsd
                frame_with_minimal_rmsd = j
        basin_structures.append(frame_with_minimal_rmsd)
    basin_structures = np.asarray(basin_structures, dtype=int)
    return basin_structures

def write_basin_structures(u, basin_structures, out='traj.dcd'):
    prot = u.select_atoms('protein')
    with mda.Writer(out, n_atoms=prot.n_atoms, multiframe=True) as W:
        for ts in u.trajectory[basin_structures]:
            W.write(prot)

def main() -> int:
    # Read command line arguments
    args = read_argument()
    psf =  args.psf
    lig = mda.Universe(args.lig)
    dna = mda.Universe(args.dna)
    ref = mda.Universe(args.ref)
    dcds = args.dcd
    atoms = args.atom
    labelfile = args.label
    centerfile = args.center
    outname = args.outflag
    rmsdout = args.rmsdout
    basins_label_out = args.basinout
    basinlig = args.basinlig
    basindna = args.basindna
   
    # Convert the cluster center coordinates into a dcd file with all protein atoms 
    labels = np.loadtxt(labelfile)
    centers = np.loadtxt(centerfile)
    generate_dcd(ref, centers, atom_selection=atoms, out=outname)
    
    # Compute the RMSD of the center dcd relative to the avg lig-bound and dna-bound
    universes = align_traj([psf], [outname], lig, atoms)
    rmsd_lig = rmsd(universes, lig, atoms)
    universes = align_traj([psf], [outname], dna, atoms)
    rmsd_dna = rmsd(universes, dna, atoms)
    rmsd_combined = rmsd_lig[:,:]
    rmsd_combined[:,1] = rmsd_dna[:,0]
    np.savetxt(rmsdout, rmsd_combined)

    '''
    # Find the cluser centers that resemble lig-bound and dna-bound, respectively
    rmsd_lig_large_id = np.argwhere(rmsd_lig[:,0]>3)
    rmsd_dna_large_id = np.argwhere(rmsd_dna[:,0]>4.5)
    rmsd_lig_small_id = np.argwhere(rmsd_lig[:,0]<=2)
    rmsd_dna_small_id = np.argwhere(rmsd_dna[:,0]<=2)
    dna_basin = np.intersect1d(rmsd_lig_large_id, rmsd_dna_small_id)
    ligand_basin = np.intersect1d(rmsd_lig_small_id, rmsd_dna_large_id)
    with open(basins_label_out, 'w') as f:
        f.write('# the first row is the ligand basin\n')
        f.write('# the second row is the dna basin\n')
        for b in ligand_basin:
            f.write(str(b) + ' ')
        f.write('\n')
        for b in dna_basin:
            f.write(str(b) + ' ')
        f.write('\n')

    # Find the frames that are closest to each basin centers
    print(dcds)
    u = mda.Universe(psf, dcds)
    alignment = align.AlignTraj(u, ref, select=atoms, in_memory=True).run()
    centers_u = mda.Universe(psf, outname)
    alignment = align.AlignTraj(centers_u, ref, select=atoms, in_memory=True).run()
    centers_coor = centers_u.trajectory.timeseries(centers_u.select_atoms(atoms), order='fac')
    basin_structures = find_basin_structure(u, centers_coor[ligand_basin,:,:], atom_selection=atoms)
    print('ligand basin:', basin_structures)
    write_basin_structures(u, basin_structures, out=basinlig)
    basin_structures = find_basin_structure(u, centers_coor[dna_basin,:,:], atom_selection=atoms)
    print('dna basin:', basin_structures)
    write_basin_structures(u, basin_structures, out=basindna)
    '''
    return 0


if __name__ == "__main__":
    sys.exit(main())
