import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts/')
import argparse

import numpy as np
import pandas as pd
import mdtraj as md


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
    parser.add_argument('-m', dest='mode', help='SASA mode', default='residue')
    parser.add_argument('-o', dest='out', help='output name', required=True)
    args = parser.parse_args()
    return args


def get_sasa(traj: md.Trajectory, probe_radius: float=0.14, 
             n_sphere_points: int=960, mode: str='residue') -> np.ndarray:
    '''
    Compute the solvent accessible surface area for selected residues.
    
    Parameters
    ----------
    traj : md.Trajectory
        mdtraj trajectory object
    probe_radius: float
        The radius of the probe, in nm
    n_sphere_points : int
        The number of points representing the surface of each atom, higher values leads to more accuracy
    mode : str
        In mode == 'atom', the extracted areas are resolved per-atom In mode == 'residue', this is consolidated down to the per-residue SASA by summing over the atoms in each residue.


    '''
    sasa = md.shrake_rupley(traj, 
                            probe_radius=probe_radius, 
                            n_sphere_points=n_sphere_points, 
                            mode=mode)
    return sasa


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
    
    mode = args.mode
    out = args.out

    sasa_all = []
    for i in range(len(psfs)):
        psf = psfs[i]
        dcd = dcds[i]
        sasa_current = []
        for chunk in md.iterload(dcd, top=psf, stride=20, chunk=1000):
            sasa = get_sasa(chunk, mode=mode)
            sasa_current.append(sasa)
        sasa_current = np.concatenate(sasa_current)
        sasa_all.append(sasa_current)
    sasa_all = np.concatenate(sasa_all)
    np.savetxt(out, sasa_all, fmt='%.3f')
    return 0


if __name__ == "__main__":
    sys.exit(main())
