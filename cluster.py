import sys
sys.path.append('/projectnb/cui-buchem/yuchen/scripts')

import argparse
import pyemma
import MDAnalysis as mda
import mdtraj as md

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
    parser.add_argument('-f', dest='feat', help='The feature file', required=True)
    parser.add_argument('-k', dest='k', type=int, help='The number of clusters', default=250)
    parser.add_argument('-o', dest='outflag', help='output file flag', default='out')
    args = parser.parse_args()
    return args

def main() -> int:
    '''
    The main function to run the script.
    '''
    args = read_argument()
    feat = np.load(args.feat)
    k = args.k
    outflag = args.outflag
    centers, labels = kmeans_scikit(feat, k=k) 
    np.savetxt("kmeans_centers_"+outflag+'.txt', centers)
    np.savetxt("kmeans_labels_"+outflag+'.txt', labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
