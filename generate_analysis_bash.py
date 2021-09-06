import argparse


# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='refpdb', help='pdb file of ref', required=True)
parser.add_argument('-pca', dest='pca', help='0: no pca; 1: run pca', default=0)
parser.add_argument('-rms', dest='rms', help='0: no rms; 1: run rms', default=0)
parser.add_argument('-struc', dest='struc', help='0: no structural change; 1: run structural change', default=0)
args = parser.parse_args()

pdb = args.refpdb

text='''#!/bin/bash -l
#$ -N anal 
#$ -l h_rt=3:00:00
#$ -pe mpi_16_tasks_per_node 16


eval "$(conda shell.bash hook)"
conda activate myenv
'''

pcapycommands = 'python -u pca.py '
rmspycommands = 'python -u rms.py '
strucpycommands = 'python -u structural_change.py '
psfcommands = '-f '
dcdcommands = '-d '
refcommands = '-p {0:s} '.format(pdb)
outrms = '-o alignto_4ac0apolig'
outstrucchange = '-o 4ac0_apolig_anton'

psf = '../4ac0_apo_ligand/4ac0_apo_prot.psf'
dcd = '../4ac0_apo_ligand/traj_nowater_rewrapped.dcd'
       
psfcommands += psf
psfcommands += ' '
dcdcommands += dcd
dcdcommands += ' '

pcapycommands += (psfcommands + dcdcommands + refcommands + '\n')
rmspycommands += (psfcommands + dcdcommands + refcommands + outrms + '\n')
strucpycommands += (psfcommands + dcdcommands + refcommands + outstrucchange + '\n')

if int(args.rms) == 1:
    text += rmspycommands
if int(args.pca) == 1:
    text += pcapycommands
if int(args.struc) == 1:
    text += strucpycommands

with open('analysis.sh', 'w') as f:
    f.write(text)
