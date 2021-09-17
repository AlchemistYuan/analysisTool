#!/bin/bash

module load vmd/1.9.1

psf=$1
dcd=$2
boxsize=$3
outname=$4

vmd -dispdev text -e /projectnb/cui-buchem/yuchen/scripts/pbcwrap.tcl -args $psf $dcd $boxsize $outname 
