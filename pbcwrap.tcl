#
# read nowater traj and rewrap protein
#

puts "This script reads in the nowater traj and then rewrap the two monomers together"
puts "usage:"
puts "vmd -dispdev text -e pbcwrap.tcl -args system.psf system.dcd boxsize"
puts " system.psf - psf file of the nowater system"
puts " system.dcd - dcd file of the nowater system"
puts " boxsize - boxsize of the pbc box"
puts " outname - output name for the rewrapped trajectory" 

proc wrap_protein {syspsf sysdcd boxsize outname} {
    package require pbctools
    mol new $syspsf
    animate read dcd $sysdcd waitfor all 0
    pbc set "$boxsize $boxsize $boxsize" -all
    pbc wrap -center com -centersel "segid PROA" -compound residue -all
    pbc wrap -center com -centersel "protein" -compound residue -all
    animate write dcd ${outname}_rewrapped.dcd beg 0 end -1 skip 1 waitfor all
}

set syspsf [lindex $argv 0];
set sysdcd [lindex $argv 1];
set boxsize [lindex $argv 2];
set outname [lindex $argv 3];

wrap_protein $syspsf $sysdcd $boxsize $outname;

exit

