# This script combined and modified three tcl scripts provided by the PSC anton team
# renum.tcl, removeHHbondInTIP3.tcl, and LoadVelFromCHARMMrst.tcl 
# It reads a solvated well-equilibrated CHARMM system and then generates an anton-ready mae file for that system
#

puts "This script reads a solvated well-equilibrated CHARMM system and then generates an anton-ready mae file for that system."
puts "usage:"
puts "vmd -dispdev text -e anton_preparation.tcl -args system.psf system.crd boxsize outname"
puts " system.psf - psf file of the system"
puts " system.crd - crd file of the system"
puts " boxsize - boxsize of the pbc box"
puts " outname - output name for the rewrapped trajectory" 

proc anton_prep {syspsf syscrd sysrst boxsize outname} {
    package require pbctools
    
    # load psf, crd, and set the box 
    mol new $syspsf
    animate read crd $syscrd waitfor all 0
    pbc set {$boxsize $boxsize $boxsize}

    # modified from the renum.tcl script
    puts "renumbering residues" 
    set rr [[atomselect top "all"] get residue]

    set l [llength $rr]
    
    for {set i 0} {$i<$l} {incr i} {
            lset rr $i [expr [lindex $rr $i] + 1 ]
    }
    
    [atomselect top "all"] set resid $rr
    
    # modified from removeHHbondInTIP3.tcl script
    puts "Removing bonds between hydrogens in TIP3P water"
    set ow [atomselect top "resname TIP3 and name OH2"]

    set a {}
    foreach o [$ow get index] { lappend a [list $o -1] }
    
    set h1 [atomselect top "resname TIP3 and name H1"]
    
    $h1 setbonds $a
    
    # modified from LoadVelFromCHARMMrst.tcl
    puts "Load CHARMM velocity from RST file"
    set infile [open $sysrst r]

    set a {}

    gets $infile line
    while { [string compare $line " !VX, VY, VZ"] != 0  } {
        gets $infile line
    }
    puts "Reached velocities section"
    set number 0
    gets $infile line
    puts $line
    while { [string compare $line " !X, Y, Z"] != 0  } {

        if { [string compare $line ""] != 0 } {
            set vx [expr 20.4582651391 * [string map {D E} [string trim [string range $line  0 21]]]]
            set vy [expr 20.4582651391 * [string map {D E} [string trim [string range $line 22 43]]]]
            set vz [expr 20.4582651391 * [string map {D E} [string trim [string range $line 44 66]]]]

            set l [string trim $line]
            lappend a [list $vx $vy $vz]
            incr number
        }
        gets $infile line
    }
    puts "Number of read entries: $number"
    close $infile

    set aa [atomselect top "all"]
    $aa set {vx vy vz} $a

    animate write mae $outname beg 0 end 1 skip 1 sel $aa waitfor all 1
    return
}

#set syspsf [lindex $argv 0];
#set syscrd [lindex $argv 1];
#set sysrst [lindex $argv 2];
#set boxsize [lindex $argv 3];
#set outname [lindex $argv 4];

#anton_prep $syspsf $syscrd $sysrst $boxsize $outname;
#exit

