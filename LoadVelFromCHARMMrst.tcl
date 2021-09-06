puts "AddVelFromRST {filenameRST}"
puts "   adding velocities from CHARMM restart file"
puts "   Something can be still wrong with units. Desmond probably uses Angstrems/ps and CHARMM probably uses AKMA in restart file."
puts "   So in this script AKMA units is converted to A/ps"

proc AddVelFromRST {filenameRST} {
    set infile [open $filenameRST r]
    
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
    return
}

