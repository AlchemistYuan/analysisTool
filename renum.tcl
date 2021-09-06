set rr [[atomselect top "all"] get residue]

set l [llength $rr]

for {set i 0} {$i<$l} {incr i} {
	lset rr $i [expr [lindex $rr $i] + 1 ]
}

[atomselect top "all"] set resid $rr

#[atomselect top "all"] writepdb sys_renum.pdb
#
##exit
