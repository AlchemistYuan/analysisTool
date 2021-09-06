set ow [atomselect top "resname TIP3 and name OH2"]

set a {}
foreach o [$ow get index] { lappend a [list $o -1] }

set h1 [atomselect top "resname TIP3 and name H1"]

$h1 setbonds $a
