# Load a Wavefront OBJ file into VMD as graphics-object triangles.
#
# Selfie writes OBJ with only `v x y z` and `f i j k` lines (1-based
# face indices, no normals or texture coords), so this trivial parser
# is sufficient. Useful because stock VMD does not ship a Wavefront
# molfile plugin.
#
# Usage (from VMD's Tk console, Extensions -> Tk Console):
#
#     source scripts/load_obj.tcl
#     load_obj /tmp/sphere.obj                       ;# cyan, transparent
#     load_obj /tmp/sphere.obj cyan Transparent
#     load_obj /tmp/sphere.obj red Opaque
#
# why: VMD's `graphics top …` fails with "invalid graphics molecule"
# when no molecule is loaded (or the top one was deleted). We create a
# fresh empty molecule purely to host the triangles, capture its id,
# and use that id everywhere — robust whether or not other molecules
# are already loaded.
#
# Returns the molecule id of the created surface, so the caller can
# `mol delete $id` to remove it later or stash it for further use.
#
# The mesh is rendered as separate `draw triangle` primitives, one
# per face. Performance is acceptable up to ~10k triangles; very
# dense meshes may want STL/PLY export and a real plugin instead.
proc load_obj {path {color cyan} {material Transparent}} {
    set fp [open $path r]
    set verts {}
    set n_faces 0

    set molid [mol new]
    mol rename $molid [file tail $path]
    graphics $molid color $color
    graphics $molid material $material

    while {[gets $fp line] >= 0} {
        if {[string match "v *" $line]} {
            lappend verts [lrange $line 1 3]
        } elseif {[string match "f *" $line]} {
            # OBJ face indices are 1-based; Tcl lists are 0-based.
            set i [expr {[lindex $line 1] - 1}]
            set j [expr {[lindex $line 2] - 1}]
            set k [expr {[lindex $line 3] - 1}]
            graphics $molid triangle \
                [lindex $verts $i] \
                [lindex $verts $j] \
                [lindex $verts $k]
            incr n_faces
        }
    }
    close $fp

    display resetview
    puts "load_obj: mol $molid — $n_faces triangles, [llength $verts] vertices ($path)"
    return $molid
}
