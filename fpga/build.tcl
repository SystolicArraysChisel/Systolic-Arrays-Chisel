# Builds the Nexys4 DDR bitstream of one array configuration from the Verilog
# written by EmitBoard (non-project flow) and writes the timing, utilization
# and power reports.
#
#   sbt "runMain tpu.board.EmitBoard 8x8"
#   vivado -mode batch -source fpga/build.tcl -tclargs 8x8   (from the repository root)
#
# The argument is the configuration name (default 4x4): the input is
# generated/<config>/, the output build/<config>/: BoardTop.bit, config.json
# (the parameters the host needs), reports/*.rpt and summary.txt (WNS, LUT,
# FF, BRAM, DSP, power). The bitstream is written even when timing fails, but
# the script then exits with an error.

set part xc7a100tcsg324-1
set root [file normalize [file dirname [info script]]/..]

set name    [expr {$argc >= 1 ? [lindex $argv 0] : "4x4"}]
set gen     "$root/generated/$name"
set verilog "$gen/BoardTop.v"
set out     "$root/build/$name"
set xdc     "$root/fpga/nexys4ddr.xdc"
set rpt     "$out/reports"

if {![file exists $verilog]} {
  error "$verilog not found: run sbt \"runMain tpu.board.EmitBoard $name\" first"
}
file mkdir $rpt
file copy -force "$gen/config.json" "$out/config.json"

# ---- synthesis ----
read_verilog $verilog
read_xdc $xdc
synth_design -top BoardTop -part $part
report_utilization -file $rpt/utilization_synth.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $rpt/utilization_hier.rpt

# ---- implementation ----
opt_design
place_design
phys_opt_design
route_design

report_timing_summary -max_paths 10 -report_unconstrained -file $rpt/timing_summary.rpt
report_timing -max_paths 20 -sort_by slack -file $rpt/timing_paths.rpt
report_utilization -file $rpt/utilization.rpt
report_power -file $rpt/power.rpt
report_drc -file $rpt/drc.rpt
write_checkpoint -force $out/BoardTop_routed.dcp
write_bitstream -force $out/BoardTop.bit

# ---- summary ----
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]

# "Used" column of a row of the utilization report
proc used {text row} {
  set n "?"
  regexp [format {\|\s*%s\s*\|\s*([0-9.]+)} $row] $text -> n
  return $n
}
set fh [open $rpt/utilization.rpt]
set util [read $fh]
close $fh
set luts [used $util {Slice LUTs}]
set ffs  [used $util {Slice Registers}]
set ramb [used $util {Block RAM Tile}]
set dsps [used $util {DSPs}]
set power "?"
set fh [open $rpt/power.rpt]
regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} [read $fh] -> power
close $fh

set summary [join [list \
  "verilog        $verilog" \
  "WNS (ns)       $wns" \
  "WHS (ns)       $whs" \
  "LUT            $luts / 63400" \
  "FF             $ffs / 126800" \
  "BRAM (36 Kb)   $ramb / 135" \
  "DSP            $dsps / 240" \
  "power (W)      $power"] "\n"]
set fh [open $out/summary.txt w]
puts $fh $summary
close $fh
puts "\n$summary\n"

if {$wns < 0 || $whs < 0} {
  error "timing not met (WNS $wns ns, WHS $whs ns): see $rpt/timing_paths.rpt"
}
puts "bitstream: $out/BoardTop.bit"
