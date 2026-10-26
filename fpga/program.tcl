# Loads a bitstream into the FPGA of the Nexys4 DDR over USB-JTAG (lost at
# power off; the board must be on and plugged into the PROG/UART port).
#
#   vivado -mode batch -source fpga/program.tcl -tclargs 8x8   (from the repository root)
#
# The argument is the configuration name (default 4x4): loads
# build/<config>/BoardTop.bit.

set root [file normalize [file dirname [info script]]/..]
set name [expr {$argc >= 1 ? [lindex $argv 0] : "4x4"}]
set bit  "$root/build/$name/BoardTop.bit"

if {![file exists $bit]} {
  error "$bit not found: run fpga/build.tcl -tclargs $name first"
}

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set dev [lindex [get_hw_devices xc7a100t*] 0]
if {$dev eq ""} {
  error "no xc7a100t on the JTAG chain: is the board on and plugged in?"
}
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
refresh_hw_device -quiet $dev
puts "programmed $dev with $bit"

close_hw_target
disconnect_hw_server
close_hw_manager
