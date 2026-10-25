package tpu.board

import chisel3._

import ifaces.DebugInfo

/** Ports of a design that BoardTop can place on the FPGA: the serial line in
  * both directions and the debug flags for the LEDs.
  */
class CoreIO extends Bundle {
  val rxd = Input(Bool()) // serial data from the PC, idle high
  val txd = Output(Bool()) // serial data to the PC, idle high
  val debug = Output(new DebugInfo)
}

/** A design BoardTop can wrap, such as Tpu. */
abstract class CoreModule extends Module {
  val io: CoreIO
}
