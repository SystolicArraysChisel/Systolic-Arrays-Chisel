package ifaces

import chisel3._
import chisel3.util._

/**
	* Tells the data mover which tile to feed: where its weights 
	* and activations start, their row spacing, and how many 
	* rows of A to stream.
	* 
	* The data mover computes the address of weight row r 
	* as bAddr + r · ldB and of activation row i as aAddr + i · ldA. 
	*/
class TileCmd(c: SAConfig) extends Bundle {
	val aAddr = UInt(c.addrW.W)
	val bAddr = UInt(c.addrW.W)

	val ldA = UInt(c.dimW.W)
	val ldB = UInt(c.dimW.W)
	val m = UInt(c.dimW.W)
}