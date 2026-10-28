package tpu

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

import tpu.board.EmitBoard
import tpu.io.BoardConfig
import tpu.ref.RefModel

/** The other array sizes that can be built for the board (EmitBoard.configs),
  * run through TpuCore with the same host as TpuSpec (which covers 4x4 in
  * depth).
  */
class TpuConfigsSpec extends AnyFlatSpec with ChiselScalatestTester {

  val b = BoardConfig(clockHz = 400_000, baud = 100_000)

  for ((name, c) <- EmitBoard.configs.toSeq.sortBy(_._1) if name != "4x4") {
    behavior of s"TpuCore $name"

    it should "compute a padded product over several K and N tiles, with requantization" in {
      test(new TpuCore(c, b)) { dut =>
        dut.clock.setTimeout(0)
        val host = new ByteHost(dut, c)
        val rnd = new Random(7)
        // One case only: simulating the larger arrays is slow (TpuSpec covers 4x4 in depth).
        for ((m, k, n, requant) <- Seq((5, 2 * c.rows + 1, c.cols + 2, true))) {
          val a = RefModel.randomMatrix(rnd, m, k)
          val bm = RefModel.randomMatrix(rnd, k, n)
          val shift = if (requant) 9 else 0
          val (got, st) = host.matmul(a, bm, requant = requant, shift = shift)
          assert(got == RefModel.matmul(a, bm, requant, shift = shift), s"${m}x${k}x$n requant=$requant")
          assert(!st.error && !st.overrun, s"STATUS: $st")
        }
      }
    }
  }
}
