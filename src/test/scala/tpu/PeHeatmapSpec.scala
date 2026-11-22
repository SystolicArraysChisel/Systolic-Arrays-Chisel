package tpu

import chiseltest._
import chiseltest.simulator.WriteVcdAnnotation
import firrtl.options.TargetDirAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

import ifaces.SAConfig
import tpu.io.BoardConfig
import tpu.ref.RefModel

/** One ragged MATMUL with a waveform dump, for scripts/pe_heatmap.py. */
class PeHeatmapSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "PE heatmap"

  it should "dump the VCD of one 16x6x5 MATMUL" in {
    val c = SAConfig()
    val annos = Seq(WriteVcdAnnotation, TargetDirAnnotation("test_run_dir/pe_heatmap"))
    test(new TpuCore(c, BoardConfig(clockHz = 400_000, baud = 100_000))).withAnnotations(annos) { dut =>
      dut.clock.setTimeout(0)
      val rnd = new Random(1)
      val a = RefModel.randomMatrix(rnd, 16, 6, lo = 1) // non-zero: only padding is idle
      val b = RefModel.randomMatrix(rnd, 6, 5, lo = 1)
      assert(new ByteHost(dut, c).matmul(a, b)._1 == RefModel.matmul(a, b))
    }
  }
}
