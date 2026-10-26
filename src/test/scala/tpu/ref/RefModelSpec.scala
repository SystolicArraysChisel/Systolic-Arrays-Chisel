package tpu.ref

import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class RefModelSpec extends AnyFlatSpec {

  behavior of "Quant and RefModel"

  it should "round ties toward +infinity, without the floor(x + 0.5) error" in {
    val cases = Seq(2.5 -> 3, -2.5 -> -2, 0.5 -> 1, -0.5 -> 0, 0.49999999999999994 -> 0, -2.6 -> -3)
    for ((v, r) <- cases) assert(Quant.roundHalfUp(v) == r, s"$v")
  }

  it should "reproduce the worked example of the workload document" in {
    assert(Seq(0.5, -0.2).map(Quant.quantize(_, 0.005)) == Seq(100, -40))
    assert(Quant.requantize(2800, 8) == 11) // truncation would give 10
    assert(math.abs(Quant.requantScale(0.005, 0.005, 8) - 0.0064) < 1e-15)
  }

  it should "saturate to +-127 and never produce -128" in {
    assert(Quant.quantize(-128.0, 1.0) == -127)
    assert(Quant.requantize(-1000, 0) == -127 && Quant.requantize(1000, 0) == 127)
  }

  it should "compute the example product of the workload document" in {
    val a = Seq(Seq(1L, 2, 3, 4), Seq(5L, 6, 7, 8), Seq(1L, 0, -1, 2))
    val b = Seq(Seq(1L, 0), Seq(0L, 1), Seq(1L, 1), Seq(2L, -1))
    assert(RefModel.matmul(a, b) == Seq(Seq(12L, 1), Seq(28L, 5), Seq(4L, -3)))
    assert(RefModel.matmul(a, b, relu = true) == Seq(Seq(12L, 1), Seq(28L, 5), Seq(4L, 0)))
  }

  it should "reject an accumulator overflow" in {
    val a = Seq(Seq.fill(3)(127L))
    val b = Seq.fill(3)(Seq(127L))
    assertThrows[ArithmeticException](RefModel.matmulAcc(a, b, accWidth = 16))
  }

  it should "generate random int8 matrices of the requested shape and range" in {
    val m = RefModel.randomMatrix(new Random(1), 3, 5)
    assert(m.length == 3 && m.forall(_.length == 5))
    assert(m.flatten.forall(v => v >= -128 && v <= 127))
  }
}
