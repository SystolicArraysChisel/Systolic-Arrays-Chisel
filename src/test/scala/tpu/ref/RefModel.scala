package tpu.ref

import scala.util.Random

/** Bit-exact model of one MATMUL, the Scala copy of host/tpu_host/reference.py.
  *
  * C = A . B on int8 inputs with an accWidth-bit accumulator, then the
  * post-processing of the spec: optional requantization, then optional ReLU.
  * Matrices are row-major Seq of rows.
  */
object RefModel {
  type Mat = Seq[Seq[Long]]

  /** Exact integer product, checked to fit in the accumulator. */
  def matmulAcc(a: Mat, b: Mat, accWidth: Int = 32): Mat = {
    val k = b.length
    require(a.forall(_.length == k), s"cannot multiply ${a.length}x${a.headOption.map(_.length)} by ${k}x?")
    val n = b.headOption.map(_.length).getOrElse(0)
    val (lo, hi) = (-(1L << (accWidth - 1)), (1L << (accWidth - 1)) - 1)
    a.map { row =>
      (0 until n).map { j =>
        val s = (0 until k).map(i => row(i) * b(i)(j)).sum
        if (s < lo || s > hi) throw new ArithmeticException(s"a sum does not fit in $accWidth bits")
        s
      }
    }
  }

  def postProcess(acc: Long, requant: Boolean, relu: Boolean, shift: Int): Long = {
    val y = if (requant) Quant.requantize(acc, shift) else acc
    if (relu) y.max(0) else y
  }

  def matmul(a: Mat, b: Mat, requant: Boolean = false, relu: Boolean = false, shift: Int = 0,
             accWidth: Int = 32): Mat =
    matmulAcc(a, b, accWidth).map(_.map(postProcess(_, requant, relu, shift)))

  /** Random rows x cols matrix with elements in [lo, hi] (int8 by default). */
  def randomMatrix(rnd: Random, rows: Int, cols: Int, lo: Int = -128, hi: Int = 127): Mat =
    Seq.fill(rows, cols)((lo + rnd.nextInt(hi - lo + 1)).toLong)
}
