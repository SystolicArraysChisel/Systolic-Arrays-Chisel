package tpu.ref

/** Quantization, the Scala copy of host/tpu_host/quant.py.
  *
  * Symmetric scheme, one scale per matrix: x ~= scale * q, q in [-127, 127].
  * Every rounding is round-half-up (ties toward +infinity: 2.5 -> 3,
  * -2.5 -> -2), as in the hardware requantization. ref_vectors.txt checks
  * that this file and the Python one agree bit for bit.
  */
object Quant {
  val QMax = 127

  /** floor(v + 0.5), computed without the rounding error of v + 0.5. */
  def roundHalfUp(v: Double): Double = {
    val f = math.floor(v)
    if (v - f >= 0.5) f + 1.0 else f
  }

  /** Scale that maps the largest |x| to 127 (1.0 for an all-zero matrix). */
  def chooseScale(xs: Iterable[Double]): Double = {
    val m = xs.map(math.abs).maxOption.getOrElse(0.0)
    if (m > 0) m / QMax else 1.0
  }

  def quantize(x: Double, scale: Double): Int =
    roundHalfUp(x / scale).max(-QMax).min(QMax).toInt

  def dequantize(q: Long, scale: Double): Double = q * scale

  /** (acc + 2^(shift-1)) >> shift, arithmetic; no rounding term if shift = 0. */
  def roundingShift(acc: Long, shift: Int): Long =
    if (shift == 0) acc else (acc + (1L << (shift - 1))) >> shift

  /** Hardware requantization: rounding shift, then saturation to [-127, 127]. */
  def requantize(acc: Long, shift: Int): Long =
    roundingShift(acc, shift).max(-QMax).min(QMax)

  /** Smallest shift for which no accumulator saturates. */
  def chooseShift(accs: Iterable[Long], accWidth: Int = 32): Int =
    (0 until accWidth)
      .find(s => accs.forall(a => math.abs(roundingShift(a, s)) <= QMax))
      .getOrElse(accWidth - 1)

  /** Scale of the int8 results of a requantized C = A . B. */
  def requantScale(scaleA: Double, scaleB: Double, shift: Int): Double =
    scaleA * scaleB * math.pow(2, shift)
}
