package tpu.compute.array

object ArrayTestUtil {

  // shortcut: local matmul, swap for the shared reference model once #7 lands.
  def matmul(a: Seq[Seq[Int]], b: Seq[Seq[Int]]): Seq[Seq[BigInt]] =
    a.map(row =>
      b.head.indices.map(j => BigInt(row.indices.map(r => row(r) * b(r)(j)).sum))
    )
}
