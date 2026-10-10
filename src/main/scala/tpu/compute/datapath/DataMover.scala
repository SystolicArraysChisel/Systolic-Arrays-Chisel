// package tpu.compute.datapath

/**
  * DataMover can't never do backpressure (wOut and aOut have
  * protocol V), so rdResp.ready must always be true.
  */