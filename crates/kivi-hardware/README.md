# kivi-hardware

Physical machine discovery, capability snapshot, and worker placement.

This crate is Kivi's only window onto the machine. It models the physical
topology, records which optional accelerators exist, and decides where Kivi's
owner threads run. It holds no database policy: the crate describes the
machine, and `kivi-memory` and `kivi-control` decide what to do about it.

No accelerator here can change what Kivi believes. Placement, SIMD width,
performance counters, huge pages, direct I/O, CXL memory, RDMA, DSA and SPDK
all change the cost of an operation and none of them change its result. A
machine with none of them runs the same exact-state semantics, more slowly.
