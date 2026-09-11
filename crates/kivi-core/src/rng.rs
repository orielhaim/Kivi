//! Deterministic randomness vocabulary.
//!
//! Simulation and protocol logic never draws from ambient randomness
//! (`thread_rng`, OS entropy, system time). Randomness arrives explicitly as
//! a [`RandomSource`]: production threads a hardware/OS source through this
//! shape, while simulation supplies a reproducible generator (see `kivi-sim`)
//! so the same seed replays the same decisions bit-for-bit.
//!
//! The provided helpers pin their exact derivation as part of the stability
//! contract: policies built atop any source behave identically given
//! identical `next_u64` streams.

/// Source of non-cryptographic randomness for protocols and simulation.
///
/// Object-safe, so heterogeneous policies can share one source as
/// `&mut dyn RandomSource`. Cryptographic randomness is conceptually
/// separate and never flows through this trait.
pub trait RandomSource {
    /// Returns the next 64 raw bits.
    fn next_u64(&mut self) -> u64;

    /// Returns 32 bits derived as the high half of [`next_u64`](Self::next_u64).
    fn next_u32(&mut self) -> u32 {
        u32::try_from(self.next_u64() >> 32).unwrap_or(u32::MAX)
    }

    /// Returns 128 bits derived as two consecutive [`next_u64`](Self::next_u64)
    /// draws, high half first.
    fn next_u128(&mut self) -> u128 {
        (u128::from(self.next_u64()) << 64) | u128::from(self.next_u64())
    }

    /// Returns a value in `0..upper` with remainder semantics over one
    /// [`next_u64`](Self::next_u64) draw.
    ///
    /// Remainder (not unbiased) sampling is specified deliberately: what
    /// matters for replay is that the mapping is frozen, not that it is
    /// uniform. Consumers needing unbiased sampling build it atop this and
    /// pin their own vectors.
    ///
    /// # Panics
    ///
    /// Panics if `upper` is zero (remainder by zero).
    fn below_u64(&mut self, upper: u64) -> u64 {
        self.next_u64() % upper
    }

    /// Fills `buf` from consecutive [`next_u64`](Self::next_u64) draws in
    /// little-endian byte order (a trailing partial chunk takes the low
    /// bytes of its draw).
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}
