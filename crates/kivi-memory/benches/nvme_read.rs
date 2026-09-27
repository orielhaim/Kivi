//! The demotion-store read path, measured against the strategy it replaced.
//!
//! `NvmeProvider::read_at` used to open the whole demotion store and copy all
//! of it to answer one record. That is O(store) per promotion, so the fifth
//! promotion of a 1 KiB object copied as much as the first promotion of a
//! 1 GiB one. It now reads the record's 20-byte header, then its payload, at
//! the record's own offset: O(record) per promotion.
//!
//! Comparing two strategies inside one binary is the only way to get an honest
//! ratio, because the alternative — replaying a whole benchmark matrix — cannot
//! separate this change from the machine's disk state on the day.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p kivi-memory --bench nvme_read
//! ```

use std::hint::black_box;
use std::path::Path;

use divan::Bencher;
use kivi_memory::nvme::{NvmeOptions, NvmeProvider};

/// Builds a store of `count` records of `payload` bytes and returns the provider
/// plus every record's offset and length.
fn store(dir: &Path, payload: usize, count: usize) -> (NvmeProvider, Vec<(u64, u64)>) {
    let mut provider = NvmeProvider::open(dir, NvmeOptions::default()).expect("open");
    let body: Vec<u8> = (0..payload)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect();
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        let (offset, len, _checksum) = provider.append_relaxed(black_box(&body)).expect("append");
        offsets.push((offset, len));
    }
    (provider, offsets)
}

/// One positioned record read: the strategy in the code today.
///
/// The record count is the axis the old cost grew along, because the old cost
/// was the whole store. Holding the payload fixed at Kivi's 1 KiB medium value
/// and growing the count is what separates O(record) from O(store).
#[divan::bench(args = [512_usize, 2048, 8192])]
fn positioned_read(b: Bencher, count: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (provider, offsets) = store(dir.path(), PAYLOAD, count);
    b.bench(|| {
        for (offset, len) in &offsets {
            black_box(provider.read_at(black_box(*offset), black_box(*len)).ok());
        }
    });
}

/// The whole-file read the positioned read replaced.
///
/// Same store, same count, so the only difference is how many bytes each
/// promotion copies.
#[divan::bench(args = [512_usize, 2048, 8192])]
fn whole_file_read(b: Bencher, count: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_provider, _offsets) = store(dir.path(), PAYLOAD, count);
    let data = dir.path().join("materializations.dat");
    b.bench(|| {
        for _ in 0..count {
            black_box(std::fs::read(black_box(&data)).ok());
        }
    });
}

/// Kivi's medium value size. Fixed so the only variable is the store length.
const PAYLOAD: usize = 1024;

fn main() {
    divan::main();
}
