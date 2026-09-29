//! Every value-size band the write path claims to own must round-trip.
//!
//! The write path routes a value by size: at or below `FABRIC_INLINE_MAX` it
//! stays inline, at or below the chunk lane's inline threshold it stages into
//! the Memory Fabric, and above that it becomes a chunked root. A band that
//! refuses its own values is a hole in the write path, not a capacity limit:
//! this pins the boundaries so a regression in one band cannot hide behind
//! the others working.

mod harness;

use bytes::Bytes;
use harness::start_single;

/// The size bands the write path routes between, named by the constant that
/// decides them.
#[derive(Debug, Clone, Copy)]
struct Band {
    label: &'static str,
    size: usize,
}

fn bands() -> Vec<Band> {
    let inline = kivi_engine::FABRIC_INLINE_MAX;
    let chunk_inline = usize::try_from(kivi_engine::ChunkFabricConfig::default().inline_threshold)
        .expect("inline threshold fits usize");
    vec![
        Band {
            label: "tiny",
            size: 16,
        },
        Band {
            label: "at-inline-boundary",
            size: inline,
        },
        Band {
            label: "just-above-inline",
            size: inline + 1,
        },
        Band {
            label: "fabric-2kib",
            size: 2048,
        },
        Band {
            label: "fabric-16kib",
            size: 16 * 1024,
        },
        Band {
            label: "fabric-64kib",
            size: 64 * 1024,
        },
        Band {
            label: "at-chunk-inline-boundary",
            size: chunk_inline,
        },
        Band {
            label: "just-above-chunk-inline",
            size: chunk_inline + 1,
        },
        Band {
            label: "chunked-1mib",
            size: 1024 * 1024,
        },
    ]
}

#[test]
fn a_value_in_every_band_round_trips_exactly() {
    let engine = start_single(kivi_engine::FabricConfig::default());
    for band in bands() {
        let key = kivi_state::Key::from(format!("readback:{}", band.size));
        let mut payload = vec![b'x'; band.size];
        // Distinguish "returned the right bytes" from "returned a buffer of
        // the right length".
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte = u8::try_from(index % 251).unwrap_or(0);
        }
        let value = Bytes::from(payload.clone());
        engine.client().set(&key, value).unwrap_or_else(|error| {
            panic!(
                "SET of {} bytes ({}) failed: {error:?}",
                band.size, band.label
            )
        });
        let read = engine.client().get(&key).unwrap_or_else(|error| {
            panic!(
                "GET of {} bytes ({}) failed: {error:?}",
                band.size, band.label
            )
        });
        assert_eq!(
            read.as_deref(),
            Some(payload.as_slice()),
            "round trip differs in the {} band ({} bytes)",
            band.label,
            band.size
        );
    }
}

/// A bounded hot arena is a cache, not a capacity limit on the database: a
/// saturated arena must take the off-core tier, not refuse the write.
#[test]
fn a_saturated_arena_tiers_to_offcore_instead_of_refusing_writes() {
    const ARENA: u64 = 1024 * 1024;
    const DEMOTION: u64 = 256 * 1024 * 1024;
    const VALUE: usize = 16 * 1024;

    let engine = start_single(kivi_engine::FabricConfig {
        arena_bytes_per_worker: ARENA,
        demotion_bytes_per_worker: DEMOTION,
    });

    // Comfortably past the arena bound: 8 MiB of live medium values into a
    // 1 MiB arena.
    let total = (8 * 1024 * 1024) / VALUE;
    for index in 0..total {
        let key = kivi_state::Key::from(format!("tier:{index}"));
        let value = Bytes::from(vec![u8::try_from(index % 251).unwrap_or(0); VALUE]);
        engine.client().set(&key, value).unwrap_or_else(|error| {
            panic!("write {index} of {total} failed once the arena filled: {error:?}")
        });
    }

    // Every value must still read back, whichever tier took it.
    for index in 0..total {
        let key = kivi_state::Key::from(format!("tier:{index}"));
        let expected = vec![u8::try_from(index % 251).unwrap_or(0); VALUE];
        let read = engine
            .client()
            .get(&key)
            .unwrap_or_else(|error| panic!("read {index} of {total} failed: {error:?}"));
        assert_eq!(
            read.as_deref(),
            Some(expected.as_slice()),
            "value {index} did not survive the tier switch"
        );
    }
}
