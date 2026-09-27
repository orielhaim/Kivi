//! Proves the direct-I/O path on the volume named as the first argument.
//!
//! Exists because "the tests passed" is not evidence that `O_DIRECT` was ever
//! used: on `tmpfs`, `overlayfs` and a 9p share the open is refused, the
//! handle silently resolves to buffered, and every assertion still passes. This
//! prints what actually happened.

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dcheck <file>");
        std::process::exit(2);
    };
    let path = std::path::Path::new(&path);
    std::fs::write(
        path,
        (0..64 * 1024_u32)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<u8>>(),
    )
    .expect("write");
    println!(
        "logical_block_size = {:?}",
        kivi_hardware::io::logical_block_size(path)
    );
    let mut file =
        kivi_hardware::io::DirectFile::open(path, kivi_hardware::io::DirectIoPolicy::Always, 8192)
            .expect("open");
    println!(
        "mode = {} unbuffered = {} align = {} len = {}",
        file.mode(),
        file.is_unbuffered(),
        file.alignment(),
        file.len()
    );
    let mut transfer = kivi_hardware::io::AlignedTransfer::new(16 * 1024, file.alignment());
    for (offset, len) in [(0_u64, 4096_usize), (1, 100), (4095, 2), (8192, 8192)] {
        let read = transfer
            .read_at(&mut file, offset, len)
            .expect("aligned read");
        println!(
            "offset {offset} len {len} -> {read} bytes, first={:?} last={:?}",
            &transfer.bytes()[..1.min(transfer.bytes().len())],
            transfer.bytes().last()
        );
    }
}
