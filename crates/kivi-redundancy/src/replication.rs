//! Replication provider: `N` verified copies behind the common abstraction.
//!
//! Replication uses the same asset identity, fragment identity, placement,
//! publication (`build → verify → publish → retire`), reconstruction,
//! repair, and observability as erasure coding. A replica fragment is one
//! full copy of the asset bytes; verification is the asset content hash,
//! never a byte count.

use crate::{RedundancyError, ReplicationParams};

/// Encodes asset bytes into `copies` identical replica fragments.
///
/// Pure: clones bytes per copy (callers stream when assets grow; the bound
/// is [`ReplicationParams::MAX_COPIES`] copies of at most
/// [`crate::RsParams::MAX_ASSET_BYTES`] each in practice).
///
/// # Errors
///
/// Returns [`RedundancyError::InvalidParams`] for out-of-range copies.
pub fn encode(bytes: &[u8], params: ReplicationParams) -> Result<Vec<Vec<u8>>, RedundancyError> {
    if params.copies == 0 || params.copies > ReplicationParams::MAX_COPIES {
        return Err(RedundancyError::invalid_params(format!(
            "replication copies {} outside 1..={}",
            params.copies,
            ReplicationParams::MAX_COPIES
        )));
    }
    Ok((0..params.copies).map(|_| bytes.to_vec()).collect())
}

/// Selects the first present replica's bytes.
///
/// Any single verified copy suffices; the caller verifies the bytes against
/// the asset id before exposing them.
///
/// # Errors
///
/// Returns [`RedundancyError::Unrecoverable`] when no copy is present.
pub fn decode(present: &[(u32, Vec<u8>)]) -> Result<Vec<u8>, RedundancyError> {
    present
        .first()
        .map(|(_, bytes)| bytes.clone())
        .ok_or_else(|| RedundancyError::Unrecoverable {
            detail: "replication needs one copy, none present".to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicas_are_identical_and_one_suffices() {
        let bytes = b"replicated information".to_vec();
        let shards = encode(&bytes, ReplicationParams { copies: 3 }).expect("encodes");
        assert_eq!(shards.len(), 3);
        for shard in &shards {
            assert_eq!(shard, &bytes);
        }
        let present = vec![(2u32, shards[2].clone())];
        assert_eq!(decode(&present).expect("decodes"), bytes);
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn bounds_fail_closed() {
        assert!(encode(b"x", ReplicationParams { copies: 0 }).is_err());
        assert!(encode(b"x", ReplicationParams { copies: 9 }).is_err());
    }
}
