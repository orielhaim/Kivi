//! Node registry: admitted cluster members and their lifecycle.
//!
//! Node identity is authenticated from the replicated registry, not from
//! static command-line trust lists. Admission becomes effective only after
//! a replicated control-plane commit.

use std::net::SocketAddr;

use kivi_types::NodeId;

/// Lifecycle of one cluster node.
///
/// Liveness distinguishes a short transient miss (`Suspect`) from a
/// repair-worthy outage (`Unavailable`): a few seconds of packet loss
/// must not trigger hundreds of migrations. Only `Unavailable` (and
/// `Draining`) nodes lose desired replicas through automatic repair.
///
/// ```text
/// Joining -> Active <-> Suspect -> Unavailable -> Active (return)
/// Active -> Draining -> Drained -> Removed
/// Unavailable -> Draining (operator drain of a dead node)
/// Any non-Removed -> Removed (operator removal)
/// Drained -> Active (re-admit)
/// Unavailable -> Active (return, fenced: old memberships never resurrect)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeState {
    /// Admitted by the control plane but not yet serving data.
    Joining,
    /// Serving data and eligible for new placements.
    Active,
    /// Transient heartbeat misses; still serves and keeps placements.
    /// Never triggers repair by itself (anti-flap).
    Suspect,
    /// Repair-worthy outage (suspicion + grace both expired). Excluded
    /// from new placements; automatic repair migrates replicas away.
    /// A returning node transitions back to `Active` but never resurrects
    /// obsolete memberships (generation/tombstone fencing at open).
    Unavailable,
    /// Excluded from new placements; existing replicas migrate away.
    Draining,
    /// No desired or actual data replicas remain; safe to remove.
    Drained,
    /// Tombstoned after removal; fences stale control actions.
    Removed,
}

impl NodeState {
    /// Whether the node may receive new replica placements.
    #[must_use]
    pub const fn accepts_new_replicas(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the node is expected to serve data traffic.
    #[must_use]
    pub const fn serves_data(self) -> bool {
        matches!(self, Self::Active | Self::Draining | Self::Suspect)
    }

    /// Whether the node is placement-eligible (healthy) for repair targets.
    #[must_use]
    pub const fn is_repair_target(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the node's replicas need replacement (repair source).
    #[must_use]
    pub const fn needs_repair(self) -> bool {
        matches!(self, Self::Unavailable | Self::Draining)
    }

    /// Whether drain has fully completed.
    #[must_use]
    pub const fn is_drained(self) -> bool {
        matches!(self, Self::Drained)
    }

    /// Encodes the state as a canonical discriminant byte.
    ///
    /// Discriminants `0..=4` are frozen (Joining/Active/Draining/Drained/
    /// Removed); `Suspect = 5` and `Unavailable = 6` extend the encoding.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Joining => 0,
            Self::Active => 1,
            Self::Draining => 2,
            Self::Drained => 3,
            Self::Removed => 4,
            Self::Suspect => 5,
            Self::Unavailable => 6,
        }
    }

    /// Decodes a canonical discriminant byte.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError::BadState`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, NodeError> {
        match byte {
            0 => Ok(Self::Joining),
            1 => Ok(Self::Active),
            2 => Ok(Self::Draining),
            3 => Ok(Self::Drained),
            4 => Ok(Self::Removed),
            5 => Ok(Self::Suspect),
            6 => Ok(Self::Unavailable),
            _ => Err(NodeError::BadState { found: byte }),
        }
    }
}

/// Why a node record was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeError {
    /// Unknown node-state discriminant in canonical bytes.
    #[error("unknown node state discriminant {found}")]
    BadState {
        /// Observed discriminant.
        found: u8,
    },
    /// Truncated canonical input.
    #[error("truncated node record")]
    Truncated,
    /// Declared length exceeds the available input.
    #[error("declared length {len} exceeds input {available}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Trailing bytes after the framed record.
    #[error("trailing bytes in node record")]
    TrailingBytes,
    /// Endpoint string is not a valid socket address.
    #[error("invalid endpoint {endpoint:?}: {detail}")]
    BadEndpoint {
        /// Offending endpoint string.
        endpoint: String,
        /// Human-readable cause.
        detail: String,
    },
}

/// One admitted cluster node.
///
/// Endpoints are dial info; the [`NodeId`] is the stable identity that
/// survives restarts. The certificate fingerprint is the BLAKE3 of the
/// node's peer-TLS certificate DER: peers authenticate a dynamically
/// joining node by comparing its presented certificate against this
/// registry-committed pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRecord {
    /// Stable process identity.
    pub node: NodeId,
    /// Consensus (H3 peer) endpoint.
    pub peer: SocketAddr,
    /// Native client endpoint (redirect target).
    pub native: SocketAddr,
    /// Admin endpoint.
    pub admin: SocketAddr,
    /// BLAKE3 of the peer-TLS certificate DER.
    pub cert_fingerprint: [u8; 32],
    /// Failure-domain label (zone/rack/host-group); empty when unknown.
    pub failure_domain: String,
    /// Capacity weight for placement balancing (1 = default).
    pub weight: u32,
    /// Lifecycle state.
    pub state: NodeState,
}

impl NodeRecord {
    /// Builds a joining node record with default weight.
    #[must_use]
    pub const fn joining(
        node: NodeId,
        peer: SocketAddr,
        native: SocketAddr,
        admin: SocketAddr,
        cert_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            node,
            peer,
            native,
            admin,
            cert_fingerprint,
            failure_domain: String::new(),
            weight: 1,
            state: NodeState::Joining,
        }
    }

    /// Encodes the canonical bytes.
    ///
    /// Layout (all little-endian):
    ///
    /// ```text
    /// node u64, state u8,
    /// peer_len u32 + peer[.], native_len u32 + native[.],
    /// admin_len u32 + admin[.], fingerprint [u8; 32],
    /// domain_len u32 + domain[.], weight u32
    /// ```
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let peer = self.peer.to_string();
        let native = self.native.to_string();
        let admin = self.admin.to_string();
        let mut out = Vec::with_capacity(
            8 + 1
                + 4
                + peer.len()
                + 4
                + native.len()
                + 4
                + admin.len()
                + 32
                + 4
                + self.failure_domain.len()
                + 4,
        );
        out.extend_from_slice(&self.node.as_u64().to_le_bytes());
        out.push(self.state.encode_byte());
        push_str(&mut out, &peer);
        push_str(&mut out, &native);
        push_str(&mut out, &admin);
        out.extend_from_slice(&self.cert_fingerprint);
        push_str(&mut out, &self.failure_domain);
        out.extend_from_slice(&self.weight.to_le_bytes());
        out
    }

    /// Decodes exactly one record.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError`] on truncation, oversize declarations,
    /// unknown state discriminants, invalid endpoints, or trailing bytes.
    pub fn decode_exact(input: &[u8]) -> Result<Self, NodeError> {
        use NodeError as Fault;
        if input.len() < 8 + 1 + 32 + 4 {
            return Err(Fault::Truncated);
        }
        let node = NodeId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let state = NodeState::decode_byte(input[8])?;
        let mut rest = &input[9..];
        let (peer_str, tail) = take_str(rest)?;
        rest = tail;
        let (native_str, tail) = take_str(rest)?;
        rest = tail;
        let (admin_str, tail) = take_str(rest)?;
        rest = tail;
        if rest.len() < 32 + 4 {
            return Err(Fault::Truncated);
        }
        let mut fingerprint = [0u8; 32];
        fingerprint.copy_from_slice(&rest[..32]);
        rest = &rest[32..];
        let (domain, tail) = take_str(rest)?;
        rest = tail;
        if rest.len() < 4 {
            return Err(Fault::Truncated);
        }
        let weight = u32::from_le_bytes(rest[..4].try_into().unwrap_or([0; 4]));
        rest = &rest[4..];
        if !rest.is_empty() {
            return Err(Fault::TrailingBytes);
        }
        let peer = peer_str
            .parse::<SocketAddr>()
            .map_err(|error| Fault::BadEndpoint {
                endpoint: peer_str.clone(),
                detail: error.to_string(),
            })?;
        let native = native_str
            .parse::<SocketAddr>()
            .map_err(|error| Fault::BadEndpoint {
                endpoint: native_str.clone(),
                detail: error.to_string(),
            })?;
        let admin = admin_str
            .parse::<SocketAddr>()
            .map_err(|error| Fault::BadEndpoint {
                endpoint: admin_str.clone(),
                detail: error.to_string(),
            })?;
        Ok(Self {
            node,
            peer,
            native,
            admin,
            cert_fingerprint: fingerprint,
            failure_domain: domain,
            weight: weight.max(1),
            state,
        })
    }
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn take_str(input: &[u8]) -> Result<(String, &[u8]), NodeError> {
    use NodeError as Fault;
    if input.len() < 4 {
        return Err(Fault::Truncated);
    }
    let len = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4])) as usize;
    if input.len() < 4 + len {
        return Err(Fault::Oversized {
            len,
            available: input.len(),
        });
    }
    let bytes = &input[4..4 + len];
    let value = String::from_utf8(bytes.to_vec()).map_err(|error| Fault::BadEndpoint {
        endpoint: format!("{bytes:?}"),
        detail: error.to_string(),
    })?;
    Ok((value, &input[4 + len..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> NodeRecord {
        NodeRecord {
            node: NodeId::from_u64(4),
            peer: "127.0.0.1:9144".parse().expect("peer"),
            native: "127.0.0.1:9044".parse().expect("native"),
            admin: "127.0.0.1:19444".parse().expect("admin"),
            cert_fingerprint: [7u8; 32],
            failure_domain: "zone-a".to_owned(),
            weight: 2,
            state: NodeState::Active,
        }
    }

    #[test]
    fn node_record_round_trips() {
        let record = sample();
        let back = NodeRecord::decode_exact(&record.encode_to_vec()).expect("decodes");
        assert_eq!(back, record);
        assert!(back.state.serves_data());
        assert!(back.state.accepts_new_replicas());
    }

    #[test]
    fn lifecycle_predicates_are_exclusive() {
        assert!(!NodeState::Joining.accepts_new_replicas());
        assert!(!NodeState::Draining.accepts_new_replicas());
        assert!(!NodeState::Suspect.accepts_new_replicas());
        assert!(!NodeState::Unavailable.accepts_new_replicas());
        assert!(NodeState::Draining.serves_data());
        assert!(NodeState::Suspect.serves_data());
        assert!(!NodeState::Unavailable.serves_data());
        assert!(!NodeState::Drained.serves_data());
        assert!(NodeState::Drained.is_drained());
        assert!(!NodeState::Active.is_drained());
        assert!(NodeState::Active.is_repair_target());
        assert!(!NodeState::Suspect.is_repair_target());
        assert!(NodeState::Unavailable.needs_repair());
        assert!(NodeState::Draining.needs_repair());
        assert!(!NodeState::Suspect.needs_repair());
    }

    #[test]
    fn corrupt_state_byte_fails() {
        assert!(matches!(
            NodeState::decode_byte(9),
            Err(NodeError::BadState { .. })
        ));
        let mut bad = sample().encode_to_vec();
        bad[8] = 9;
        assert!(NodeRecord::decode_exact(&bad).is_err());
    }
}
