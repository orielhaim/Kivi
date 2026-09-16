//! Namespace and secondary-index catalogs: the control plane's record of
//! what keyspaces exist, how they are laid out, and which indexes feed
//! off them.
//!
//! Namespaces are the main semantic boundary (one layout each: hash for
//! point-lookup partitioning, ordered for range scans and indexes).
//! Indexes are ordered representations of primary data maintained
//! transactionally — the control plane owns their definitions and
//! lifecycle states, never their entries (entries are ordinary data-plane
//! state in ordered namespaces).

use kivi_types::NamespaceId;

/// Physical layout of one namespace's keyspace (mirrors
/// `kivi_tablet::NamespaceLayout` without depending on it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CatalogLayout {
    /// Point-lookup-friendly hash partitioning.
    Hash,
    /// Ordered key-byte partitioning (range scans, secondary indexes).
    Ordered,
}

impl CatalogLayout {
    /// Canonical encoding byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Hash => 0,
            Self::Ordered => 1,
        }
    }

    /// Decodes a canonical layout byte.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::BadLayout`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, CatalogError> {
        match byte {
            0 => Ok(Self::Hash),
            1 => Ok(Self::Ordered),
            _ => Err(CatalogError::BadLayout { found: byte }),
        }
    }
}

/// One registered namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamespaceRecord {
    /// Namespace identity.
    pub id: NamespaceId,
    /// Physical layout.
    pub layout: CatalogLayout,
}

impl NamespaceRecord {
    /// Encodes the canonical bytes (`id u64, layout u8`).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9);
        out.extend_from_slice(&self.id.as_u64().to_le_bytes());
        out.push(self.layout.encode_byte());
        out
    }

    /// Decodes exactly one record.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] on truncation or unknown layouts.
    pub fn decode_exact(input: &[u8]) -> Result<Self, CatalogError> {
        if input.len() != 9 {
            return Err(CatalogError::Truncated);
        }
        Ok(Self {
            id: NamespaceId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8]))),
            layout: CatalogLayout::decode_byte(input[8])?,
        })
    }
}

/// Secondary index uniqueness contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CatalogIndexKind {
    /// Many primaries per term.
    NonUnique,
    /// One primary per term (concurrent claims conflict).
    Unique,
}

impl CatalogIndexKind {
    /// Canonical encoding byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::NonUnique => 0,
            Self::Unique => 1,
        }
    }

    /// Decodes a canonical kind byte.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::BadIndexKind`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, CatalogError> {
        match byte {
            0 => Ok(Self::NonUnique),
            1 => Ok(Self::Unique),
            _ => Err(CatalogError::BadIndexKind { found: byte }),
        }
    }
}

/// Secondary index lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CatalogIndexState {
    /// Accepting maintenance writes, invisible to queries.
    Building,
    /// Maintained and queryable.
    Ready,
    /// Maintenance stopped, entries being removed.
    Dropping,
}

impl CatalogIndexState {
    /// Canonical encoding byte.
    #[must_use]
    pub const fn encode_byte(self) -> u8 {
        match self {
            Self::Building => 0,
            Self::Ready => 1,
            Self::Dropping => 2,
        }
    }

    /// Decodes a canonical state byte.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::BadIndexState`] for unknown discriminants.
    pub const fn decode_byte(byte: u8) -> Result<Self, CatalogError> {
        match byte {
            0 => Ok(Self::Building),
            1 => Ok(Self::Ready),
            2 => Ok(Self::Dropping),
            _ => Err(CatalogError::BadIndexState { found: byte }),
        }
    }
}

/// One secondary index definition: which primary namespace it feeds off
/// and how. Entries live in the primary's ordered namespace under the
/// reserved index prefix (V1: same-namespace indexes; hash namespaces
/// cannot host index entries).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexRecord {
    /// Index identity.
    pub id: u64,
    /// Primary namespace.
    pub primary: NamespaceId,
    /// Uniqueness contract.
    pub kind: CatalogIndexKind,
    /// Lifecycle state.
    pub state: CatalogIndexState,
}

impl IndexRecord {
    /// Encodes the canonical bytes (`id u64, primary u64, kind u8, state
    /// u8`).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(18);
        out.extend_from_slice(&self.id.to_le_bytes());
        out.extend_from_slice(&self.primary.as_u64().to_le_bytes());
        out.push(self.kind.encode_byte());
        out.push(self.state.encode_byte());
        out
    }

    /// Decodes exactly one record.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] on truncation or unknown discriminants.
    pub fn decode_exact(input: &[u8]) -> Result<Self, CatalogError> {
        if input.len() != 18 {
            return Err(CatalogError::Truncated);
        }
        Ok(Self {
            id: u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])),
            primary: NamespaceId::from_u64(u64::from_le_bytes(
                input[8..16].try_into().unwrap_or([0; 8]),
            )),
            kind: CatalogIndexKind::decode_byte(input[16])?,
            state: CatalogIndexState::decode_byte(input[17])?,
        })
    }
}

/// Catalog decode failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CatalogError {
    /// Truncated input.
    #[error("truncated catalog record")]
    Truncated,
    /// Unknown layout discriminant.
    #[error("unknown namespace layout {found}")]
    BadLayout {
        /// Observed discriminant.
        found: u8,
    },
    /// Unknown index-kind discriminant.
    #[error("unknown index kind {found}")]
    BadIndexKind {
        /// Observed discriminant.
        found: u8,
    },
    /// Unknown index-state discriminant.
    #[error("unknown index state {found}")]
    BadIndexState {
        /// Observed discriminant.
        found: u8,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let ns = NamespaceRecord {
            id: NamespaceId::from_u64(7),
            layout: CatalogLayout::Ordered,
        };
        assert_eq!(NamespaceRecord::decode_exact(&ns.encode_to_vec()), Ok(ns));
        let index = IndexRecord {
            id: 3,
            primary: NamespaceId::from_u64(7),
            kind: CatalogIndexKind::Unique,
            state: CatalogIndexState::Ready,
        };
        assert_eq!(IndexRecord::decode_exact(&index.encode_to_vec()), Ok(index));
        assert!(CatalogLayout::decode_byte(9).is_err());
        assert!(CatalogIndexKind::decode_byte(9).is_err());
        assert!(CatalogIndexState::decode_byte(9).is_err());
    }
}
