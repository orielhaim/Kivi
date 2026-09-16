//! Namespace descriptors: the physical layout of a keyspace (RFC §7–§9).
//!
//! A [`NamespaceDescriptor`] binds a [`NamespaceId`] to one
//! [`NamespaceLayout`]:
//!
//! ```text
//! NamespaceLayout::Hash    → PartitionHasher → HashRange → tablet
//! NamespaceLayout::Ordered → raw logical key → OrderedRange → tablet
//! ```
//!
//! Hash preserves the point-lookup-friendly partitioning model; Ordered
//! partitions directly over logical key bytes for range scans and secondary
//! indexes. They are distinct placement semantics, never a compatibility
//! flag on each other.

use core::fmt;

use kivi_types::NamespaceId;

use crate::range::PartitionKind;

/// Physical layout of one namespace's keyspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamespaceLayout {
    /// Point-lookup-friendly hash partitioning (`key → PartitionHash`).
    Hash,
    /// Ordered key-byte partitioning (`[start, end)` ranges).
    Ordered,
}

impl NamespaceLayout {
    /// The partition kind this layout routes with.
    #[must_use]
    pub const fn kind(self) -> PartitionKind {
        match self {
            Self::Hash => PartitionKind::Hash,
            Self::Ordered => PartitionKind::Ordered,
        }
    }
}

impl fmt::Display for NamespaceLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hash => write!(f, "hash"),
            Self::Ordered => write!(f, "ordered"),
        }
    }
}

/// Kivi-owned namespace descriptor: identity plus physical layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamespaceDescriptor {
    /// Namespace identity.
    pub id: NamespaceId,
    /// Physical layout of the keyspace.
    pub layout: NamespaceLayout,
}

impl NamespaceDescriptor {
    /// Builds a descriptor (total: every id/layout pair is valid).
    #[must_use]
    pub const fn new(id: NamespaceId, layout: NamespaceLayout) -> Self {
        Self { id, layout }
    }
}

impl fmt::Display for NamespaceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "namespace {} ({})", self.id.as_u64(), self.layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_map_to_partition_kinds() {
        assert_eq!(NamespaceLayout::Hash.kind(), PartitionKind::Hash);
        assert_eq!(NamespaceLayout::Ordered.kind(), PartitionKind::Ordered);
        assert_eq!(NamespaceLayout::Hash.to_string(), "hash");
        assert_eq!(NamespaceLayout::Ordered.to_string(), "ordered");
    }

    #[test]
    fn descriptor_displays_identity_and_layout() {
        let descriptor =
            NamespaceDescriptor::new(NamespaceId::from_u64(9), NamespaceLayout::Ordered);
        assert_eq!(descriptor.to_string(), "namespace 9 (ordered)");
    }
}
