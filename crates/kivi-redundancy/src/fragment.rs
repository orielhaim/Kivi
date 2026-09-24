//! Fragment identity and durable metadata: project-owned, versioned, deterministic.
//!
//! A [`FragmentId`] binds source asset identity, redundancy generation and
//! layout, fragment index/role, encoding parameters where required, and the
//! fragment's own content/integrity identity. Filesystem paths and storage
//! offsets never enter the identity: physical locations are replaceable.
//!
//! Durable layout metadata is versioned little-endian binary (magic, version,
//! lengths, CRC32C — no serde/bincode/rkyv near durable bytes, mirroring the
//! chunk and checkpoint crates). This is pre-1.0: unknown versions are
//! rejected outright, never migrated.

use kivi_codec::integrity::{blake3_256, crc32c_checksum};

use crate::{AssetId, AssetKind, LocalityRequirement, RedundancyError, RsParams, SchemeParams};

/// Magic word: ASCII `"KVRF"` read as little-endian `u32`.
pub const LAYOUT_MAGIC: u32 = 0x4652_564B;
/// Layout major version minted here.
pub const LAYOUT_MAJOR: u16 = 1;
/// Layout minor version minted here.
pub const LAYOUT_MINOR: u16 = 1;
/// Encoded layout header length: magic u32, major u16, minor u16,
/// kind u8, scheme u8, domain u64, asset hash [32], `logical_len` u64,
/// generation u64, data u8, parity u8, copies u8, reserved u8,
/// `fragment_len` u32, `fragment_count` u32, header CRC u32.
pub const LAYOUT_HEADER_LEN: usize = 4 + 2 + 2 + 1 + 1 + 8 + 32 + 8 + 8 + 1 + 1 + 1 + 1 + 4 + 4 + 4;
/// Encoded per-fragment record: index u32, role u8, reserved u8,
/// node u64, content hash [32], `stored_len` u64, record CRC u32.
pub const FRAGMENT_RECORD_LEN: usize = 4 + 1 + 1 + 8 + 32 + 8 + 4;
/// Footer: body CRC u32, BLAKE3(header[..len-4] + body) [32], footer CRC u32.
pub const LAYOUT_FOOTER_LEN: usize = 4 + 32 + 4;
/// Contract block magic: ASCII `"KVRC"`.
pub const CONTRACT_MAGIC: u32 = 0x4352_564B;
/// Contract block length: magic, version, reserved, minimum available, CRC.
pub const CONTRACT_HEADER_LEN: usize = 4 + 1 + 1 + 1 + 4 + 2;
const CONTRACT_CRC_LEN: usize = 4;

/// Which role a fragment plays in its layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentRole {
    /// Full copy (replication).
    Replica,
    /// Systematic data shard (erasure coding).
    Data,
    /// Parity shard (erasure coding).
    Parity,
}

impl FragmentRole {
    /// Wraps a raw discriminant.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] for unknown values.
    pub const fn from_u8(value: u8) -> Result<Self, RedundancyError> {
        match value {
            0 => Ok(Self::Replica),
            1 => Ok(Self::Data),
            2 => Ok(Self::Parity),
            _ => Err(RedundancyError::BadLayout {
                detail: String::new(),
            }),
        }
    }

    /// Returns the frozen discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Replica => 0,
            Self::Data => 1,
            Self::Parity => 2,
        }
    }
}

/// Project-owned identity of one redundancy fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentId {
    /// Source asset.
    pub asset: AssetId,
    /// Redundancy generation (layout version for this asset).
    pub generation: u64,
    /// Fragment index (`0..total`).
    pub index: u32,
    /// Role within the layout.
    pub role: FragmentRole,
    /// Encoding parameters hash (0 for replication; `fragment_len` mix for
    /// RS — binds the exact widths that produced these bytes).
    pub params_tag: u32,
    /// Content/integrity identity (BLAKE3 of the fragment bytes).
    pub content: [u8; 32],
}

impl FragmentId {
    /// Derives the identity for fragment bytes under a layout.
    #[must_use]
    pub fn for_bytes(
        asset: AssetId,
        generation: u64,
        index: u32,
        params: SchemeParams,
        bytes: &[u8],
    ) -> Self {
        let role = match params {
            SchemeParams::Replication(_) => FragmentRole::Replica,
            SchemeParams::ReedSolomon(rs) => {
                if u32::from(rs.data) > index {
                    FragmentRole::Data
                } else {
                    FragmentRole::Parity
                }
            }
        };
        let params_tag = match params {
            SchemeParams::Replication(_) => 0,
            SchemeParams::ReedSolomon(rs) => {
                (u32::from(rs.data) << 24) ^ (u32::from(rs.parity) << 16) ^ rs.fragment_len
            }
        };
        Self {
            asset,
            generation,
            index,
            role,
            params_tag,
            content: blake3_256(bytes),
        }
    }

    /// Verifies candidate bytes against the content identity.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::CorruptFragment`] on hash mismatch.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), RedundancyError> {
        if blake3_256(bytes) != self.content {
            return Err(RedundancyError::CorruptFragment {
                detail: format!(
                    "fragment {} gen {} index {} content mismatch",
                    self.asset, self.generation, self.index
                ),
            });
        }
        Ok(())
    }
}

/// One fragment's durable record: identity plus current location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentRecord {
    /// Fragment identity.
    pub id: FragmentId,
    /// Node currently holding it.
    pub node: kivi_types::NodeId,
    /// Stored bytes length (replicas: logical length; RS: `fragment_len`).
    pub stored_len: u64,
}

/// One published layout generation: scheme, fragments, verification root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedundancyLayout {
    /// Asset protected.
    pub asset: AssetId,
    /// Exact logical bytes.
    pub logical_len: u64,
    /// Generation (monotonic per asset, fenced).
    pub generation: u64,
    /// Scheme and bounded parameters.
    pub params: SchemeParams,
    /// Minimum retrievable fragment count promised by the publisher.
    pub min_available: u32,
    /// Placement locality requirement promised by the publisher.
    pub locality: LocalityRequirement,
    /// Failure-domain labels permitted by the publisher.
    pub allowed_domains: Vec<String>,
    /// Fragments in index order.
    pub fragments: Vec<FragmentRecord>,
}

impl RedundancyLayout {
    /// Builds a layout, validating scheme bounds and fragment coherence.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on invalid params or fragment mismatch.
    pub fn new(
        asset: AssetId,
        logical_len: u64,
        generation: u64,
        params: SchemeParams,
        fragments: Vec<FragmentRecord>,
    ) -> Result<Self, RedundancyError> {
        Self::new_with_min_available(asset, logical_len, generation, params, fragments, 1)
    }

    /// Builds a layout with a persisted minimum retrievable-fragment floor.
    #[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
    pub fn new_with_min_available(
        asset: AssetId,
        logical_len: u64,
        generation: u64,
        params: SchemeParams,
        fragments: Vec<FragmentRecord>,
        min_available: u32,
    ) -> Result<Self, RedundancyError> {
        Self::new_with_contract(
            asset,
            logical_len,
            generation,
            params,
            fragments,
            min_available,
            LocalityRequirement::None,
            Vec::new(),
        )
    }

    /// Builds a layout with its complete semantic placement contract.
    #[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
    pub fn new_with_contract(
        asset: AssetId,
        logical_len: u64,
        generation: u64,
        params: SchemeParams,
        fragments: Vec<FragmentRecord>,
        min_available: u32,
        locality: LocalityRequirement,
        mut allowed_domains: Vec<String>,
    ) -> Result<Self, RedundancyError> {
        params.validate()?;
        if min_available == 0 || min_available > params.total_fragments() {
            return Err(RedundancyError::invalid_params(format!(
                "minimum available {min_available} outside 1..={}",
                params.total_fragments()
            )));
        }
        if allowed_domains.len() > u16::MAX as usize || allowed_domains.iter().any(String::is_empty)
        {
            return Err(RedundancyError::invalid_params(
                "allowed domain contract is invalid",
            ));
        }
        allowed_domains.sort_unstable();
        allowed_domains.dedup();
        if generation == 0 {
            return Err(RedundancyError::invalid_asset(
                "generation 0 is the invalid sentinel",
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        let have = fragments.len() as u32;
        if have != params.total_fragments() {
            return Err(RedundancyError::invalid_params(format!(
                "layout holds {} fragments but {:?} needs {}",
                fragments.len(),
                params.scheme(),
                params.total_fragments()
            )));
        }
        for (index, record) in fragments.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let want = index as u32;
            if record.id.index != want
                || record.id.asset != asset
                || record.id.generation != generation
            {
                return Err(RedundancyError::invalid_params(format!(
                    "fragment record {index} misbinds asset/generation/index"
                )));
            }
        }
        Ok(Self {
            asset,
            logical_len,
            generation,
            params,
            min_available,
            locality,
            allowed_domains,
            fragments,
        })
    }

    /// Encodes the deterministic durable bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut header = vec![0u8; LAYOUT_HEADER_LEN];
        header[0..4].copy_from_slice(&LAYOUT_MAGIC.to_le_bytes());
        header[4..6].copy_from_slice(&LAYOUT_MAJOR.to_le_bytes());
        header[6..8].copy_from_slice(&LAYOUT_MINOR.to_le_bytes());
        header[8] = self.asset.kind.as_u8();
        header[9] = self.params.scheme().as_u8();
        header[10..18].copy_from_slice(&self.asset.domain.to_le_bytes());
        header[18..50].copy_from_slice(&self.asset.hash);
        header[50..58].copy_from_slice(&self.logical_len.to_le_bytes());
        header[58..66].copy_from_slice(&self.generation.to_le_bytes());
        let (data, parity, copies, fragment_len) = match self.params {
            SchemeParams::Replication(p) => (0u8, 0u8, p.copies, 0u32),
            SchemeParams::ReedSolomon(p) => (p.data, p.parity, 0u8, p.fragment_len),
        };
        header[66] = data;
        header[67] = parity;
        header[68] = copies;
        header[69] = 0;
        header[70..74].copy_from_slice(&fragment_len.to_le_bytes());
        // Total fragments <=40, far below `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let count = self.fragments.len() as u32;
        header[74..78].copy_from_slice(&count.to_le_bytes());
        let crc = crc32c_checksum(&header[..78]);
        header[78..82].copy_from_slice(&crc.to_le_bytes());
        let mut body = Vec::with_capacity(self.fragments.len() * FRAGMENT_RECORD_LEN);
        for record in &self.fragments {
            let mut entry = vec![0u8; FRAGMENT_RECORD_LEN];
            entry[0..4].copy_from_slice(&record.id.index.to_le_bytes());
            entry[4] = record.id.role.as_u8();
            entry[5] = 0;
            entry[6..14].copy_from_slice(&record.node.as_u64().to_le_bytes());
            entry[14..46].copy_from_slice(&record.id.content);
            entry[46..54].copy_from_slice(&record.stored_len.to_le_bytes());
            let crc = crc32c_checksum(&entry[..54]);
            entry[54..58].copy_from_slice(&crc.to_le_bytes());
            body.extend_from_slice(&entry);
        }
        let mut contract = Vec::with_capacity(CONTRACT_HEADER_LEN + CONTRACT_CRC_LEN);
        contract.extend_from_slice(&CONTRACT_MAGIC.to_le_bytes());
        contract.push(1);
        contract.push(self.locality.as_u8());
        contract.push(0);
        contract.extend_from_slice(&self.min_available.to_le_bytes());
        let domain_count = u16::try_from(self.allowed_domains.len()).unwrap_or(u16::MAX);
        contract.extend_from_slice(&domain_count.to_le_bytes());
        for domain in &self.allowed_domains {
            let length = u32::try_from(domain.len()).unwrap_or(u32::MAX);
            contract.extend_from_slice(&length.to_le_bytes());
            contract.extend_from_slice(domain.as_bytes());
        }
        let contract_crc = crc32c_checksum(&contract);
        contract.extend_from_slice(&contract_crc.to_le_bytes());
        let mut body_with_contract = body;
        body_with_contract.extend_from_slice(&contract);
        let body_crc = crc32c_checksum(&body_with_contract);
        let content = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&header[..78]);
            hasher.update(&body_with_contract);
            hasher.finalize()
        };
        let mut footer = vec![0u8; LAYOUT_FOOTER_LEN];
        footer[0..4].copy_from_slice(&body_crc.to_le_bytes());
        footer[4..36].copy_from_slice(content.as_bytes());
        let crc = crc32c_checksum(&footer[..36]);
        footer[36..40].copy_from_slice(&crc.to_le_bytes());
        [header, body_with_contract, footer].concat()
    }

    /// Decodes and fully verifies durable bytes (identity, CRCs, coherence).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] on any violation. Unknown
    /// versions are rejected outright (pre-1.0: break, never migrate).
    // Decoder is linear field-by-field like the WAL scanner; splitting it
    // would obscure the wire order.
    #[allow(clippy::too_many_lines)]
    pub fn decode(bytes: &[u8]) -> Result<Self, RedundancyError> {
        let bad = |detail: String| RedundancyError::BadLayout { detail };
        if bytes.len()
            < LAYOUT_HEADER_LEN + LAYOUT_FOOTER_LEN + CONTRACT_HEADER_LEN + CONTRACT_CRC_LEN
        {
            return Err(bad(format!(
                "layout shorter than header + contract + footer: {}",
                bytes.len()
            )));
        }
        let (header, rest) = bytes.split_at(LAYOUT_HEADER_LEN);
        let (body_with_contract, footer) = rest.split_at(rest.len() - LAYOUT_FOOTER_LEN);
        if u32::from_le_bytes(
            header[0..4]
                .try_into()
                .map_err(|_| bad("magic unreadable".to_owned()))?,
        ) != LAYOUT_MAGIC
        {
            return Err(bad("layout magic mismatch".to_owned()));
        }
        if crc32c_checksum(&header[..78])
            != u32::from_le_bytes(
                header[78..82]
                    .try_into()
                    .map_err(|_| bad("header CRC unreadable".to_owned()))?,
            )
        {
            return Err(bad("layout header CRC mismatch".to_owned()));
        }
        let major = u16::from_le_bytes(
            header[4..6]
                .try_into()
                .map_err(|_| bad("major unreadable".to_owned()))?,
        );
        let minor = u16::from_le_bytes(
            header[6..8]
                .try_into()
                .map_err(|_| bad("minor unreadable".to_owned()))?,
        );
        if major != LAYOUT_MAJOR || minor != LAYOUT_MINOR {
            return Err(bad(format!("layout version {major}.{minor} unsupported")));
        }
        let kind = AssetKind::from_u8(header[8])?;
        let scheme = crate::SchemeId::from_u8(header[9])?;
        let domain = u64::from_le_bytes(
            header[10..18]
                .try_into()
                .map_err(|_| bad("domain unreadable".to_owned()))?,
        );
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&header[18..50]);
        let asset =
            AssetId::new(kind, domain, hash).map_err(|e| bad(format!("asset identity: {e}")))?;
        let logical_len = u64::from_le_bytes(
            header[50..58]
                .try_into()
                .map_err(|_| bad("len unreadable".to_owned()))?,
        );
        let generation = u64::from_le_bytes(
            header[58..66]
                .try_into()
                .map_err(|_| bad("generation unreadable".to_owned()))?,
        );
        if generation == 0 {
            return Err(bad("generation 0 is the invalid sentinel".to_owned()));
        }
        let (data, parity, copies, fragment_len) = (
            header[66],
            header[67],
            header[68],
            u32::from_le_bytes(
                header[70..74]
                    .try_into()
                    .map_err(|_| bad("shard unreadable".to_owned()))?,
            ),
        );
        if header[69] != 0 {
            return Err(bad("layout reserved byte nonzero".to_owned()));
        }
        let count = u32::from_le_bytes(
            header[74..78]
                .try_into()
                .map_err(|_| bad("count unreadable".to_owned()))?,
        ) as usize;
        let params = match scheme {
            crate::SchemeId::Replication => {
                SchemeParams::Replication(crate::ReplicationParams { copies })
            }
            crate::SchemeId::ReedSolomon => SchemeParams::ReedSolomon(RsParams {
                data,
                parity,
                fragment_len,
            }),
        };
        params.validate().map_err(|e| bad(format!("params: {e}")))?;
        // Count is a validated `u32` widened to `usize`; total <=40.
        #[allow(clippy::cast_possible_truncation)]
        let count_u32 = count as u32;
        if count_u32 != params.total_fragments() {
            return Err(bad(format!(
                "fragment count {count} disagrees with scheme total {}",
                params.total_fragments()
            )));
        }
        let body_len = count * FRAGMENT_RECORD_LEN;
        if body_with_contract.len() < body_len + CONTRACT_HEADER_LEN + CONTRACT_CRC_LEN {
            return Err(bad(format!(
                "body length {} disagrees with {count} fragments and contract",
                body_with_contract.len()
            )));
        }
        let body = &body_with_contract[..body_len];
        let contract = &body_with_contract[body_len..];
        if contract.len() < CONTRACT_HEADER_LEN + CONTRACT_CRC_LEN {
            return Err(bad("layout contract block is truncated".to_owned()));
        }
        let contract_payload = &contract[..contract.len() - CONTRACT_CRC_LEN];
        let mut contract_crc = [0u8; 4];
        contract_crc.copy_from_slice(&contract[contract.len() - CONTRACT_CRC_LEN..]);
        let mut contract_magic = [0u8; 4];
        contract_magic.copy_from_slice(&contract_payload[0..4]);
        if u32::from_le_bytes(contract_magic) != CONTRACT_MAGIC
            || contract_payload[4] != 1
            || contract_payload[5] > 2
            || contract_payload[6] != 0
            || crc32c_checksum(contract_payload) != u32::from_le_bytes(contract_crc)
        {
            return Err(bad("layout contract block is invalid".to_owned()));
        }
        let locality = LocalityRequirement::from_u8(contract_payload[5])
            .ok_or_else(|| bad("layout locality is invalid".to_owned()))?;
        let mut minimum = [0u8; 4];
        minimum.copy_from_slice(&contract_payload[7..11]);
        let min_available = u32::from_le_bytes(minimum);
        let mut domain_count_bytes = [0u8; 2];
        domain_count_bytes.copy_from_slice(&contract_payload[11..13]);
        let domain_count = usize::from(u16::from_le_bytes(domain_count_bytes));
        let mut cursor = CONTRACT_HEADER_LEN;
        let mut allowed_domains = Vec::with_capacity(domain_count);
        for _ in 0..domain_count {
            if cursor + 4 > contract_payload.len() {
                return Err(bad("layout domain length is truncated".to_owned()));
            }
            let mut length_bytes = [0u8; 4];
            length_bytes.copy_from_slice(&contract_payload[cursor..cursor + 4]);
            cursor += 4;
            let length = usize::try_from(u32::from_le_bytes(length_bytes))
                .map_err(|_| bad("layout domain length is invalid".to_owned()))?;
            if length == 0 || cursor + length > contract_payload.len() {
                return Err(bad("layout domain payload is invalid".to_owned()));
            }
            let domain = std::str::from_utf8(&contract_payload[cursor..cursor + length])
                .map_err(|_| bad("layout domain is not UTF-8".to_owned()))?
                .to_owned();
            cursor += length;
            allowed_domains.push(domain);
        }
        if cursor != contract_payload.len() {
            return Err(bad("layout contract has trailing bytes".to_owned()));
        }
        if min_available == 0 || min_available > params.total_fragments() {
            return Err(bad(
                "layout minimum available is outside scheme bounds".to_owned()
            ));
        }
        if u32::from_le_bytes(
            footer[0..4]
                .try_into()
                .map_err(|_| bad("body CRC unreadable".to_owned()))?,
        ) != crc32c_checksum(body_with_contract)
        {
            return Err(bad("layout body CRC mismatch".to_owned()));
        }
        let expect_content = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&header[..78]);
            hasher.update(body_with_contract);
            *hasher.finalize().as_bytes()
        };
        if footer[4..36] != expect_content {
            return Err(bad("layout content hash mismatch".to_owned()));
        }
        if crc32c_checksum(&footer[..36])
            != u32::from_le_bytes(
                footer[36..40]
                    .try_into()
                    .map_err(|_| bad("footer CRC unreadable".to_owned()))?,
            )
        {
            return Err(bad("layout footer CRC mismatch".to_owned()));
        }
        let mut fragments = Vec::with_capacity(count);
        for index in 0..count {
            let entry = &body[index * FRAGMENT_RECORD_LEN..(index + 1) * FRAGMENT_RECORD_LEN];
            if crc32c_checksum(&entry[..54])
                != u32::from_le_bytes(
                    entry[54..58]
                        .try_into()
                        .map_err(|_| bad("record CRC unreadable".to_owned()))?,
                )
            {
                return Err(bad(format!("fragment {index} record CRC mismatch")));
            }
            let frag_index = u32::from_le_bytes(
                entry[0..4]
                    .try_into()
                    .map_err(|_| bad("index unreadable".to_owned()))?,
            );
            if frag_index as usize != index {
                return Err(bad(format!("fragment order broken at {index}")));
            }
            if entry[5] != 0 {
                return Err(bad(format!("fragment {index} reserved nonzero")));
            }
            let role = FragmentRole::from_u8(entry[4])?;
            let node = kivi_types::NodeId::from_u64(u64::from_le_bytes(
                entry[6..14]
                    .try_into()
                    .map_err(|_| bad("node unreadable".to_owned()))?,
            ));
            let mut content = [0u8; 32];
            content.copy_from_slice(&entry[14..46]);
            if content == [0; 32] {
                return Err(bad(format!(
                    "fragment {index} content is the zero sentinel"
                )));
            }
            let stored_len = u64::from_le_bytes(
                entry[46..54]
                    .try_into()
                    .map_err(|_| bad("stored len unreadable".to_owned()))?,
            );
            let params_tag = match params {
                SchemeParams::Replication(_) => 0,
                SchemeParams::ReedSolomon(rs) => {
                    (u32::from(rs.data) << 24) ^ (u32::from(rs.parity) << 16) ^ rs.fragment_len
                }
            };
            fragments.push(FragmentRecord {
                id: FragmentId {
                    asset,
                    generation,
                    index: frag_index,
                    role,
                    params_tag,
                    content,
                },
                node,
                stored_len,
            });
        }
        Self::new_with_contract(
            asset,
            logical_len,
            generation,
            params,
            fragments,
            min_available,
            locality,
            allowed_domains,
        )
        .map_err(|e| bad(format!("coherence: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, ReplicationParams, SchemeParams};
    use kivi_types::NodeId;

    fn layout() -> RedundancyLayout {
        let asset = AssetId::new(AssetKind::Chunk, 7, [11; 32]).expect("asset");
        let params = SchemeParams::Replication(ReplicationParams { copies: 3 });
        let fragments = (0..3u32)
            .map(|index| FragmentRecord {
                id: FragmentId::for_bytes(asset, 5, index, params, b"bytes"),
                node: NodeId::from_u64(u64::from(index) + 1),
                stored_len: 5,
            })
            .collect();
        RedundancyLayout::new(asset, 5, 5, params, fragments).expect("builds")
    }

    #[test]
    fn layout_round_trips_deterministically() {
        let first = layout();
        let bytes = first.encode();
        let second = RedundancyLayout::decode(&bytes).expect("decodes");
        assert_eq!(first, second);
        assert_eq!(first.encode(), second.encode());
    }

    #[test]
    fn minimum_available_floor_round_trips() {
        let asset = AssetId::new(AssetKind::Chunk, 7, [12; 32]).expect("asset");
        let params = SchemeParams::Replication(ReplicationParams { copies: 3 });
        let fragments = (0..3u32)
            .map(|index| FragmentRecord {
                id: FragmentId::for_bytes(asset, 1, index, params, b"bytes"),
                node: NodeId::from_u64(u64::from(index) + 1),
                stored_len: 5,
            })
            .collect();
        let first = RedundancyLayout::new_with_contract(
            asset,
            5,
            1,
            params,
            fragments,
            2,
            LocalityRequirement::RackSpanning,
            vec![
                "rack-b".to_owned(),
                "rack-a".to_owned(),
                "rack-a".to_owned(),
            ],
        )
        .expect("builds");
        let second = RedundancyLayout::decode(&first.encode()).expect("decodes");
        assert_eq!(second.min_available, 2);
        assert_eq!(second.locality, LocalityRequirement::RackSpanning);
        assert_eq!(second.allowed_domains, ["rack-a", "rack-b"]);
        assert_eq!(first, second);
    }

    #[test]
    fn every_byte_flip_is_detected() {
        let bytes = layout().encode();
        for index in [0, 10, bytes.len() / 2, bytes.len() - 1] {
            let mut damaged = bytes.clone();
            damaged[index] ^= 0xFF;
            assert!(RedundancyLayout::decode(&damaged).is_err(), "byte {index}");
        }
    }

    #[test]
    fn old_versions_are_rejected_not_migrated() {
        let mut bytes = layout().encode();
        bytes[4] = 0x7F;
        assert!(RedundancyLayout::decode(&bytes).is_err());
    }
}
