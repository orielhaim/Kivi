//! Fragment wire codec: project-owned, versioned, deterministic RPC bytes.
//!
//! Nodes exchange fragments and layouts as [`FragmentRpc`] requests and
//! [`FragmentReply`] answers. Every message is little-endian with a fixed
//! header (`magic "KVFR"`, `major 1`, `minor 0`, tag, length) plus a
//! trailing CRC32C over all preceding bytes. Unknown versions are rejected
//! outright (pre-1.0: break, never migrate), trailing bytes are rejected,
//! and oversize payloads fail closed before allocation. This codec never
//! duplicates [`crate::RedundancyLayout`]'s durable encoding: layouts travel
//! opaquely as the exact bytes [`crate::RedundancyLayout::encode`] minted,
//! verified by the receiver with [`crate::RedundancyLayout::decode`].

use kivi_codec::integrity::crc32c_checksum;

use crate::{AssetId, RedundancyError};

/// Wire major version minted here.
pub const PROTO_MAJOR: u16 = 1;
/// Wire minor version minted here.
pub const PROTO_MINOR: u16 = 0;
/// Maximum fragment bytes accepted on the wire (8 MiB, matches the RS shard cap).
pub const MAX_FRAGMENT_WIRE_BYTES: usize = 8 * 1024 * 1024;
/// Maximum layout bytes accepted on the wire (4 MiB).
pub const MAX_LAYOUT_WIRE_BYTES: usize = 4 * 1024 * 1024;

/// Magic word: ASCII `"KVFR"` as little-endian `u32`.
const WIRE_MAGIC: u32 = 0x5246_564B;
/// Header length: magic `u32`, major `u16`, minor `u16`, tag `u8`,
/// reserved `u8`, body length `u32`.
const HEADER_LEN: usize = 14;
/// Trailer length: CRC32C `u32` over every preceding byte.
const TRAILER_LEN: usize = 4;
/// Largest body any message may carry (fragment cap plus field slack).
const MAX_BODY_LEN: usize = MAX_FRAGMENT_WIRE_BYTES + 256;
/// Encoded [`FragmentKey`] length: kind `u8`, domain `u64`, hash `[u8; 32]`,
/// generation `u64`, index `u32`.
const KEY_LEN: usize = 53;

/// Storage coordinates of one fragment: which asset, which generation,
/// which index. Carries no bytes, no placement, no scheme facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentKey {
    /// Asset family discriminant ([`crate::AssetKind`] frozen values).
    pub asset_kind: u8,
    /// Raw security-domain value.
    pub asset_domain: u64,
    /// Content hash minted by the owning fabric.
    pub asset_hash: [u8; 32],
    /// Redundancy generation.
    pub generation: u64,
    /// Fragment index (`0..total`).
    pub index: u32,
}

impl FragmentKey {
    /// Builds coordinates from an asset identity, generation, and index.
    #[must_use]
    pub const fn from_asset(asset: AssetId, generation: u64, index: u32) -> Self {
        Self {
            asset_kind: asset.kind.as_u8(),
            asset_domain: asset.domain,
            asset_hash: asset.hash,
            generation,
            index,
        }
    }

    /// Recovers the asset identity, rejecting unknown kinds and the zero sentinel.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] for unknown kind
    /// discriminants or the all-zero hash.
    pub fn to_asset(self) -> Result<AssetId, RedundancyError> {
        let kind = crate::AssetKind::from_u8(self.asset_kind)?;
        AssetId::new(kind, self.asset_domain, self.asset_hash).map_err(|error| {
            RedundancyError::BadLayout {
                detail: format!("fragment key asset: {error}"),
            }
        })
    }

    /// Encodes the deterministic 53 bytes.
    #[must_use]
    pub fn encode(self) -> [u8; KEY_LEN] {
        let mut out = [0u8; KEY_LEN];
        out[0] = self.asset_kind;
        out[1..9].copy_from_slice(&self.asset_domain.to_le_bytes());
        out[9..41].copy_from_slice(&self.asset_hash);
        out[41..49].copy_from_slice(&self.generation.to_le_bytes());
        out[49..53].copy_from_slice(&self.index.to_le_bytes());
        out
    }

    /// Decodes coordinates from exactly [`KEY_LEN`] bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] on short input.
    pub fn decode(bytes: &[u8]) -> Result<Self, RedundancyError> {
        if bytes.len() != KEY_LEN {
            return Err(RedundancyError::BadLayout {
                detail: format!("fragment key length {} is not {KEY_LEN}", bytes.len()),
            });
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[9..41]);
        Ok(Self {
            asset_kind: bytes[0],
            asset_domain: u64::from_le_bytes(into8(&bytes[1..9])),
            asset_hash: hash,
            generation: u64::from_le_bytes(into8(&bytes[41..49])),
            index: u32::from_le_bytes(into4(&bytes[49..53])),
        })
    }
}

/// One fragment RPC request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentRpc {
    /// Store fragment bytes (verified by the receiver before storing).
    Put {
        /// Fragment coordinates.
        key: FragmentKey,
        /// Role discriminant ([`crate::FragmentRole`] frozen values).
        role: u8,
        /// Encoding parameters tag (opaque here, bound by the layout).
        params_tag: u32,
        /// BLAKE3 of `bytes` (receiver verifies before storing).
        content: [u8; 32],
        /// Declared stored length (must equal `bytes.len()`).
        stored_len: u64,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
        /// Fragment bytes.
        bytes: Vec<u8>,
    },
    /// Fetch fragment bytes (hash-verified by the holder before serving).
    Get {
        /// Fragment coordinates.
        key: FragmentKey,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Probe fragment presence and health.
    Has {
        /// Fragment coordinates.
        key: FragmentKey,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Remove one fragment (idempotent).
    Drop {
        /// Fragment coordinates.
        key: FragmentKey,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Store a published layout's opaque bytes.
    PutLayout {
        /// Asset family discriminant.
        asset_kind: u8,
        /// Raw security-domain value.
        asset_domain: u64,
        /// Content hash minted by the owning fabric.
        asset_hash: [u8; 32],
        /// Layout generation.
        generation: u64,
        /// Control-plane generation that published it (fencing).
        control_generation: u64,
        /// Exact bytes [`crate::RedundancyLayout::encode`] minted.
        layout: Vec<u8>,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Fetch a published layout's opaque bytes.
    GetLayout {
        /// Asset family discriminant.
        asset_kind: u8,
        /// Raw security-domain value.
        asset_domain: u64,
        /// Content hash minted by the owning fabric.
        asset_hash: [u8; 32],
        /// Layout generation.
        generation: u64,
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Probe an asset's newest generation and layout hash.
    ProbeAsset {
        /// Asset family discriminant.
        asset_kind: u8,
        /// Raw security-domain value.
        asset_domain: u64,
        /// Content hash minted by the owning fabric.
        asset_hash: [u8; 32],
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
    /// Reclaim staged fragments that no accepted layout names. Disk
    /// hygiene only: orphans were never authoritative and can never answer
    /// a read, so the holder sweeps them under its own lock.
    SweepOrphans {
        /// Target incarnation (`0` = bootstrap probe).
        target_incarnation: u64,
    },
}

/// One fragment RPC answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentReply {
    /// Fragment stored; echoes the verified content hash.
    Stored {
        /// BLAKE3 of the stored bytes.
        content: [u8; 32],
    },
    /// Fragment bytes with their content hash.
    Bytes {
        /// BLAKE3 of `bytes`.
        content: [u8; 32],
        /// Hash-verified fragment bytes.
        bytes: Vec<u8>,
    },
    /// Presence and health probe answer.
    Presence {
        /// A representation exists.
        present: bool,
        /// The representation hash-verifies now.
        healthy: bool,
    },
    /// Fragment removed (idempotent: already-absent counts as dropped).
    Dropped,
    /// Opaque published layout bytes.
    LayoutBytes {
        /// Exact bytes the holder accepted.
        bytes: Vec<u8>,
    },
    /// Asset probe answer: newest generation plus layout hash.
    AssetStatus {
        /// Newest accepted generation (`0` with the zero hash when unknown).
        generation: u64,
        /// BLAKE3 of the accepted layout bytes (zero when unknown).
        layout_hash: [u8; 32],
    },
    /// Fragments reclaimed by one orphan sweep.
    Swept {
        /// How many staged fragments the holder removed.
        count: u64,
    },
    /// The holder refused the request (fencing, never silent loss).
    Refused {
        /// Why the request was refused.
        reason: RefuseReason,
    },
}

/// Why a holder refused an RPC (fencing and bounds, never silent loss).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefuseReason {
    /// Request generation is stale; carries the current generation.
    StaleGeneration {
        /// Newest generation the holder knows.
        current: u64,
    },
    /// Request incarnation is stale; carries the current incarnation.
    StaleIncarnation {
        /// Incarnation the holder currently serves.
        current: u64,
    },
    /// Payload exceeds the wire caps.
    Oversize,
    /// Bytes failed hash verification (never stored, never served).
    HashMismatch,
    /// No layout or fragment is known here.
    UnknownAsset,
    /// The holder is shutting down.
    ShuttingDown,
}

impl RefuseReason {
    /// Encodes the reason: code `u8` plus `u64` payload for stale variants.
    #[must_use]
    pub fn encode(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9);
        match self {
            Self::StaleGeneration { current } => {
                out.push(0);
                out.extend_from_slice(&current.to_le_bytes());
            }
            Self::StaleIncarnation { current } => {
                out.push(1);
                out.extend_from_slice(&current.to_le_bytes());
            }
            Self::Oversize => out.push(2),
            Self::HashMismatch => out.push(3),
            Self::UnknownAsset => out.push(4),
            Self::ShuttingDown => out.push(5),
        }
        out
    }

    /// Decodes a reason, rejecting unknown codes and ragged payloads.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] on unknown codes or length
    /// disagreement.
    pub fn decode(bytes: &[u8]) -> Result<Self, RedundancyError> {
        let bad = |detail: String| RedundancyError::BadLayout { detail };
        let Some((&code, rest)) = bytes.split_first() else {
            return Err(bad("refuse reason is empty".to_owned()));
        };
        match code {
            0 | 1 => {
                if rest.len() != 8 {
                    return Err(bad(format!("stale refuse payload holds {}", rest.len())));
                }
                let current = u64::from_le_bytes(into8(rest));
                if code == 0 {
                    Ok(Self::StaleGeneration { current })
                } else {
                    Ok(Self::StaleIncarnation { current })
                }
            }
            2 => {
                if !rest.is_empty() {
                    return Err(bad("oversize refuse carries a payload".to_owned()));
                }
                Ok(Self::Oversize)
            }
            3 => {
                if !rest.is_empty() {
                    return Err(bad("hash-mismatch refuse carries a payload".to_owned()));
                }
                Ok(Self::HashMismatch)
            }
            4 => {
                if !rest.is_empty() {
                    return Err(bad("unknown-asset refuse carries a payload".to_owned()));
                }
                Ok(Self::UnknownAsset)
            }
            5 => {
                if !rest.is_empty() {
                    return Err(bad("shutting-down refuse carries a payload".to_owned()));
                }
                Ok(Self::ShuttingDown)
            }
            other => Err(bad(format!("unknown refuse code {other}"))),
        }
    }
}

impl FragmentRpc {
    /// Message tag for [`Self::encode`].
    #[must_use]
    const fn tag(&self) -> u8 {
        match self {
            Self::Put { .. } => 1,
            Self::Get { .. } => 2,
            Self::Has { .. } => 3,
            Self::Drop { .. } => 4,
            Self::PutLayout { .. } => 5,
            Self::GetLayout { .. } => 6,
            Self::ProbeAsset { .. } => 7,
            Self::SweepOrphans { .. } => 8,
        }
    }

    /// Encodes the deterministic wire bytes (header, body, CRC32C).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let body = self.encode_body();
        encode_message(self.tag(), &body)
    }

    /// Decodes and fully verifies wire bytes (magic, version, length,
    /// CRC, exact body shape; trailing bytes rejected).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] on structural violations or
    /// unsupported versions, [`RedundancyError::CorruptFragment`] on CRC
    /// mismatch, and [`RedundancyError::Overloaded`] on oversize payloads.
    pub fn decode(bytes: &[u8]) -> Result<Self, RedundancyError> {
        let (tag, body) = decode_message(bytes)?;
        match tag {
            1 => decode_put(body),
            2..=4 => decode_keyed(tag, body),
            5 => decode_put_layout(body),
            6 => decode_get_layout(body),
            7 => decode_probe(body),
            8 => {
                if body.len() != 8 {
                    return Err(RedundancyError::BadLayout {
                        detail: format!("sweep-orphans rpc holds {} bytes", body.len()),
                    });
                }
                Ok(Self::SweepOrphans {
                    target_incarnation: u64::from_le_bytes(into8(body)),
                })
            }
            other => Err(RedundancyError::BadLayout {
                detail: format!("unknown rpc tag {other}"),
            }),
        }
    }

    /// Encodes the tag-specific body.
    fn encode_body(&self) -> Vec<u8> {
        match self {
            Self::Put {
                key,
                role,
                params_tag,
                content,
                stored_len,
                target_incarnation,
                bytes,
            } => {
                // Fixed part is 111 bytes; total stays under `u32::MAX`
                // because oversize payloads are rejected at decode.
                #[allow(clippy::cast_possible_truncation)]
                let bytes_len = bytes.len() as u32;
                let mut out = Vec::with_capacity(111 + bytes.len());
                out.extend_from_slice(&key.encode());
                out.push(*role);
                out.push(0);
                out.extend_from_slice(&params_tag.to_le_bytes());
                out.extend_from_slice(content);
                out.extend_from_slice(&stored_len.to_le_bytes());
                out.extend_from_slice(&target_incarnation.to_le_bytes());
                out.extend_from_slice(&bytes_len.to_le_bytes());
                out.extend_from_slice(bytes);
                out
            }
            Self::Get {
                key,
                target_incarnation,
            }
            | Self::Has {
                key,
                target_incarnation,
            }
            | Self::Drop {
                key,
                target_incarnation,
            } => {
                let mut out = Vec::with_capacity(KEY_LEN + 8);
                out.extend_from_slice(&key.encode());
                out.extend_from_slice(&target_incarnation.to_le_bytes());
                out
            }
            Self::PutLayout {
                asset_kind,
                asset_domain,
                asset_hash,
                generation,
                control_generation,
                layout,
                target_incarnation,
            } => {
                #[allow(clippy::cast_possible_truncation)]
                let layout_len = layout.len() as u32;
                let mut out = Vec::with_capacity(70 + layout.len());
                out.push(*asset_kind);
                out.push(0);
                out.extend_from_slice(&asset_domain.to_le_bytes());
                out.extend_from_slice(asset_hash);
                out.extend_from_slice(&generation.to_le_bytes());
                out.extend_from_slice(&control_generation.to_le_bytes());
                out.extend_from_slice(&target_incarnation.to_le_bytes());
                out.extend_from_slice(&layout_len.to_le_bytes());
                out.extend_from_slice(layout);
                out
            }
            Self::GetLayout {
                asset_kind,
                asset_domain,
                asset_hash,
                generation,
                target_incarnation,
            } => {
                let mut out = Vec::with_capacity(58);
                out.push(*asset_kind);
                out.push(0);
                out.extend_from_slice(&asset_domain.to_le_bytes());
                out.extend_from_slice(asset_hash);
                out.extend_from_slice(&generation.to_le_bytes());
                out.extend_from_slice(&target_incarnation.to_le_bytes());
                out
            }
            Self::ProbeAsset {
                asset_kind,
                asset_domain,
                asset_hash,
                target_incarnation,
            } => {
                let mut out = Vec::with_capacity(50);
                out.push(*asset_kind);
                out.push(0);
                out.extend_from_slice(&asset_domain.to_le_bytes());
                out.extend_from_slice(asset_hash);
                out.extend_from_slice(&target_incarnation.to_le_bytes());
                out
            }
            Self::SweepOrphans { target_incarnation } => target_incarnation.to_le_bytes().to_vec(),
        }
    }
}

impl FragmentReply {
    /// Message tag for encoding.
    #[must_use]
    const fn tag(&self) -> u8 {
        match self {
            Self::Stored { .. } => 1,
            Self::Bytes { .. } => 2,
            Self::Presence { .. } => 3,
            Self::Dropped => 4,
            Self::LayoutBytes { .. } => 5,
            Self::AssetStatus { .. } => 6,
            Self::Refused { .. } => 7,
            Self::Swept { .. } => 8,
        }
    }

    /// Encodes the deterministic wire bytes (header, body, CRC32C).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let body = match self {
            Self::Stored { content } => content.to_vec(),
            Self::Bytes { content, bytes } => {
                #[allow(clippy::cast_possible_truncation)]
                let bytes_len = bytes.len() as u32;
                let mut out = Vec::with_capacity(36 + bytes.len());
                out.extend_from_slice(content);
                out.extend_from_slice(&bytes_len.to_le_bytes());
                out.extend_from_slice(bytes);
                out
            }
            Self::Presence { present, healthy } => {
                vec![u8::from(*present), u8::from(*healthy)]
            }
            Self::Dropped => Vec::new(),
            Self::LayoutBytes { bytes } => {
                #[allow(clippy::cast_possible_truncation)]
                let bytes_len = bytes.len() as u32;
                let mut out = Vec::with_capacity(4 + bytes.len());
                out.extend_from_slice(&bytes_len.to_le_bytes());
                out.extend_from_slice(bytes);
                out
            }
            Self::AssetStatus {
                generation,
                layout_hash,
            } => {
                let mut out = Vec::with_capacity(40);
                out.extend_from_slice(&generation.to_le_bytes());
                out.extend_from_slice(layout_hash);
                out
            }
            Self::Refused { reason } => reason.encode(),
            Self::Swept { count } => count.to_le_bytes().to_vec(),
        };
        encode_message(self.tag(), &body)
    }

    /// Decodes and fully verifies wire bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::BadLayout`] on structural violations or
    /// unsupported versions, [`RedundancyError::CorruptFragment`] on CRC
    /// mismatch, and [`RedundancyError::Overloaded`] on oversize payloads.
    pub fn decode(bytes: &[u8]) -> Result<Self, RedundancyError> {
        let (tag, body) = decode_message(bytes)?;
        match tag {
            1 => {
                if body.len() != 32 {
                    return Err(RedundancyError::BadLayout {
                        detail: format!("stored reply holds {} bytes", body.len()),
                    });
                }
                let mut content = [0u8; 32];
                content.copy_from_slice(body);
                Ok(Self::Stored { content })
            }
            2 => {
                if body.len() < 36 {
                    return Err(RedundancyError::BadLayout {
                        detail: format!("bytes reply holds {} bytes", body.len()),
                    });
                }
                let mut content = [0u8; 32];
                content.copy_from_slice(&body[..32]);
                let blob = capped_blob(&body[32..], MAX_FRAGMENT_WIRE_BYTES)?;
                Ok(Self::Bytes {
                    content,
                    bytes: blob.to_vec(),
                })
            }
            3 => {
                if body.len() != 2 || body[0] > 1 || body[1] > 1 {
                    return Err(RedundancyError::BadLayout {
                        detail: "presence reply is not two flag bytes".to_owned(),
                    });
                }
                Ok(Self::Presence {
                    present: body[0] == 1,
                    healthy: body[1] == 1,
                })
            }
            4 => {
                if !body.is_empty() {
                    return Err(RedundancyError::BadLayout {
                        detail: "dropped reply carries a payload".to_owned(),
                    });
                }
                Ok(Self::Dropped)
            }
            5 => {
                let blob = capped_blob(body, MAX_LAYOUT_WIRE_BYTES)?;
                Ok(Self::LayoutBytes {
                    bytes: blob.to_vec(),
                })
            }
            6 => {
                if body.len() != 40 {
                    return Err(RedundancyError::BadLayout {
                        detail: format!("asset status holds {} bytes", body.len()),
                    });
                }
                let mut layout_hash = [0u8; 32];
                layout_hash.copy_from_slice(&body[8..40]);
                Ok(Self::AssetStatus {
                    generation: u64::from_le_bytes(into8(&body[..8])),
                    layout_hash,
                })
            }
            7 => Ok(Self::Refused {
                reason: RefuseReason::decode(body)?,
            }),
            8 => {
                if body.len() != 8 {
                    return Err(RedundancyError::BadLayout {
                        detail: format!("swept reply holds {} bytes", body.len()),
                    });
                }
                Ok(Self::Swept {
                    count: u64::from_le_bytes(into8(body)),
                })
            }
            other => Err(RedundancyError::BadLayout {
                detail: format!("unknown reply tag {other}"),
            }),
        }
    }
}

/// Frames one message: header, body, CRC32C over all preceding bytes.
fn encode_message(tag: u8, body: &[u8]) -> Vec<u8> {
    // Bodies are capped at decode; lengths here stay under `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let body_len = body.len() as u32;
    let mut out = Vec::with_capacity(HEADER_LEN + body.len() + TRAILER_LEN);
    out.extend_from_slice(&WIRE_MAGIC.to_le_bytes());
    out.extend_from_slice(&PROTO_MAJOR.to_le_bytes());
    out.extend_from_slice(&PROTO_MINOR.to_le_bytes());
    out.push(tag);
    out.push(0);
    out.extend_from_slice(&body_len.to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32c_checksum(&out).to_le_bytes());
    out
}

/// Verifies framing and returns the tag plus body slice.
fn decode_message(bytes: &[u8]) -> Result<(u8, &[u8]), RedundancyError> {
    let bad = |detail: String| RedundancyError::BadLayout { detail };
    if bytes.len() < HEADER_LEN + TRAILER_LEN {
        return Err(bad(format!("message holds {} bytes", bytes.len())));
    }
    if u32::from_le_bytes(into4(&bytes[0..4])) != WIRE_MAGIC {
        return Err(bad("wire magic mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(into2(&bytes[4..6]));
    let minor = u16::from_le_bytes(into2(&bytes[6..8]));
    if major != PROTO_MAJOR || minor != PROTO_MINOR {
        return Err(bad(format!("wire version {major}.{minor} unsupported")));
    }
    if bytes[9] != 0 {
        return Err(bad("wire reserved byte nonzero".to_owned()));
    }
    let tag = bytes[8];
    let body_len = u32::from_le_bytes(into4(&bytes[10..14])) as usize;
    if body_len > MAX_BODY_LEN {
        return Err(RedundancyError::Overloaded {
            detail: format!("wire body {body_len} exceeds {MAX_BODY_LEN}"),
        });
    }
    if bytes.len() != HEADER_LEN + body_len + TRAILER_LEN {
        return Err(bad(format!(
            "wire length {} disagrees with body {body_len}",
            bytes.len()
        )));
    }
    let (framed, trailer) = bytes.split_at(HEADER_LEN + body_len);
    let expect = crc32c_checksum(framed);
    if u32::from_le_bytes(into4(trailer)) != expect {
        return Err(RedundancyError::CorruptFragment {
            detail: "wire CRC mismatch".to_owned(),
        });
    }
    Ok((tag, &framed[HEADER_LEN..]))
}

/// Reads a length-prefixed blob, enforcing the byte cap before slicing.
fn capped_blob(body: &[u8], cap: usize) -> Result<&[u8], RedundancyError> {
    if body.len() < 4 {
        return Err(RedundancyError::BadLayout {
            detail: format!("blob prefix holds {} bytes", body.len()),
        });
    }
    // `u32` widened to `usize` (never truncates).
    let blob_len = u32::from_le_bytes(into4(&body[..4])) as usize;
    if blob_len > cap {
        return Err(RedundancyError::Overloaded {
            detail: format!("wire blob {blob_len} exceeds {cap}"),
        });
    }
    if body.len() != 4 + blob_len {
        return Err(RedundancyError::BadLayout {
            detail: format!("blob length {} disagrees with {blob_len}", body.len()),
        });
    }
    Ok(&body[4..])
}

/// Decodes a `Put` body (exact shape, capped bytes).
fn decode_put(body: &[u8]) -> Result<FragmentRpc, RedundancyError> {
    if body.len() < 111 {
        return Err(RedundancyError::BadLayout {
            detail: format!("put holds {} bytes", body.len()),
        });
    }
    let key = FragmentKey::decode(&body[..KEY_LEN])?;
    if body[KEY_LEN + 1] != 0 {
        return Err(RedundancyError::BadLayout {
            detail: "put reserved byte nonzero".to_owned(),
        });
    }
    let mut content = [0u8; 32];
    content.copy_from_slice(&body[KEY_LEN + 6..KEY_LEN + 38]);
    let blob = capped_blob(&body[KEY_LEN + 54..], MAX_FRAGMENT_WIRE_BYTES)?;
    Ok(FragmentRpc::Put {
        key,
        role: body[KEY_LEN],
        params_tag: u32::from_le_bytes(into4(&body[KEY_LEN + 2..KEY_LEN + 6])),
        content,
        stored_len: u64::from_le_bytes(into8(&body[KEY_LEN + 38..KEY_LEN + 46])),
        target_incarnation: u64::from_le_bytes(into8(&body[KEY_LEN + 46..KEY_LEN + 54])),
        bytes: blob.to_vec(),
    })
}

/// Decodes `Get`/`Has`/`Drop` bodies (key plus incarnation).
fn decode_keyed(tag: u8, body: &[u8]) -> Result<FragmentRpc, RedundancyError> {
    if body.len() != KEY_LEN + 8 {
        return Err(RedundancyError::BadLayout {
            detail: format!("keyed rpc holds {} bytes", body.len()),
        });
    }
    let key = FragmentKey::decode(&body[..KEY_LEN])?;
    let target_incarnation = u64::from_le_bytes(into8(&body[KEY_LEN..]));
    match tag {
        2 => Ok(FragmentRpc::Get {
            key,
            target_incarnation,
        }),
        3 => Ok(FragmentRpc::Has {
            key,
            target_incarnation,
        }),
        _ => Ok(FragmentRpc::Drop {
            key,
            target_incarnation,
        }),
    }
}

/// Decodes a `PutLayout` body (exact shape, capped layout).
fn decode_put_layout(body: &[u8]) -> Result<FragmentRpc, RedundancyError> {
    if body.len() < 70 {
        return Err(RedundancyError::BadLayout {
            detail: format!("put-layout holds {} bytes", body.len()),
        });
    }
    if body[1] != 0 {
        return Err(RedundancyError::BadLayout {
            detail: "put-layout reserved byte nonzero".to_owned(),
        });
    }
    let mut asset_hash = [0u8; 32];
    asset_hash.copy_from_slice(&body[10..42]);
    let blob = capped_blob(&body[66..], MAX_LAYOUT_WIRE_BYTES)?;
    Ok(FragmentRpc::PutLayout {
        asset_kind: body[0],
        asset_domain: u64::from_le_bytes(into8(&body[2..10])),
        asset_hash,
        generation: u64::from_le_bytes(into8(&body[42..50])),
        control_generation: u64::from_le_bytes(into8(&body[50..58])),
        layout: blob.to_vec(),
        target_incarnation: u64::from_le_bytes(into8(&body[58..66])),
    })
}

/// Decodes a `GetLayout` body (exact 58 bytes).
fn decode_get_layout(body: &[u8]) -> Result<FragmentRpc, RedundancyError> {
    if body.len() != 58 || body[1] != 0 {
        return Err(RedundancyError::BadLayout {
            detail: "get-layout shape invalid".to_owned(),
        });
    }
    let mut asset_hash = [0u8; 32];
    asset_hash.copy_from_slice(&body[10..42]);
    Ok(FragmentRpc::GetLayout {
        asset_kind: body[0],
        asset_domain: u64::from_le_bytes(into8(&body[2..10])),
        asset_hash,
        generation: u64::from_le_bytes(into8(&body[42..50])),
        target_incarnation: u64::from_le_bytes(into8(&body[50..58])),
    })
}

/// Decodes a `ProbeAsset` body (exact 50 bytes).
fn decode_probe(body: &[u8]) -> Result<FragmentRpc, RedundancyError> {
    if body.len() != 50 || body[1] != 0 {
        return Err(RedundancyError::BadLayout {
            detail: "probe shape invalid".to_owned(),
        });
    }
    let mut asset_hash = [0u8; 32];
    asset_hash.copy_from_slice(&body[10..42]);
    Ok(FragmentRpc::ProbeAsset {
        asset_kind: body[0],
        asset_domain: u64::from_le_bytes(into8(&body[2..10])),
        asset_hash,
        target_incarnation: u64::from_le_bytes(into8(&body[42..50])),
    })
}

/// Copies two bytes into an array (length pre-checked by the caller).
fn into2(bytes: &[u8]) -> [u8; 2] {
    let mut out = [0u8; 2];
    out.copy_from_slice(&bytes[..2]);
    out
}

/// Copies four bytes into an array (length pre-checked by the caller).
fn into4(bytes: &[u8]) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(&bytes[..4]);
    out
}

/// Copies eight bytes into an array (length pre-checked by the caller).
fn into8(bytes: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&bytes[..8]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, ReplicationParams, SchemeParams};

    fn asset() -> AssetId {
        AssetId::new(AssetKind::Chunk, 7, [11; 32]).expect("asset")
    }

    fn key() -> FragmentKey {
        FragmentKey::from_asset(asset(), 5, 2)
    }

    fn rpcs() -> Vec<FragmentRpc> {
        vec![
            FragmentRpc::Put {
                key: key(),
                role: 1,
                params_tag: 0x0102_0304,
                content: [9; 32],
                stored_len: 3,
                target_incarnation: 12,
                bytes: b"shard".to_vec(),
            },
            FragmentRpc::Get {
                key: key(),
                target_incarnation: 0,
            },
            FragmentRpc::Has {
                key: key(),
                target_incarnation: 4,
            },
            FragmentRpc::Drop {
                key: key(),
                target_incarnation: 4,
            },
            FragmentRpc::PutLayout {
                asset_kind: 0,
                asset_domain: 7,
                asset_hash: [11; 32],
                generation: 3,
                control_generation: 40,
                layout: b"layout-bytes".to_vec(),
                target_incarnation: 1,
            },
            FragmentRpc::GetLayout {
                asset_kind: 1,
                asset_domain: 2,
                asset_hash: [5; 32],
                generation: 9,
                target_incarnation: 0,
            },
            FragmentRpc::ProbeAsset {
                asset_kind: 2,
                asset_domain: 3,
                asset_hash: [6; 32],
                target_incarnation: 8,
            },
            FragmentRpc::SweepOrphans {
                target_incarnation: 5,
            },
        ]
    }

    fn replies() -> Vec<FragmentReply> {
        vec![
            FragmentReply::Stored { content: [1; 32] },
            FragmentReply::Bytes {
                content: [2; 32],
                bytes: b"frag".to_vec(),
            },
            FragmentReply::Presence {
                present: true,
                healthy: false,
            },
            FragmentReply::Dropped,
            FragmentReply::LayoutBytes {
                bytes: b"gen-3".to_vec(),
            },
            FragmentReply::AssetStatus {
                generation: 7,
                layout_hash: [3; 32],
            },
            FragmentReply::Refused {
                reason: RefuseReason::StaleGeneration { current: 9 },
            },
            FragmentReply::Refused {
                reason: RefuseReason::StaleIncarnation { current: 2 },
            },
            FragmentReply::Refused {
                reason: RefuseReason::Oversize,
            },
            FragmentReply::Refused {
                reason: RefuseReason::HashMismatch,
            },
            FragmentReply::Refused {
                reason: RefuseReason::UnknownAsset,
            },
            FragmentReply::Refused {
                reason: RefuseReason::ShuttingDown,
            },
            FragmentReply::Swept { count: 17 },
        ]
    }

    #[test]
    fn every_rpc_round_trips_deterministically() {
        for rpc in rpcs() {
            let bytes = rpc.encode();
            let back = FragmentRpc::decode(&bytes).expect("decodes");
            assert_eq!(rpc, back);
            assert_eq!(rpc.encode(), back.encode());
        }
    }

    #[test]
    fn every_reply_round_trips_deterministically() {
        for reply in replies() {
            let bytes = reply.encode();
            let back = FragmentReply::decode(&bytes).expect("decodes");
            assert_eq!(reply, back);
            assert_eq!(reply.encode(), back.encode());
        }
    }

    #[test]
    fn key_converts_to_asset_identity() {
        let back = key().to_asset().expect("converts");
        assert_eq!(back, asset());
        let bad_kind = FragmentKey {
            asset_kind: 99,
            ..key()
        };
        assert!(bad_kind.to_asset().is_err());
        let zero = FragmentKey {
            asset_hash: [0; 32],
            ..key()
        };
        assert!(zero.to_asset().is_err());
    }

    #[test]
    fn every_byte_flip_is_detected() {
        let bytes = rpcs()[0].encode();
        for index in 0..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[index] ^= 0xFF;
            assert!(
                FragmentRpc::decode(&damaged).is_err(),
                "byte {index} undetected"
            );
        }
    }

    #[test]
    fn truncation_is_rejected() {
        let bytes = rpcs()[0].encode();
        assert!(FragmentRpc::decode(&[]).is_err());
        assert!(FragmentRpc::decode(&bytes[..10]).is_err());
        assert!(FragmentRpc::decode(&bytes[..bytes.len() - 1]).is_err());
        let reply = FragmentReply::Bytes {
            content: [2; 32],
            bytes: b"frag".to_vec(),
        }
        .encode();
        assert!(FragmentReply::decode(&reply[..reply.len() - 2]).is_err());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = rpcs()[1].encode();
        bytes.push(0);
        assert!(FragmentRpc::decode(&bytes).is_err());
        let mut reply = FragmentReply::Dropped.encode();
        reply.extend_from_slice(&[1, 2, 3]);
        assert!(FragmentReply::decode(&reply).is_err());
    }

    #[test]
    fn unknown_versions_are_rejected_not_migrated() {
        let mut bytes = rpcs()[1].encode();
        bytes[4] = 0x7F;
        assert!(FragmentRpc::decode(&bytes).is_err());
        let mut minor = rpcs()[1].encode();
        minor[6] = 0x01;
        assert!(FragmentRpc::decode(&minor).is_err());
    }

    #[test]
    fn oversize_payloads_fail_closed_before_serving() {
        let big = FragmentRpc::Put {
            key: key(),
            role: 0,
            params_tag: 0,
            content: [1; 32],
            stored_len: (MAX_FRAGMENT_WIRE_BYTES + 1) as u64,
            target_incarnation: 0,
            bytes: vec![7u8; MAX_FRAGMENT_WIRE_BYTES + 1],
        };
        assert!(FragmentRpc::decode(&big.encode()).is_err());
        // A lying length prefix is rejected without allocating the claim.
        let mut header_only = vec![0u8; HEADER_LEN];
        header_only[0..4].copy_from_slice(&WIRE_MAGIC.to_le_bytes());
        header_only[4..6].copy_from_slice(&PROTO_MAJOR.to_le_bytes());
        header_only[6..8].copy_from_slice(&PROTO_MINOR.to_le_bytes());
        header_only[8] = 1;
        header_only[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(FragmentRpc::decode(&header_only).is_err());
    }

    #[test]
    fn layout_bytes_reference_real_layout_codec() {
        let asset_id = asset();
        let params = SchemeParams::Replication(ReplicationParams { copies: 2 });
        let fragments = (0..2u32)
            .map(|index| crate::FragmentRecord {
                id: crate::FragmentId::for_bytes(asset_id, 1, index, params, b"bytes"),
                node: kivi_types::NodeId::from_u64(u64::from(index) + 1),
                stored_len: 5,
            })
            .collect();
        let layout =
            crate::RedundancyLayout::new(asset_id, 5, 1, params, fragments).expect("builds");
        let rpc = FragmentRpc::PutLayout {
            asset_kind: 0,
            asset_domain: 7,
            asset_hash: [11; 32],
            generation: 1,
            control_generation: 2,
            layout: layout.encode(),
            target_incarnation: 0,
        };
        let back = FragmentRpc::decode(&rpc.encode()).expect("decodes");
        assert_eq!(rpc, back);
    }
}
