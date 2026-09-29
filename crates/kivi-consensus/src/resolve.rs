//! Sidecar content resolution: the seam between replicated sidecars and
//! whatever can produce them.
//!
//! The Raft durability gate needs one thing from the outside world: given a
//! manifest (or chunk) id, hand back bytes that are complete and verified. It
//! does not care whether those bytes came from a local store, a peer, a
//! redundancy reconstruction, or some other storage tier, and it must not grow
//! knowledge of erasure coding to get them. That is the whole reason this
//! module exists.
//!
//! Two directions cross the seam, and they are deliberately different traits
//! because their contracts are different:
//!
//! * [`SidecarResolver`] is *authoritative and fallible*. The gate calls it
//!   when no peer could serve the content, and it must answer with either
//!   complete verified bytes or a [`ResolveError`] that says which of the three
//!   distinct things went wrong.
//! * [`SidecarProtector`] is *best-effort and infallible*. The propose path
//!   hands it a freshly staged root and moves on. Redundancy is a second copy,
//!   never a correctness condition, so a protection that fails costs
//!   redundancy and nothing else.
//!
//! Collapsing these into one trait would force the gate to tolerate silent
//! protection failures or force the propose path to handle reconstruction
//! errors, and both would be wrong.
//!
//! ## Why the outcome has three arms, not one
//!
//! The previous gate reported every acquisition failure as
//! `SidecarError::Missing { detail }` and, on the serving side, a refusal
//! carrying only the manifest id. That is a contentless error: a waiter could
//! not tell "nobody has it and nobody can rebuild it" (permanently lost) from
//! "it is being staged right now, try again shortly" (a few hundred
//! milliseconds). Every caller then had to guess, and guessing wrong in the
//! permissive direction is how a recoverable condition becomes a dead node.
//! [`ResolveError`] makes the distinction at the point where the information
//! exists and preserves it through every mapping.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Resolution result future.
///
/// Deliberately not `Send`. The seam is consumed on a Compio reactor inside the
/// consensus plane, which is thread-per-core by architecture (`OpenRaft` runs
/// with `single-threaded` for exactly this reason), and a resolver that waits
/// out a "not published yet" answer has to sleep on that reactor's clock. A
/// `Send` bound here would forbid the most natural implementation of the trait
/// and buy nothing: the guarantee the gate needs is that returned bytes are
/// complete and verified, not which thread produced them.
pub type ResolveFuture<T> = Pin<Box<dyn Future<Output = Result<T, ResolveError>>>>;

/// Why content could not be produced.
///
/// The three arms are not a taxonomy for its own sake - each one licenses a
/// different decision, and collapsing them is what made the old gate's
/// behaviour unpredictable.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResolveError {
    /// Not available now, but expected to be shortly: a holder is still
    /// staging, a peer is restarting, or a layout is mid-publication. A later
    /// attempt is worth making and bounded waiting is the right response.
    #[error("sidecar pending: {detail}")]
    Pending {
        /// What the resolver was waiting on.
        detail: String,
    },
    /// Not available and not coming: no complete copy anywhere and not enough
    /// surviving material to rebuild one. Retrying changes nothing until a
    /// peer publishes a copy, so callers should surface it rather than wait.
    #[error("sidecar unrecoverable: {detail}")]
    Unavailable {
        /// Why no copy could be produced.
        detail: String,
    },
    /// Located, but failed verification. Never becomes available and is never
    /// installed: corruption is a hard failure, never an availability
    /// problem.
    #[error("sidecar corrupt: {detail}")]
    Corrupt {
        /// What failed verification.
        detail: String,
    },
}

impl ResolveError {
    /// Whether waiting and trying again can plausibly change the answer.
    ///
    /// Only [`ResolveError::Pending`] qualifies. Retrying an
    /// [`ResolveError::Unavailable`] answer burns the caller's budget against
    /// a fixed fact, and retrying [`ResolveError::Corrupt`] is not just futile
    /// but wrong-headed: it is the one case where a second attempt is more
    /// likely to install bad content than good.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }
}

/// Produces complete, verified sidecar content by identity.
///
/// Implementations own every decision the gate must not make: which tiers to
/// consult, in what order, how much to wait, and what to do when a tier is
/// mid-write. They return bytes only after the content verifies against the
/// requested identity, so the gate's own verification is a second, independent
/// check rather than the only one.
pub trait SidecarResolver: Send + Sync + fmt::Debug {
    /// Returns the complete canonical manifest bytes for `manifest`.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] describing whether the content is pending,
    /// unrecoverable, or corrupt.
    fn resolve_manifest(&self, manifest: ManifestId) -> ResolveFuture<Vec<u8>>;

    /// Returns the complete chunk bytes for `chunk`, whose logical length is
    /// `len`.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] describing whether the content is pending,
    /// unrecoverable, or corrupt.
    fn resolve_chunk(&self, chunk: ChunkId, len: u64) -> ResolveFuture<Vec<u8>>;
}

/// One freshly staged root handed over for redundancy protection.
///
/// Only identities travel: the protection implementation reads the bytes back
/// from the sidecar store it already shares with the stager, so the caller's
/// ownership of a multi-megabyte value is not extended across a hand-off it
/// does not take part in.
///
/// Named apart from [`crate::sidecar::StagedRoot`] deliberately. That one is
/// what a stager receives back, canonical bytes included, because it still has
/// to prove and install them; this one is a pure hand-off across the seam. The
/// domain travels here too, which is the one field the stager's own view does
/// not carry, and a root filed under the wrong domain would protect content
/// nothing can ever ask for again.
#[derive(Debug, Clone)]
pub struct ProtectedRoot {
    /// Security domain the root belongs to.
    pub domain: SecurityDomainId,
    /// Manifest addressing the value.
    pub manifest: ManifestId,
    /// Total logical bytes.
    pub logical_len: u64,
    /// Chunk ids in order.
    pub chunks: Vec<ChunkId>,
}

/// Takes custody of a staged root for redundancy protection.
///
/// Never fails and never blocks: the caller has already proven local
/// durability, so a protection miss costs a redundant copy and nothing more.
/// Implementations that cannot start work immediately must drop it rather than
/// delay the write it follows.
pub trait SidecarProtector: Send + Sync + fmt::Debug {
    /// Accepts one staged root for protection.
    fn protect_root(&self, root: ProtectedRoot);
}

/// Late-bound holders for the two seams above.
///
/// The gate is created while the node is opening, but the redundancy plane is
/// built *from* an already-open node, so no resolver or protector exists at
/// construction time. These slots close that cycle without a second code
/// path: the gate and the propose path read the same cells, and the plane
/// installs into them once it is up.
///
/// Reads are clone-under-`Arc` and never hold a lock across await. A poisoned
/// lock reads as absent rather than panicking: protection and redundancy
/// resolution are both additive, and neither may be the thing that takes down
/// a node whose local durability is already proven.
#[derive(Debug, Default)]
pub struct ResolutionSlots {
    resolver: RwLock<Option<Arc<dyn SidecarResolver>>>,
    protector: RwLock<Option<Arc<dyn SidecarProtector>>>,
}

impl ResolutionSlots {
    /// Creates empty slots.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs (or replaces) the content resolver.
    pub fn install_resolver(&self, resolver: Arc<dyn SidecarResolver>) {
        if let Ok(mut slot) = self.resolver.write() {
            *slot = Some(resolver);
        }
    }

    /// Installs (or replaces) the staged-root protector.
    pub fn install_protector(&self, protector: Arc<dyn SidecarProtector>) {
        if let Ok(mut slot) = self.protector.write() {
            *slot = Some(protector);
        }
    }

    /// Returns the installed resolver, if any.
    #[must_use]
    pub fn resolver(&self) -> Option<Arc<dyn SidecarResolver>> {
        self.resolver.read().ok().and_then(|slot| slot.clone())
    }

    /// Returns the installed protector, if any.
    #[must_use]
    pub fn protector(&self) -> Option<Arc<dyn SidecarProtector>> {
        self.protector.read().ok().and_then(|slot| slot.clone())
    }
}

/// Hands one staged root to the installed protector, if there is one.
///
/// A no-op without a protector (single-node engines, ephemeral mode, and
/// before the redundancy plane opens) and a no-op when the hand-off itself
/// panics: protection is additive, so it must never be able to fail a write
/// whose local durability is already proven.
pub fn protect_staged_root(slots: &ResolutionSlots, root: ProtectedRoot) {
    let Some(protector) = slots.protector() else {
        return;
    };
    protector.protect_root(root);
}
