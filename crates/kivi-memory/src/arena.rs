//! Worker-local relocatable behavior-aware arenas.
//!
//! One [`WorkerArenas`] owns seven [`ClassArena`]s (one per
//! [`BehaviorClass`]), each a compact generational slab over `Bytes`
//! payloads. The owning worker is the single mutator (RFC §16): no locks,
//! no atomics on the fast path. Relocation and compaction only rewrite
//! slot contents and bump bookkeeping; outstanding [`ObjectHandle`]s stay
//! valid across moves *within* a slot, and slot reuse bumps the
//! generation so stale handles fail loudly instead of reading freed bytes.
//!
//! OBASE contributes the grouping insight (behavior, not size); Kivi's
//! single-owner model drops OBASE's lock-free concurrency complexity —
//! there are no concurrent mutators to coordinate with.
//!
//! Handles never escape as raw pointers: resolution clones the resident
//! `Bytes` (a cheap refcount bump) or reports a stale-handle error.

use bytes::Bytes;

use crate::error::MemoryError;
use crate::handle::{BehaviorClass, ObjectHandle};

/// One slot in a class arena.
#[derive(Debug, Clone)]
struct Slot {
    /// Current generation (`0` = never issued; handles start at 1).
    generation: u32,
    /// Resident payload, or `None` when the slot is free.
    payload: Option<Bytes>,
    /// Live bytes hint for compaction accounting.
    live: bool,
}

/// One behavior class's slab.
#[derive(Debug)]
struct ClassArena {
    class: BehaviorClass,
    slots: Vec<Slot>,
    free: Vec<u32>,
    live_count: usize,
    live_bytes: u64,
}

impl ClassArena {
    fn new(class: BehaviorClass) -> Self {
        Self {
            class,
            slots: Vec::new(),
            free: Vec::new(),
            live_count: 0,
            live_bytes: 0,
        }
    }

    fn alloc(&mut self, payload: Bytes) -> Result<ObjectHandle, MemoryError> {
        if payload.len() > crate::MEDIUM_MAX {
            return Err(MemoryError::TooLarge {
                len: payload.len() as u64,
                max: crate::MEDIUM_MAX as u64,
                context: "arena payload",
            });
        }
        let len = payload.len() as u64;
        if let Some(slot) = self.free.pop() {
            let index = slot as usize;
            if index >= self.slots.len() {
                return Err(MemoryError::Invalid {
                    detail: "arena free list out of bounds".to_owned(),
                });
            }
            let entry = &mut self.slots[index];
            entry.generation = entry.generation.wrapping_add(1).max(1);
            entry.payload = Some(payload);
            entry.live = true;
            self.live_count += 1;
            self.live_bytes += len;
            Ok(ObjectHandle::new(slot, entry.generation))
        } else {
            if self.slots.len() >= u32::MAX as usize - 1 {
                return Err(MemoryError::Overloaded {
                    queue: "arena slots",
                });
            }
            let Ok(slot) = u32::try_from(self.slots.len()) else {
                return Err(MemoryError::Overloaded {
                    queue: "arena slots",
                });
            };
            self.slots.push(Slot {
                generation: 1,
                payload: Some(payload),
                live: true,
            });
            self.live_count += 1;
            self.live_bytes += len;
            Ok(ObjectHandle::new(slot, 1))
        }
    }

    fn get(&self, handle: ObjectHandle) -> Result<Bytes, MemoryError> {
        let index = handle.slot as usize;
        let Some(entry) = self.slots.get(index) else {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: 0,
            });
        };
        if entry.generation != handle.generation || !entry.live {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: entry.generation,
            });
        }
        entry.payload.clone().ok_or(MemoryError::StaleHandle {
            handle,
            expected: handle.generation,
            found: entry.generation,
        })
    }

    fn update(&mut self, handle: ObjectHandle, payload: Bytes) -> Result<(), MemoryError> {
        let index = handle.slot as usize;
        let Some(entry) = self.slots.get_mut(index) else {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: 0,
            });
        };
        if entry.generation != handle.generation || !entry.live {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: entry.generation,
            });
        }
        if payload.len() > crate::MEDIUM_MAX {
            return Err(MemoryError::TooLarge {
                len: payload.len() as u64,
                max: crate::MEDIUM_MAX as u64,
                context: "arena payload",
            });
        }
        let old_len = entry.payload.as_ref().map_or(0, Bytes::len) as u64;
        let new_len = payload.len() as u64;
        entry.payload = Some(payload);
        self.live_bytes = self
            .live_bytes
            .saturating_sub(old_len)
            .saturating_add(new_len);
        Ok(())
    }

    fn free_slot(&mut self, handle: ObjectHandle) -> Result<Bytes, MemoryError> {
        let index = handle.slot as usize;
        let Some(entry) = self.slots.get_mut(index) else {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: 0,
            });
        };
        if entry.generation != handle.generation || !entry.live {
            return Err(MemoryError::StaleHandle {
                handle,
                expected: handle.generation,
                found: entry.generation,
            });
        }
        let payload = entry.payload.take().ok_or(MemoryError::StaleHandle {
            handle,
            expected: handle.generation,
            found: entry.generation,
        })?;
        entry.live = false;
        // Bump generation so the freed handle immediately goes stale.
        entry.generation = entry.generation.wrapping_add(1).max(1);
        self.live_count = self.live_count.saturating_sub(1);
        self.live_bytes = self.live_bytes.saturating_sub(payload.len() as u64);
        self.free.push(handle.slot);
        Ok(payload)
    }

    /// Compacts by dropping freed slots from the free list tail when the
    /// arena is fragmented: rebuilds the slot vector without dead entries
    /// and returns the remapping for live handles the caller must
    /// publish. In practice the fabric drives this during idle budget
    /// windows and publishes the remap atomically with the object table.
    fn compact(&mut self) -> Vec<(ObjectHandle, ObjectHandle)> {
        if self.free.is_empty() {
            return Vec::new();
        }
        // Only compact when fragmentation is material (>25% dead).
        let dead = self.free.len();
        let total = self.slots.len();
        if total == 0 || dead * 4 < total {
            return Vec::new();
        }
        let mut remap = Vec::new();
        let mut new_slots: Vec<Slot> = Vec::with_capacity(self.live_count);
        let mut old_to_new = vec![u32::MAX; total];
        for (old_index, slot) in self.slots.iter().enumerate() {
            if slot.live {
                // Slot counts stay below u32::MAX (allocation enforces
                // the bound), so saturation here is unreachable.
                old_to_new[old_index] = u32::try_from(new_slots.len()).unwrap_or(u32::MAX);
                new_slots.push(slot.clone());
            }
        }
        for (old_index, new_index) in old_to_new.iter().enumerate() {
            if *new_index != u32::MAX {
                let generation = new_slots[*new_index as usize].generation;
                // See above: indices stay below u32::MAX.
                let old_handle =
                    ObjectHandle::new(u32::try_from(old_index).unwrap_or(u32::MAX), generation);
                remap.push((old_handle, ObjectHandle::new(*new_index, generation)));
            }
        }
        self.slots = new_slots;
        self.free.clear();
        remap
    }

    const fn stats(&self) -> ArenaStats {
        ArenaStats {
            class: self.class,
            slots: self.slots.len(),
            live_count: self.live_count,
            live_bytes: self.live_bytes,
            free_count: self.free.len(),
        }
    }
}

/// Per-class statistics for observability and calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaStats {
    /// Which class these numbers describe.
    pub class: BehaviorClass,
    /// Total slots ever allocated (including dead).
    pub slots: usize,
    /// Live objects.
    pub live_count: usize,
    /// Live bytes.
    pub live_bytes: u64,
    /// Free slots available for reuse.
    pub free_count: usize,
}

/// Worker-local set of behavior-aware arenas.
///
/// Single-owner: `&mut self` methods run on the owning worker; `&self`
/// reads (handle resolution) never mutate. No locks, no atomics.
#[derive(Debug)]
pub struct WorkerArenas {
    arenas: [ClassArena; 7],
    capacity_bytes: u64,
}

impl WorkerArenas {
    /// Creates an empty arena set with the given DRAM capacity bound.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            arenas: [
                ClassArena::new(BehaviorClass::HotMutable),
                ClassArena::new(BehaviorClass::HotReadMostly),
                ClassArena::new(BehaviorClass::Warm),
                ClassArena::new(BehaviorClass::ColdCandidate),
                ClassArena::new(BehaviorClass::ShortLived),
                ClassArena::new(BehaviorClass::LongLivedMetadata),
                ClassArena::new(BehaviorClass::LargeRootMetadata),
            ],
            capacity_bytes,
        }
    }

    /// Returns the index of a behavior class in the arena array.
    const fn index_of(class: BehaviorClass) -> usize {
        match class {
            BehaviorClass::HotMutable => 0,
            BehaviorClass::HotReadMostly => 1,
            BehaviorClass::Warm => 2,
            BehaviorClass::ColdCandidate => 3,
            BehaviorClass::ShortLived => 4,
            BehaviorClass::LongLivedMetadata => 5,
            BehaviorClass::LargeRootMetadata => 6,
        }
    }

    /// Allocates a payload in the given behavior class.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::TooLarge`] when the payload exceeds the
    /// medium-object ceiling, or [`MemoryError::Overloaded`] when the
    /// arena set exceeds its capacity bound.
    pub fn alloc(
        &mut self,
        class: BehaviorClass,
        payload: Bytes,
    ) -> Result<ObjectHandle, MemoryError> {
        if self.used_bytes().saturating_add(payload.len() as u64) > self.capacity_bytes {
            return Err(MemoryError::Overloaded {
                queue: "arena capacity",
            });
        }
        self.arenas[Self::index_of(class)].alloc(payload)
    }

    /// Resolves a handle to its resident bytes.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::StaleHandle`] when the handle's generation
    /// does not match (relocated, compacted, or freed).
    pub fn get(&self, class: BehaviorClass, handle: ObjectHandle) -> Result<Bytes, MemoryError> {
        self.arenas[Self::index_of(class)].get(handle)
    }

    /// Replaces the bytes under a live handle (same slot, same
    /// generation). Used for in-place mutation of DRAM-resident objects.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::StaleHandle`] for dead handles.
    pub fn update(
        &mut self,
        class: BehaviorClass,
        handle: ObjectHandle,
        payload: Bytes,
    ) -> Result<(), MemoryError> {
        self.arenas[Self::index_of(class)].update(handle, payload)
    }

    /// Frees a handle, returning its bytes for migration. The handle goes
    /// stale immediately.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::StaleHandle`] for dead handles.
    pub fn free(
        &mut self,
        class: BehaviorClass,
        handle: ObjectHandle,
    ) -> Result<Bytes, MemoryError> {
        self.arenas[Self::index_of(class)].free_slot(handle)
    }

    /// Compacts fragmented classes, returning live-handle remaps the
    /// caller must publish atomically. Empty when nothing material.
    pub fn compact(&mut self) -> Vec<(BehaviorClass, ObjectHandle, ObjectHandle)> {
        let mut out = Vec::new();
        for arena in &mut self.arenas {
            let class = arena.class;
            for (old, new) in arena.compact() {
                out.push((class, old, new));
            }
        }
        out
    }

    /// Total live bytes across all classes.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.arenas.iter().map(|arena| arena.live_bytes).sum()
    }

    /// Total live objects across all classes.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.arenas.iter().map(ClassArena::live_count).sum()
    }

    /// Configured capacity bound.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Pressure in `[0, 1]`: used / capacity, saturating at 1.0.
    ///
    /// Byte counts are far below 2^53, so float conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn pressure(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 1.0;
        }
        (self.used_bytes() as f64 / self.capacity_bytes as f64).clamp(0.0, 1.0)
    }

    /// Per-class statistics.
    #[must_use]
    pub fn stats(&self) -> [ArenaStats; 7] {
        [
            self.arenas[0].stats(),
            self.arenas[1].stats(),
            self.arenas[2].stats(),
            self.arenas[3].stats(),
            self.arenas[4].stats(),
            self.arenas[5].stats(),
            self.arenas[6].stats(),
        ]
    }
}

impl ClassArena {
    const fn live_count(&self) -> usize {
        self.live_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_get_update_free_roundtrip() {
        let mut arenas = WorkerArenas::new(1 << 20);
        let handle = arenas
            .alloc(BehaviorClass::HotMutable, Bytes::from_static(b"hello"))
            .expect("alloc");
        assert_eq!(
            arenas
                .get(BehaviorClass::HotMutable, handle)
                .expect("get")
                .as_ref(),
            b"hello"
        );
        arenas
            .update(
                BehaviorClass::HotMutable,
                handle,
                Bytes::from_static(b"world!"),
            )
            .expect("update");
        assert_eq!(
            arenas
                .get(BehaviorClass::HotMutable, handle)
                .expect("get")
                .as_ref(),
            b"world!"
        );
        let bytes = arenas
            .free(BehaviorClass::HotMutable, handle)
            .expect("free");
        assert_eq!(bytes.as_ref(), b"world!");
        let stale = arenas.get(BehaviorClass::HotMutable, handle);
        assert!(matches!(stale, Err(MemoryError::StaleHandle { .. })));
    }

    #[test]
    fn freed_slots_are_reused_with_new_generations() {
        let mut arenas = WorkerArenas::new(1 << 20);
        let first = arenas
            .alloc(BehaviorClass::Warm, Bytes::from_static(b"a"))
            .expect("alloc");
        arenas.free(BehaviorClass::Warm, first).expect("free");
        let second = arenas
            .alloc(BehaviorClass::Warm, Bytes::from_static(b"b"))
            .expect("alloc");
        assert_eq!(first.slot, second.slot);
        assert_ne!(first.generation, second.generation);
        assert!(matches!(
            arenas.get(BehaviorClass::Warm, first),
            Err(MemoryError::StaleHandle { .. })
        ));
    }

    #[test]
    fn classes_do_not_intermingle_accounting() {
        let mut arenas = WorkerArenas::new(1 << 20);
        arenas
            .alloc(BehaviorClass::HotMutable, Bytes::from_static(b"hot"))
            .expect("alloc");
        arenas
            .alloc(BehaviorClass::ColdCandidate, Bytes::from_static(b"cold"))
            .expect("alloc");
        let stats = arenas.stats();
        assert_eq!(stats[0].live_count, 1);
        assert_eq!(stats[3].live_count, 1);
        assert_eq!(stats[1].live_count, 0);
    }
}
