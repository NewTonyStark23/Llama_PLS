// VeloxLLM Arena Allocator
// Zero-allocation memory management for the inference hot path.
// All memory is pre-allocated and recycled - no malloc/free during inference.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// Stack-allocated arena that never calls malloc/free during inference.
/// Uses a linear allocation strategy with O(1) reset.
pub struct Arena<const SIZE: usize> {
    buffer: UnsafeCell<[MaybeUninit<u8>; SIZE]>,
    offset: UnsafeCell<usize>,
    watermark: UnsafeCell<usize>,
}

unsafe impl<const SIZE: usize> Send for Arena<SIZE> {}
unsafe impl<const SIZE: usize> Sync for Arena<SIZE> {}

impl<const SIZE: usize> Arena<SIZE> {
    /// Create a new arena. Memory is uninitialized (no zeroing overhead).
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(unsafe { MaybeUninit::uninit().assume_init() }),
            offset: UnsafeCell::new(0),
            watermark: UnsafeCell::new(0),
        }
    }

    /// Allocate a typed slice from the arena. O(1), no system calls.
    /// Panics if arena is exhausted.
    #[inline(always)]
    pub fn alloc<T>(&self, len: usize) -> &mut [T] {
        let size = len * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>().max(16); // 16-byte alignment for SIMD

        unsafe {
            let offset = &mut *self.offset.get();
            // Align up to required alignment
            let aligned = (*offset + align - 1) & !(align - 1);
            let new_offset = aligned + size;

            if new_offset > SIZE {
                panic!(
                    "Arena exhausted: requested {} bytes, available {} bytes",
                    size,
                    SIZE - *offset
                );
            }

            let buffer = &mut *self.buffer.get();
            let ptr = buffer.as_mut_ptr().add(aligned) as *mut T;
            *offset = new_offset;

            // Update high-water mark
            let wm = &mut *self.watermark.get();
            if new_offset > *wm {
                *wm = new_offset;
            }

            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Allocate a single value
    #[inline(always)]
    pub fn alloc_one<T>(&self, val: T) -> &mut T {
        let slice = self.alloc::<T>(1);
        slice[0] = val;
        &mut slice[0]
    }

    /// Reset the arena - O(1), just resets the offset.
    /// All previous allocations become invalid.
    #[inline(always)]
    pub fn reset(&self) {
        unsafe {
            *self.offset.get() = 0;
        }
    }

    /// Get current usage in bytes
    pub fn used(&self) -> usize {
        unsafe { *self.offset.get() }
    }

    /// Get high-water mark (peak usage)
    pub fn watermark(&self) -> usize {
        unsafe { *self.watermark.get() }
    }

    /// Get total capacity
    pub const fn capacity(&self) -> usize {
        SIZE
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        SIZE - self.used()
    }
}

impl<const SIZE: usize> Default for Arena<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped arena borrow - restores offset when dropped
pub struct ArenaBorrow<'a, const SIZE: usize> {
    arena: &'a Arena<SIZE>,
    saved_offset: usize,
}

impl<const SIZE: usize> Arena<SIZE> {
    /// Create a scoped borrow. When dropped, restores the offset.
    pub fn borrow(&self) -> ArenaBorrow<'_, SIZE> {
        ArenaBorrow {
            arena: self,
            saved_offset: unsafe { *self.offset.get() },
        }
    }
}

impl<'a, const SIZE: usize> Drop for ArenaBorrow<'a, SIZE> {
    fn drop(&mut self) {
        unsafe {
            *self.arena.offset.get() = self.saved_offset;
        }
    }
}

// Common arena sizes for LLM inference
pub type InferenceArena = Arena<{ 256 * 1024 }>; // 256 KB for small models
pub type LargeArena = Arena<{ 4 * 1024 * 1024 }>; // 4 MB for large models

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let arena = Arena::<1024>::new();
        let slice = arena.alloc::<f32>(10);
        assert_eq!(slice.len(), 10);
        assert_eq!(arena.used(), 40 + 16); // 10 * 4 bytes + alignment padding
    }

    #[test]
    fn test_reset() {
        let arena = Arena::<1024>::new();
        let _ = arena.alloc::<f32>(10);
        assert!(arena.used() > 0);
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_watermark() {
        let arena = Arena::<1024>::new();
        let _ = arena.alloc::<f32>(10);
        let wm = arena.watermark();
        arena.reset();
        assert_eq!(arena.watermark(), wm); // Watermark persists after reset
    }
}
