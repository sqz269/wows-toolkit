//! Process-wide record of which game files and assets.bin prototype records a
//! command read.
//!
//! Recording is off by default and costs one relaxed atomic load per read.
//! A caller turns it on with [`enable`] before running a command, then calls
//! [`take`] afterwards to get the de-duplicated set. The CLI uses this for
//! `--deps-out`: an export can store exactly which inputs it consumed, so a
//! later game build can be checked for changes to just those inputs.
//!
//! Two kinds of reads are recorded:
//! - VFS file opens, by VFS path (`/content/.../foo.geometry`). Both IDX/PKG
//!   files and the virtual files synthesised from assets.bin go through here.
//! - Direct prototype-record lookups on a parsed `PrototypeDatabase`, by
//!   `(blob_index, record_index)`. Code that parses assets.bin itself and
//!   indexes records never opens a VFS path, so those reads are caught here
//!   and mapped back to paths by the caller.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<DepsState> = Mutex::new(DepsState { files: BTreeSet::new(), prototypes: BTreeSet::new() });

#[derive(Debug, Default, Clone)]
pub struct DepsState {
    /// VFS paths opened as files.
    pub files: BTreeSet<String>,
    /// `(blob_index, record_index)` pairs read from a `PrototypeDatabase`.
    pub prototypes: BTreeSet<(usize, usize)>,
}

/// Start recording reads for the rest of the process (until [`disable`]).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Stop recording. Already-recorded reads are kept until [`take`].
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Record that a VFS file was opened.
#[inline]
pub fn record_file(path: &str) {
    if !is_enabled() {
        return;
    }
    let normalized = if path.starts_with('/') { path.to_string() } else { format!("/{path}") };
    if let Ok(mut state) = STATE.lock() {
        state.files.insert(normalized);
    }
}

/// Record that a prototype record was read straight from a parsed database.
#[inline]
pub fn record_prototype(blob_index: usize, record_index: usize) {
    if !is_enabled() {
        return;
    }
    if let Ok(mut state) = STATE.lock() {
        state.prototypes.insert((blob_index, record_index));
    }
}

/// Return everything recorded so far and clear it.
pub fn take() -> DepsState {
    match STATE.lock() {
        Ok(mut state) => std::mem::take(&mut *state),
        Err(_) => DepsState::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_only_while_enabled() {
        // Single test so the process-wide state isn't shared across threads.
        let _ = take();
        record_file("/content/ignored.geometry");
        assert!(take().files.is_empty());

        enable();
        record_file("content/a.geometry");
        record_file("/content/a.geometry");
        record_prototype(1, 7);
        record_prototype(1, 7);
        disable();
        record_file("/content/after.geometry");

        let state = take();
        assert_eq!(state.files.into_iter().collect::<Vec<_>>(), vec!["/content/a.geometry".to_string()]);
        assert_eq!(state.prototypes.into_iter().collect::<Vec<_>>(), vec![(1, 7)]);
        assert!(take().files.is_empty());
    }
}
