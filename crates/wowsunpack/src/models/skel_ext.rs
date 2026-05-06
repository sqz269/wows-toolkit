//! Parser for `.skel_ext` files — per-hull-segment skeleton extension data
//! containing decorative prop placements (ventilators, hatches, bollards, fire
//! gear, life buoys, etc.).
//!
//! A WoWS ship has 8 `.skel_ext` files (4 base + 4 `_ep` variants) under its
//! model directory, e.g.:
//! `content/gameplay/usa/ship/battleship/ASB017_Montana_1945/
//!  ASB017_Montana_1945_MidFront_ep.skel_ext`
//!
//! These are referenced at runtime via `ModelPrototype.skel_ext_res_ids` (u64
//! selfIds) but the selfIds point at `SkeletonExtenderPrototype` records in
//! `assets.bin` blob 2. The files themselves live in the VFS.
//!
//! See `tools/reference/skel_ext_findings.md` in the pipeline repo for a full
//! format spec. Summary:
//!
//! ```text
//! +0x00000000  Record array @ 0x20 stride
//!   Per record (0x20 bytes):
//!     +0x00  u16 type      ∈ {0, 1}; semantics unclear
//!     +0x02  u16 count     number of entries in p0/p1
//!     +0x04  u32 pad       always zero
//!     +0x08  u64 p0        offset → u32[count] (opaque per-placement hash)
//!     +0x10  u64 p1        offset → u32[count] (parent-bone hash; Murmur3_32
//!                                               seed=0, e.g. "Scene Root"
//!                                               = 0x10C30510)
//!     +0x18  u64 p2        offset → (count-1) × 64B matrices + 32B trailer
//!
//! +~0x9060   Record array ends. Variable per-file.
//!
//! +~0x9060 .. ~0x26980E0   40 MB middle payload (likely merged mesh data,
//!                          not referenced by top record array — skip).
//!
//! +~0x26980E0 .. EOF       Payload for records: p0/p1 u32 arrays +
//!                          p2 matrix arrays (count-1 full 4×4 f32 column-major
//!                          + a 32-byte trailer per record).
//! ```
//!
//! # What this parser returns
//!
//! Every well-formed affine 4×4 matrix from every record, with metadata
//! (record offset/type/count, per-placement p0 and p1 hashes). The matrix is
//! raw WG-native — **left-handed, 1 unit = 15 m**. Apply
//! `negate_z_transform` and multiply the translation by 15 to get metric
//! glTF/Unity-ready transforms (see [`negate_z_transform`]).
//!
//! Each ship produces ~100k matrices across 8 files. Only a fraction are
//! actual top-level placements (the legacy pipeline found 696 on Montana);
//! the rest are per-LOD / per-variant duplicates + per-bone local transforms.
//! Downstream consumers are responsible for filtering / deduplication.
//! Dedupe helpers are exported below.

use std::collections::HashMap;

use thiserror::Error;

/// Record stride in the header record array (bytes).
pub const RECORD_STRIDE: usize = 0x20;

/// Matrix stride in the p2 payload (bytes): 4×4 f32 column-major.
pub const MATRIX_SIZE: usize = 64;

/// Size of the unexplained trailer at the end of each record's p2 region.
pub const TRAILER_SIZE: usize = 32;

/// Upper bound on bytes to scan for records at file start. Montana's base
/// files end around 0x9060, `_ep` variants around 0x1FE80 (~4080 records ×
/// 0x20). 0x20000 is a tight upper bound matching the Python reference; bigger
/// values pull in spurious header-looking bytes from the payload region.
pub const HEADER_SCAN_LIMIT: usize = 0x20000;

/// Max plausible record count. The largest seen (Montana `_ep` files) is
/// ~474. Guard rail against runaway parsing of bogus data.
pub const MAX_RECORD_COUNT: u16 = 10_000;

/// MurmurHash3_32(seed=0) of "Scene Root". The most common p1 hash,
/// identifying placements attached directly to the ship's hull root.
pub const SCENE_ROOT_HASH: u32 = 0x10C3_0510;

#[derive(Debug, Error)]
pub enum SkelExtError {
    #[error("skel_ext file too short ({size} bytes, need >= {RECORD_STRIDE})")]
    TooShort { size: usize },
    #[error("no valid records found in header")]
    NoRecords,
}

/// One 0x20-byte record from the start of a `.skel_ext` file.
///
/// The record itself is a header; actual content lives in the
/// (p0, p1, p2) arrays pointed at by the u64 offsets.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SkelExtRecord {
    /// Offset of this record within the `.skel_ext` file (debug aid).
    pub file_offset: u64,
    /// 0 or 1; exact meaning TBD.
    pub type_: u16,
    /// Number of u32 entries in p0 and p1 arrays. p2 holds `count - 1`
    /// full matrices + 32 B trailer.
    pub count: u16,
    /// File offset → u32[count] (opaque per-placement identifier).
    pub p0: u64,
    /// File offset → u32[count] (parent-bone Murmur3_32 hash).
    pub p1: u64,
    /// File offset → `(count-1) × 64 B` matrices + 32 B trailer.
    pub p2: u64,
}

/// One affine placement recovered from a record's p2 array.
///
/// The `matrix` is column-major, in WG native left-handed coordinates,
/// with 1 unit = 15 m. Convert to glTF/Unity space before using:
/// apply [`negate_z_transform`] and scale translation by 15.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SkelExtPlacement {
    /// For debugging: which record in the file it came from.
    pub record_offset: u64,
    /// Record type field (0 or 1).
    pub record_type: u16,
    /// Record count field.
    pub record_count: u16,
    /// Index within the record's matrix array: 0..(count-1).
    pub matrix_index: u32,
    /// Per-placement identifier hash.
    ///
    /// Reversed 2026-05-06:
    /// `p0_hash = Murmur3_x86_32(seed=0, "MP_<asset_id>[_INDEX_<n>]")`
    /// where `MP_` is the WG content-pipeline "Mesh Placement" prefix and
    /// `_INDEX_<n>` (1-based) disambiguates multiple instances of the same
    /// asset on a hull section. Some Maya/Blender duplicates use `.001` /
    /// `.002` suffixes instead.
    ///
    /// Verified end-to-end on ARP TAKAO BLUE/RED:
    /// `MP_JM743_Searchlight_Red_Arpeggio` → `0xCA841EE4` (in RED skel_ext);
    /// `MP_JM501_Searchlight_Arpeggio`     → `0x8F5530CF` (in BLUE skel_ext).
    ///
    /// Resolves the asset_id from a placement without needing a legacy
    /// gmconvert scan: hash `MP_<asset_id>[_INDEX_<n>]` for every asset_id
    /// in the VFS misc / gun / director / finder / radar / catapult corpus
    /// and intersect with the observed p0 set.
    ///
    /// See `variant_ship_accessory_swap.md` in the pipeline repo for the
    /// crack + empirical proof.
    pub p0_hash: u32,
    /// Parent skeleton node hash. Murmur3_32(seed=0) of the hull-skeleton
    /// bone name. E.g. [`SCENE_ROOT_HASH`] for "Scene Root".
    pub p1_hash: u32,
    /// 4×4 column-major affine transform in WG native coords.
    pub matrix: [f32; 16],
}

impl SkelExtPlacement {
    /// World position in native coords (m[12..14]).
    pub fn position(&self) -> [f32; 3] {
        [self.matrix[12], self.matrix[13], self.matrix[14]]
    }
}

fn read_u16_le(buf: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buf.get(off..off + 2)?.try_into().ok()?))
}
fn read_u32_le(buf: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(off..off + 4)?.try_into().ok()?))
}
fn read_u64_le(buf: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(buf.get(off..off + 8)?.try_into().ok()?))
}
fn read_f32_le(buf: &[u8], off: usize) -> Option<f32> {
    Some(f32::from_le_bytes(buf.get(off..off + 4)?.try_into().ok()?))
}

/// Parse the record array at the file start.
///
/// Scans 0x20-byte slots up to [`HEADER_SCAN_LIMIT`] (or end-of-file),
/// accepting entries that look plausibly like valid records. Records
/// whose `file_offset >= min(p0)` are discarded — those sit within the
/// payload region and aren't part of the record table.
pub fn parse_records(buf: &[u8]) -> Vec<SkelExtRecord> {
    let size = buf.len();
    if size < RECORD_STRIDE {
        return Vec::new();
    }

    let mut recs = Vec::new();
    let scan_limit = size.min(HEADER_SCAN_LIMIT);
    let mut off = 0usize;

    while off + RECORD_STRIDE <= scan_limit {
        let t = read_u16_le(buf, off).unwrap();
        let c = read_u16_le(buf, off + 2).unwrap();
        let _pad = read_u32_le(buf, off + 4).unwrap();
        let p0 = read_u64_le(buf, off + 8).unwrap();
        let p1 = read_u64_le(buf, off + 16).unwrap();
        let p2 = read_u64_le(buf, off + 24).unwrap();

        let valid = (t == 0 || t == 1)
            && c >= 1
            && c <= MAX_RECORD_COUNT
            && p0 > 0
            && (p0 as usize) < size
            && p1 > 0
            && (p1 as usize) < size
            && p2 > 0
            && (p2 as usize) < size;

        if valid {
            recs.push(SkelExtRecord {
                file_offset: off as u64,
                type_: t,
                count: c,
                p0,
                p1,
                p2,
            });
        }
        off += RECORD_STRIDE;
    }

    if recs.is_empty() {
        return recs;
    }

    // Records whose file_offset is at/past the first payload address are
    // phantom "records" caught inside the payload data. Drop them.
    let first_payload = recs.iter().map(|r| r.p0).min().unwrap_or(u64::MAX);
    recs.retain(|r| r.file_offset < first_payload);
    recs
}

/// True if a column-major 4×4 has row 3 ≈ (0, 0, 0, 1) — i.e., a valid
/// affine transform. Non-affine matrices in this data are typically
/// skeleton-local or skin-weight data, not world placements.
pub fn is_affine(m: &[f32; 16]) -> bool {
    m[3].abs() < 1e-3 && m[7].abs() < 1e-3 && m[11].abs() < 1e-3 && (m[15] - 1.0).abs() < 1e-3
}

/// Read a 4×4 f32 matrix (column-major) from a byte offset.
fn read_matrix(buf: &[u8], off: usize) -> Option<[f32; 16]> {
    if off + MATRIX_SIZE > buf.len() {
        return None;
    }
    let mut m = [0f32; 16];
    for i in 0..16 {
        m[i] = read_f32_le(buf, off + i * 4)?;
    }
    Some(m)
}

/// Parse all affine placements from a `.skel_ext` file.
///
/// For each parsed record, reads the `count - 1` full matrices at p2 (skipping
/// the trailing 32 B) and returns those that pass [`is_affine`]. Each
/// placement also carries the per-index p0 and p1 u32 values from the
/// record's arrays.
pub fn parse_skel_ext(buf: &[u8]) -> Result<Vec<SkelExtPlacement>, SkelExtError> {
    if buf.len() < RECORD_STRIDE {
        return Err(SkelExtError::TooShort { size: buf.len() });
    }
    let recs = parse_records(buf);
    if recs.is_empty() {
        return Err(SkelExtError::NoRecords);
    }

    let mut out = Vec::new();
    for r in &recs {
        let c = r.count as usize;
        if c < 2 {
            continue;
        }
        let p0_off = r.p0 as usize;
        let p1_off = r.p1 as usize;
        let p2_off = r.p2 as usize;

        // Bounds-check each array. If any would overflow the buffer, skip
        // the whole record (better to drop questionable data than panic).
        if p0_off + c * 4 > buf.len()
            || p1_off + c * 4 > buf.len()
            || p2_off + (c - 1) * MATRIX_SIZE > buf.len()
        {
            continue;
        }

        for i in 0..(c - 1) {
            let m_off = p2_off + i * MATRIX_SIZE;
            let Some(m) = read_matrix(buf, m_off) else {
                continue;
            };
            if !is_affine(&m) {
                continue;
            }
            let p0_hash = read_u32_le(buf, p0_off + i * 4).unwrap_or(0);
            let p1_hash = read_u32_le(buf, p1_off + i * 4).unwrap_or(0);
            out.push(SkelExtPlacement {
                record_offset: r.file_offset,
                record_type: r.type_,
                record_count: r.count,
                matrix_index: i as u32,
                p0_hash,
                p1_hash,
                matrix: m,
            });
        }
    }
    Ok(out)
}

/// Coordinate conversion: WG native (left-handed) → glTF/Unity (right-handed).
///
/// Mirrors the convention used elsewhere in the toolkit (see
/// `export::gltf_export::negate_z_transform`): negates the Z basis and the Z
/// translation, preserving X and Y. After this, multiply translation
/// components by 15 to get metres.
pub fn negate_z_transform(m: [f32; 16]) -> [f32; 16] {
    [
        m[0], m[1], -m[2], m[3], // col 0
        m[4], m[5], -m[6], m[7], // col 1
        -m[8], -m[9], m[10], m[11], // col 2
        m[12], m[13], -m[14], m[15], // col 3 (translation)
    ]
}

/// WG native → metres multiplier (1 unit = 15 m).
pub const NATIVE_TO_METRES: f32 = 15.0;

/// Apply `negate_z_transform` and scale translation to metres.
///
/// The resulting matrix is column-major, right-handed, with translation in
/// metres — the same convention the toolkit uses for `HP_*` mount placements.
pub fn to_metric_glft(m: [f32; 16]) -> [f32; 16] {
    let mut out = negate_z_transform(m);
    out[12] *= NATIVE_TO_METRES;
    out[13] *= NATIVE_TO_METRES;
    out[14] *= NATIVE_TO_METRES;
    out
}

/// Deduplicate placements by quantized world-position.
///
/// Groups placements whose rounded `(x, y, z)` position (at `quantum_m`
/// precision, native-coord) matches and keeps one representative per group
/// (the first encountered, which is stable given record iteration order).
///
/// Montana has ~100k affine matrices per ship; each real placement is
/// duplicated ~40–60× for different LODs and render-set variants. Position
/// dedupe at 1 mm (quantum=0.001) gets ~58k unique; at 1 cm ~57k; at 10 cm
/// ~23k. Choose based on how tight the downstream position match needs to be.
pub fn dedupe_by_position(placements: &[SkelExtPlacement], quantum_m: f32) -> Vec<SkelExtPlacement> {
    let mut seen: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut out = Vec::new();
    for pl in placements {
        let p = pl.position();
        let key = (
            (p[0] / quantum_m).round() as i64,
            (p[1] / quantum_m).round() as i64,
            (p[2] / quantum_m).round() as i64,
        );
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
            e.insert(out.len());
            out.push(pl.clone());
        }
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_errors() {
        assert!(matches!(
            parse_skel_ext(&[]),
            Err(SkelExtError::TooShort { .. })
        ));
    }

    #[test]
    fn garbage_buffer_has_no_records() {
        // A 1 KB buffer of zeros — no valid records should parse.
        let buf = vec![0u8; 1024];
        let res = parse_skel_ext(&buf);
        assert!(matches!(res, Err(SkelExtError::NoRecords)));
    }

    #[test]
    fn is_affine_rejects_bad_bottom_row() {
        // Column-major: row 3 is at indices 3, 7, 11, 15.
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        assert!(is_affine(&identity));
        let mut bad = identity;
        bad[15] = 0.5; // row 3 col 3 should be 1.0
        assert!(!is_affine(&bad));
        let mut bad2 = identity;
        bad2[3] = 0.5; // row 3 col 0 should be 0
        assert!(!is_affine(&bad2));
    }

    #[test]
    fn negate_z_transform_is_involution_on_rotation_free() {
        let m = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let flipped = negate_z_transform(m);
        assert_eq!(flipped[14], -3.0);
        assert_eq!(flipped[10], 1.0); // col 2 row 2: -(-1) = 1 (from -m[8]=-0, -m[9]=-0, m[10]=1)
        assert_eq!(negate_z_transform(flipped), m);
    }

    #[test]
    fn to_metric_scales_translation() {
        let m = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let metric = to_metric_glft(m);
        assert_eq!(metric[12], 15.0);
        assert_eq!(metric[13], 30.0);
        assert_eq!(metric[14], -45.0); // Z negated then ×15
    }
}
