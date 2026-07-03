//! Parser for space-level `forest.bin` (SpeedTree vegetation placement) files.
//!
//! These files store per-instance placement data for SpeedTree vegetation
//! (trees, bushes, algae, etc.) across a map space.
//!
//! ## File Layout (engine-verified, 2026-05-22 audit)
//!
//! ```text
//! Header (16 bytes):
//!   u32 num_species + 4-byte pad
//!   i64 string_table_relptr  (relative to file base; lands at abs 0x250 in all observed files)
//!
//! Layers (4 × 0x90 bytes, starting at file offset 0x10):
//!   Layer[0] @ 0x10   = "Primary"        (LOD-0 above-water placements)
//!   Layer[1] @ 0xA0   = "PrimaryRare"
//!   Layer[2] @ 0x130  = "Underwater"
//!   Layer[3] @ 0x1C0  = "UnderwaterRare"
//!
//!   Per-layer struct (0x90 bytes):
//!     +0x00  i64  buffer_relptr   (relative to layer base; points to instance data)
//!     +0x08  u32  buffer_count    (number of f32x4 instance records)
//!     +0x0c  u32  pad/unknown     (always 0 in observed files)
//!     +0x10  cell[0] (0x80 bytes):
//!       num_species × (u32 start, u32 count)  // per-species (start, count) ranges
//!       ...rest of cell is zero padding to 0x80
//!
//! String Table (@ string_table_relptr; abs 0x250):
//!   num_species × (u64 len, i64 relptr)  // entries
//!   ...followed by null-terminated string data...
//!
//! Instance Data (per-layer; reached via Layer.buffer_relptr):
//!   Dense array of 16-byte records (f32 x, f32 y, f32 z, f32 w).
//!   Layer 0's records typically start right after the string pool.
//! ```
//!
//! Engine: `forest::ForestSystemImpl::loadForestData` @ 0x140fdddc0 →
//! root parser `FUN_140fe2240` → per-layer parser `FUN_140fe24c0`. Layer
//! label strings live at `.rdata` `0x14251c690ff`.
//!
//! All four layers parse with the identical record shape (verified across
//! all 70 non-empty corpus files + the engine's literal 4-name loop; see
//! reference/maps/grounding_2026_07_03/vegetation_tint_layers.md).
//! Semantics: Primary/PrimaryRare = above-water (y >= 0); Underwater/
//! UnderwaterRare = algae (y <= 0). "Rare" layers are decimated
//! far-density companions of their base layer (an exact copy on small
//! maps, ~45-55% thinned on large) — consumers should draw base OR rare
//! per distance, never both additively.

use rootcause::Report;
use thiserror::Error;
use winnow::Parser;
use winnow::binary::le_f32;
use winnow::binary::le_i64;
use winnow::binary::le_u64;
use winnow::combinator::repeat;

use winnow::error::ContextError;
use winnow::error::ErrMode;

use crate::data::parser_utils::WResult;
use crate::data::parser_utils::resolve_relptr;

const INSTANCE_SIZE: usize = 16;

#[derive(Debug, Error)]
pub enum ForestError {
    #[error("data too short: need {need} bytes at offset 0x{offset:X}, have {have}")]
    DataTooShort { offset: usize, need: usize, have: usize },
    #[error("parse error: {0}")]
    ParseError(String),
}

/// A single vegetation instance with its species assignment.
#[derive(Debug, Clone, Copy)]
pub struct ForestInstance {
    pub species_index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Layer names in file order (engine-fixed).
pub const LAYER_NAMES: [&str; 4] = ["Primary", "PrimaryRare", "Underwater", "UnderwaterRare"];

/// Parsed `forest.bin` file — all four placement layers.
#[derive(Debug)]
pub struct Forest {
    /// SpeedTree species asset paths (`.stsdk` files), shared by all layers.
    pub species: Vec<String>,
    /// Per-layer vegetation instances, indexed per [`LAYER_NAMES`].
    pub layers: [Vec<ForestInstance>; 4],
}

/// Parse a single string table entry: `(u64 len, i64 relptr)`.
fn parse_string_table_entry(input: &mut &[u8]) -> WResult<(u64, i64)> {
    let len = le_u64.parse_next(input)?;
    let relptr = le_i64.parse_next(input)?;
    Ok((len, relptr))
}

/// Raw instance record from the file.
struct RawInstance {
    x: f32,
    y: f32,
    z: f32,
}

fn parse_raw_instance(input: &mut &[u8]) -> WResult<RawInstance> {
    let x = le_f32.parse_next(input)?;
    let y = le_f32.parse_next(input)?;
    let z = le_f32.parse_next(input)?;
    let _w = le_f32.parse_next(input)?;
    Ok(RawInstance { x, y, z })
}

// Layer offsets in the file (4 layers, 0x90 bytes each, starting at 0x10).
// Cell[0] inside each layer holds the per-species (u32 start, u32 count) table.
const LAYER_BASE: usize = 0x10;
const LAYER_STRIDE: usize = 0x90;
const CELL_OFFSET_IN_LAYER: usize = 0x10;

/// Read one layer struct from its engine-defined fixed offset.
///
/// Returns `(instance_data_abs_offset, instance_count, per_species_ranges)`.
///
/// The original implementation byte-grepped for a
/// `(num_species, 0, num_species, 1)` marker, which failed on 70/70
/// non-empty corpus files (the marker hits a false positive in 1/82 and
/// never matches the real table location). Engine ground truth from
/// `FUN_140fe24c0` (per-layer parser): the (start, count) table lives at
/// a fixed offset (`layer_base + 0x10`) inside the 0x90-byte layer
/// struct, and the instance buffer pointer lives at the layer base
/// itself.
fn read_layer(
    data: &[u8],
    layer_idx: usize,
    num_species: usize,
) -> Result<(usize, usize, Vec<(usize, usize)>), ForestError> {
    let layer_off = LAYER_BASE + layer_idx * LAYER_STRIDE;
    let cell_off = layer_off + CELL_OFFSET_IN_LAYER;
    let need = cell_off + num_species * 8;
    if data.len() < need {
        return Err(ForestError::DataTooShort { offset: layer_off, need, have: data.len() });
    }

    let buf_relptr = i64::from_le_bytes(data[layer_off..layer_off + 8].try_into().unwrap());
    let buf_count = u32::from_le_bytes(data[layer_off + 8..layer_off + 12].try_into().unwrap()) as usize;
    // buffer_relptr is relative to the layer base; resolve to abs file offset.
    let buf_abs = (layer_off as i64).wrapping_add(buf_relptr) as usize;
    if buf_count > 0 && buf_abs + buf_count * INSTANCE_SIZE > data.len() {
        return Err(ForestError::ParseError(format!(
            "Primary layer buffer out of bounds: abs=0x{buf_abs:X} count={buf_count} file_len={}",
            data.len(),
        )));
    }

    let mut ranges = Vec::with_capacity(num_species);
    for i in 0..num_species {
        let o = cell_off + i * 8;
        let start = u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(data[o + 4..o + 8].try_into().unwrap()) as usize;
        ranges.push((start, count));
    }
    Ok((buf_abs, buf_count, ranges))
}

/// Parse one layer's instances using its per-species (start, count) table.
fn parse_layer_instances(
    file_data: &[u8],
    layer_idx: usize,
    num_species: usize,
) -> Result<Vec<ForestInstance>, Report<ForestError>> {
    let (instances_abs, layer_total, species_table) =
        read_layer(file_data, layer_idx, num_species).map_err(Report::new)?;
    if layer_total == 0 {
        return Ok(Vec::new());
    }

    // Sanity: the per-species count sums should equal the layer's
    // buffer_count. Diverges on malformed files; warn but trust the ranges.
    let sum: usize = species_table.iter().map(|(_, c)| *c).sum();
    if sum != layer_total {
        eprintln!(
            "Warning: forest.bin layer {} per-species sum ({sum}) != layer count ({layer_total}); using per-species ranges",
            LAYER_NAMES[layer_idx],
        );
    }

    let bytes_needed = layer_total * INSTANCE_SIZE;
    if instances_abs + bytes_needed > file_data.len() {
        return Err(Report::new(ForestError::DataTooShort {
            offset: instances_abs,
            need: bytes_needed,
            have: file_data.len().saturating_sub(instances_abs),
        }));
    }
    let input = &mut &file_data[instances_abs..instances_abs + bytes_needed];
    let raw_instances: Vec<RawInstance> = repeat(layer_total, parse_raw_instance)
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(ForestError::ParseError(format!("{e}"))))?;

    let mut instances = Vec::with_capacity(layer_total);
    for (sp_idx, &(start, count)) in species_table.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let end = start + count;
        if end > raw_instances.len() {
            eprintln!(
                "Warning: forest layer {} species {sp_idx} range {start}..{end} exceeds instance count {}",
                LAYER_NAMES[layer_idx],
                raw_instances.len(),
            );
            continue;
        }
        for raw in &raw_instances[start..end] {
            instances.push(ForestInstance { species_index: sp_idx, x: raw.x, y: raw.y, z: raw.z });
        }
    }
    Ok(instances)
}

/// Parse a `forest.bin` file — species table + all four placement layers.
pub fn parse_forest(file_data: &[u8]) -> Result<Forest, Report<ForestError>> {
    if file_data.len() < 32 {
        return Err(Report::new(ForestError::DataTooShort { offset: 0, need: 32, have: file_data.len() }));
    }

    // Parse header.
    let header_input = &mut &file_data[0x00..];
    let num_species = le_u64
        .parse_next(header_input)
        .map_err(|e: ErrMode<ContextError>| Report::new(ForestError::ParseError(format!("{e}"))))?
        as usize;
    let string_table_offset = le_u64
        .parse_next(header_input)
        .map_err(|e: ErrMode<ContextError>| Report::new(ForestError::ParseError(format!("{e}"))))?
        as usize;

    if num_species == 0 {
        return Ok(Forest { species: Vec::new(), layers: Default::default() });
    }
    if num_species > 1000 {
        return Err(Report::new(ForestError::ParseError(format!("unreasonable species count: {num_species}"))));
    }

    // Validate string table bounds.
    let string_table_end = string_table_offset + num_species * 16;
    if string_table_end > file_data.len() {
        return Err(Report::new(ForestError::DataTooShort {
            offset: string_table_offset,
            need: num_species * 16,
            have: file_data.len().saturating_sub(string_table_offset),
        }));
    }

    // Parse species string table. Names are emitted in file order, indexed
    // by species_index in the per-layer (start, count) cell tables.
    let mut species = Vec::with_capacity(num_species);

    for i in 0..num_species {
        let entry_off = string_table_offset + i * 16;
        let input = &mut &file_data[entry_off..];
        let (str_len, str_relptr) = parse_string_table_entry(input)
            .map_err(|e: ErrMode<ContextError>| Report::new(ForestError::ParseError(format!("{e}"))))?;

        let str_len = str_len as usize;
        let str_abs = resolve_relptr(entry_off, str_relptr);

        if str_len == 0 || str_abs + str_len > file_data.len() {
            species.push(format!("species_{i}"));
            continue;
        }

        // Exclude null terminator.
        let name_bytes = &file_data[str_abs..str_abs + str_len - 1];
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        species.push(name);
    }

    // Read all four layers from their fixed engine-defined offsets. Each
    // layer carries its own instance buffer pointer + count + per-species
    // (start, count) table; no byte-grep heuristic needed.
    let layers = [
        parse_layer_instances(file_data, 0, num_species)?,
        parse_layer_instances(file_data, 1, num_species)?,
        parse_layer_instances(file_data, 2, num_species)?,
        parse_layer_instances(file_data, 3, num_species)?,
    ];

    Ok(Forest { species, layers })
}
