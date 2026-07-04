//! Parser for space-level `terrain.bin` compiled height maps.
//!
//! Engine ground truth: `Terrain::CompiledHeightMap::loadInternal`
//! (`lib/terrain/terrain2/compiled_height_map.cpp`, client build 12506899).
//!
//! File layout (little-endian):
//!
//! ```text
//! 0x00  u32  magic 'trb\0' (0x00627274)
//! 0x04  u32  stored width  = chunks_per_axis * chunk_grid_edge
//! 0x08  u32  stored height = chunks_per_axis * chunk_grid_edge
//! 0x0c  u32  lo16 = chunks_per_axis (terrain chunk grid dimension)
//!            hi16 = chunk_grid_edge (poles per chunk edge, incl. borders)
//! 0x10  f32  global min height, f32 global max height
//! 0x18  per-chunk (min, max) f32 pairs, chunks_per_axis² entries,
//!       row-major over the chunk grid
//! ...   RLE-compressed f32 stream, decompressing to exactly width*height
//! ```
//!
//! RLE encoding (deterministic — the engine decode has NO heuristics):
//! a single word is a literal; a doubled word is an escape whose third
//! word is the TOTAL repeat count. Trailing 1–2 words are literal. The
//! decoder must produce exactly `width * height` values ("decompressed
//! data size doesn't match size from header" otherwise).
//!
//! The decompressed buffer is a row-major MOSAIC of per-chunk blocks:
//! each chunk owns an `edge × edge` pole block laid inline, and every
//! block carries a 3-pole border (`edge - 3` visible cells; poles
//! [1 ..= edge-2] of adjacent blocks overlap by two — verified
//! numerically: `left[64] == right[0]`, `left[65] == right[1]` across
//! chunk seams, plus 196/196 authored per-chunk (min,max) pairs match
//! this layout exactly on s07_Advance and 324/324 on 40_Okinawa).
//! Treating the mosaic as a uniform image (the old behaviour) squeezes
//! every chunk by (edge-3)/edge and shifts features by up to 3 poles —
//! coastline-scale distortion. This parser therefore resamples the
//! mosaic into a UNIFORM `(chunks*(edge-3) + 1)²` pole grid spanning the
//! chunk-grid bounds from space.settings; pole anchor = 1 (visible span
//! = poles [1 ..= edge-2], empirically pinned by a 35K-tree terrain-fit
//! on 40_Okinawa: MAD 0.065 native units ≈ 2 m).

use rootcause::Report;
use thiserror::Error;

/// Magic bytes: `trb\0` = 0x00627274 little-endian.
pub const TERRAIN_MAGIC: u32 = 0x00627274;
/// Visible-pole anchor inside each chunk block (see module docs).
const POLE_ANCHOR: usize = 1;
/// Border poles per chunk block edge.
const BORDER_POLES: usize = 3;

#[derive(Debug, Error)]
pub enum TerrainError {
    #[error("data too short: need {need} bytes, have {have}")]
    DataTooShort { need: usize, have: usize },
    #[error("bad magic: expected 0x{:08X}, got 0x{got:08X}", TERRAIN_MAGIC)]
    BadMagic { got: u32 },
    #[error("inconsistent header: {w}x{h} vs {chunks} chunks of edge {edge}")]
    BadHeader { w: u32, h: u32, chunks: u16, edge: u16 },
    #[error("RLE decode produced {decoded} values, expected exactly {expected} (width*height)")]
    SizeMismatch { decoded: usize, expected: usize },
}

/// Parsed terrain heightmap, resampled to a uniform grid.
#[derive(Debug)]
pub struct Terrain {
    /// Uniform grid poles per row: `chunks_per_axis * (edge - 3) + 1`.
    pub width: u32,
    /// Uniform grid poles per column (same formula; grids are square).
    pub height: u32,
    /// Terrain chunks per axis (== the space.settings chunk-grid extent).
    pub chunks_per_axis: u16,
    /// Poles per chunk block edge in the stored mosaic (incl. 3 borders).
    pub chunk_grid_edge: u16,
    /// Global min/max heights from the file header (metres).
    pub min_height: f32,
    pub max_height: f32,
    /// Uniform row-major heightmap (`width * height` entries, metres);
    /// row 0 = minimum BW z edge of the chunk grid.
    pub heightmap: Vec<f32>,
    /// Chunk blocks whose decoded (min, max) disagreed with the authored
    /// per-chunk table — nonzero means format drift; investigate.
    pub minmax_mismatches: u32,
}

/// Parse a `terrain.bin` file.
pub fn parse_terrain(file_data: &[u8]) -> Result<Terrain, Report<TerrainError>> {
    if file_data.len() < 0x18 {
        return Err(Report::new(TerrainError::DataTooShort { need: 0x18, have: file_data.len() }));
    }
    let u32_at = |o: usize| u32::from_le_bytes(file_data[o..o + 4].try_into().unwrap());
    let f32_at = |o: usize| f32::from_le_bytes(file_data[o..o + 4].try_into().unwrap());

    let magic = u32_at(0);
    if magic != TERRAIN_MAGIC {
        return Err(Report::new(TerrainError::BadMagic { got: magic }));
    }
    let stored_w = u32_at(4);
    let stored_h = u32_at(8);
    let packed = u32_at(12);
    let chunks = (packed & 0xFFFF) as u16;
    let edge = (packed >> 16) as u16;
    if stored_w != chunks as u32 * edge as u32
        || stored_h != chunks as u32 * edge as u32
        || (edge as usize) <= BORDER_POLES
    {
        return Err(Report::new(TerrainError::BadHeader { w: stored_w, h: stored_h, chunks, edge }));
    }
    let min_height = f32_at(16);
    let max_height = f32_at(20);

    // Authored per-chunk (min, max) table.
    let n_chunks = chunks as usize * chunks as usize;
    let table_off = 0x18;
    let data_off = table_off + n_chunks * 8;
    if file_data.len() < data_off {
        return Err(Report::new(TerrainError::DataTooShort { need: data_off, have: file_data.len() }));
    }

    // Engine RLE decode: single word = literal; doubled word = escape,
    // third word = total repeat count; trailing 1-2 words literal.
    let body = &file_data[data_off..];
    let n_words = body.len() / 4;
    let word = |i: usize| u32::from_le_bytes(body[i * 4..i * 4 + 4].try_into().unwrap());
    let expected = (stored_w as usize) * (stored_h as usize);
    let mut mosaic: Vec<f32> = Vec::with_capacity(expected);
    let mut i = 0usize;
    while i + 2 < n_words {
        let v = word(i);
        if word(i + 1) == v {
            let count = word(i + 2) as usize;
            if mosaic.len() + count > expected + 8 {
                // Runaway run — corrupt data; bail with a size error.
                return Err(Report::new(TerrainError::SizeMismatch {
                    decoded: mosaic.len() + count,
                    expected,
                }));
            }
            mosaic.resize(mosaic.len() + count, f32::from_bits(v));
            i += 3;
        } else {
            mosaic.push(f32::from_bits(v));
            i += 1;
        }
    }
    while i < n_words {
        mosaic.push(f32::from_bits(word(i)));
        i += 1;
    }
    if mosaic.len() != expected {
        return Err(Report::new(TerrainError::SizeMismatch { decoded: mosaic.len(), expected }));
    }

    // Validate decoded blocks against the authored per-chunk (min, max)
    // table — a free integrity check on both the RLE and the layout.
    let dim = chunks as usize;
    let edge_us = edge as usize;
    let mosaic_w = stored_w as usize;
    let mut minmax_mismatches = 0u32;
    for bz in 0..dim {
        for bx in 0..dim {
            let mut bmin = f32::INFINITY;
            let mut bmax = f32::NEG_INFINITY;
            for pz in 0..edge_us {
                let row = (bz * edge_us + pz) * mosaic_w + bx * edge_us;
                for &h in &mosaic[row..row + edge_us] {
                    bmin = bmin.min(h);
                    bmax = bmax.max(h);
                }
            }
            let t_off = table_off + (bz * dim + bx) * 8;
            let tmin = f32_at(t_off);
            let tmax = f32_at(t_off + 4);
            if (bmin - tmin).abs() > 1e-3 || (bmax - tmax).abs() > 1e-3 {
                minmax_mismatches += 1;
            }
        }
    }

    // Resample the bordered mosaic into a uniform pole grid: global pole
    // g maps to chunk b = min(g / vis, dim-1), block pole p = g - b*vis
    // + POLE_ANCHOR (the last global pole reads the final chunk's pole
    // anchor+vis — the chunk-grid max edge).
    let vis = edge_us - BORDER_POLES;
    let n = dim * vis + 1;
    let src_index = |g: usize| -> usize {
        let b = (g / vis).min(dim - 1);
        b * edge_us + (g - b * vis) + POLE_ANCHOR
    };
    let mut heightmap = Vec::with_capacity(n * n);
    for gz in 0..n {
        let src_row = src_index(gz) * mosaic_w;
        for gx in 0..n {
            heightmap.push(mosaic[src_row + src_index(gx)]);
        }
    }

    Ok(Terrain {
        width: n as u32,
        height: n as u32,
        chunks_per_axis: chunks,
        chunk_grid_edge: edge,
        min_height,
        max_height,
        heightmap,
        minmax_mismatches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic terrain.bin: dim=2, edge=7 (vis=4, mosaic 14x14).
    /// Heights encode their mosaic position so the uniform resample can be
    /// verified pole-exactly; a long run exercises the RLE escape.
    fn synthetic() -> Vec<u8> {
        let dim = 2usize;
        let edge = 7usize;
        let w = dim * edge;
        // mosaic value at (row, col): block-local poles copied from a
        // conceptual global field f(gx, gz) = gz*100 + gx, where block
        // (bx,bz) pole (px,pz) sits at global (bx*4 + px - 1, bz*4 + pz - 1).
        let f = |g_signed: i64| g_signed as f32;
        let global = |b: usize, p: usize| (b * 4 + p) as i64 - 1;
        let mut mosaic = vec![0f32; w * w];
        for bz in 0..dim {
            for bx in 0..dim {
                for pz in 0..edge {
                    for px in 0..edge {
                        let gv = f(global(bz, pz)) * 100.0 + f(global(bx, px));
                        mosaic[(bz * edge + pz) * w + bx * edge + px] = gv;
                    }
                }
            }
        }
        // Overwrite one block with a constant to force an RLE run.
        for pz in 0..edge {
            for px in 0..edge {
                mosaic[(edge + pz) * w + edge + px] = 5.0; // block (1,1)
            }
        }
        // Header.
        let mut out = Vec::new();
        out.extend_from_slice(&TERRAIN_MAGIC.to_le_bytes());
        out.extend_from_slice(&(w as u32).to_le_bytes());
        out.extend_from_slice(&(w as u32).to_le_bytes());
        out.extend_from_slice(&(((edge as u32) << 16) | dim as u32).to_le_bytes());
        let gmin = mosaic.iter().cloned().fold(f32::INFINITY, f32::min);
        let gmax = mosaic.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        out.extend_from_slice(&gmin.to_le_bytes());
        out.extend_from_slice(&gmax.to_le_bytes());
        // Per-chunk minmax table.
        for bz in 0..dim {
            for bx in 0..dim {
                let mut bmin = f32::INFINITY;
                let mut bmax = f32::NEG_INFINITY;
                for pz in 0..edge {
                    for px in 0..edge {
                        let v = mosaic[(bz * edge + pz) * w + bx * edge + px];
                        bmin = bmin.min(v);
                        bmax = bmax.max(v);
                    }
                }
                out.extend_from_slice(&bmin.to_le_bytes());
                out.extend_from_slice(&bmax.to_le_bytes());
            }
        }
        // RLE encode: runs >= 2 as [v, v, count], singles literal — the
        // encoder convention the engine decoder implies.
        let mut i = 0usize;
        while i < mosaic.len() {
            let v = mosaic[i].to_bits();
            let mut run = 1usize;
            while i + run < mosaic.len() && mosaic[i + run].to_bits() == v {
                run += 1;
            }
            if run >= 2 {
                out.extend_from_slice(&v.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
                out.extend_from_slice(&(run as u32).to_le_bytes());
            } else {
                out.extend_from_slice(&v.to_le_bytes());
            }
            i += run;
        }
        out
    }

    #[test]
    fn parses_synthetic_uniform_grid() {
        let data = synthetic();
        let t = parse_terrain(&data).expect("parse");
        assert_eq!(t.chunks_per_axis, 2);
        assert_eq!(t.chunk_grid_edge, 7);
        assert_eq!(t.width, 9); // 2*4 + 1
        assert_eq!(t.height, 9);
        assert_eq!(t.heightmap.len(), 81);
        assert_eq!(t.minmax_mismatches, 0);
        // Uniform pole (gx, gz) = f(gz)*100 + f(gx) wherever the source
        // block wasn't the constant-5 one. Global poles 0..8 map to
        // world-continuous field values 0..8 (anchor=1 ↔ f(g) = g).
        for gz in 0..9usize {
            for gx in 0..9usize {
                let v = t.heightmap[gz * 9 + gx];
                // block(1,1) region: gx >= 4 && gz >= 4 comes from the
                // constant block (except poles read from neighbours at
                // the seam g==4, which resolve to block bx==1 too since
                // b = min(g/vis, dim-1) = 1).
                let from_const = gx >= 4 && gz >= 4;
                if from_const {
                    assert_eq!(v, 5.0, "pole ({gx},{gz})");
                } else {
                    assert_eq!(v, gz as f32 * 100.0 + gx as f32, "pole ({gx},{gz})");
                }
            }
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = synthetic();
        data[0] = 0xFF;
        assert!(parse_terrain(&data).is_err());
    }

    #[test]
    fn rejects_truncated_stream() {
        let data = synthetic();
        assert!(parse_terrain(&data[..data.len() - 8]).is_err());
    }
}
