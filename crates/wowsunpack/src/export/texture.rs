//! DDS texture loading and conversion for glTF export.

use std::io::Cursor;

use image_dds::image::ExtendedColorType;
use image_dds::image::ImageEncoder;
use image_dds::image::codecs::png::PngEncoder;
use rootcause::Report;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("failed to parse DDS: {0}")]
    DdsParse(String),
    #[error("failed to decode DDS image: {0}")]
    DdsDecode(String),
    #[error("failed to encode PNG: {0}")]
    PngEncode(String),
    #[error("failed to encode DDS: {0}")]
    DdsEncode(String),
    #[error("io error: {0}")]
    IoError(String),
}

/// Force all alpha values to 255 in an RGBA8 PNG buffer.
/// Re-decodes and re-encodes the PNG. Used for model textures where the DDS alpha
/// channel stores non-opacity data (height, roughness).
pub fn force_png_opaque(png_bytes: &mut Vec<u8>) {
    use image_dds::image::ImageReader;
    let Ok(reader) = ImageReader::new(Cursor::new(&*png_bytes)).with_guessed_format() else {
        return;
    };
    let Ok(img) = reader.decode() else { return };
    let mut rgba = img.into_rgba8();
    for pixel in rgba.pixels_mut() {
        pixel[3] = 255;
    }
    let mut buf = Vec::new();
    if PngEncoder::new(&mut buf)
        .write_image(rgba.as_raw(), rgba.width(), rgba.height(), ExtendedColorType::Rgba8)
        .is_ok()
    {
        *png_bytes = buf;
    }
}

/// Decode DDS bytes to PNG bytes (RGBA8), optionally downsampling to a max size.
///
/// If `max_size` is `Some(n)`, the image is downsampled using box filtering so
/// that neither dimension exceeds `n`. This is a simple but effective way to
/// reduce texture memory for map-scale visualization.
pub fn dds_to_png_resized(dds_bytes: &[u8], max_size: Option<u32>) -> Result<Vec<u8>, Report<TextureError>> {
    let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(dds_bytes))
        .map_err(|e| Report::new(TextureError::DdsParse(e.to_string())))?;

    let rgba_image =
        image_dds::image_from_dds(&dds, 0).map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;

    let (w, h) = (rgba_image.width(), rgba_image.height());

    // Downsample if needed.
    let (out_w, out_h, pixels) = if let Some(max) = max_size
        && (w > max || h > max)
    {
        let scale = (max as f32 / w as f32).min(max as f32 / h as f32);
        let nw = ((w as f32 * scale) as u32).max(1);
        let nh = ((h as f32 * scale) as u32).max(1);
        let src = rgba_image.as_raw();
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        // Box filter: average source pixels that map to each destination pixel.
        for dy in 0..nh {
            let sy0 = (dy as f64 * h as f64 / nh as f64) as u32;
            let sy1 = (((dy + 1) as f64 * h as f64 / nh as f64) as u32).min(h);
            for dx in 0..nw {
                let sx0 = (dx as f64 * w as f64 / nw as f64) as u32;
                let sx1 = (((dx + 1) as f64 * w as f64 / nw as f64) as u32).min(w);
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                let mut a = 0u32;
                let mut count = 0u32;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let i = (sy * w + sx) as usize * 4;
                        r += src[i] as u32;
                        g += src[i + 1] as u32;
                        b += src[i + 2] as u32;
                        a += src[i + 3] as u32;
                        count += 1;
                    }
                }
                if count > 0 {
                    let di = (dy * nw + dx) as usize * 4;
                    dst[di] = (r / count) as u8;
                    dst[di + 1] = (g / count) as u8;
                    dst[di + 2] = (b / count) as u8;
                    dst[di + 3] = (a / count) as u8;
                }
            }
        }
        (nw, nh, dst)
    } else {
        (w, h, rgba_image.into_raw())
    };

    let mut png_buf = Vec::new();
    PngEncoder::new(&mut png_buf)
        .write_image(&pixels, out_w, out_h, ExtendedColorType::Rgba8)
        .map_err(|e| Report::new(TextureError::PngEncode(e.to_string())))?;

    Ok(png_buf)
}

/// Decode DDS bytes to PNG bytes (RGBA8).
pub fn dds_to_png(dds_bytes: &[u8]) -> Result<Vec<u8>, Report<TextureError>> {
    let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(dds_bytes))
        .map_err(|e| Report::new(TextureError::DdsParse(e.to_string())))?;

    let rgba_image =
        image_dds::image_from_dds(&dds, 0).map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;

    let mut png_buf = Vec::new();
    PngEncoder::new(&mut png_buf)
        .write_image(rgba_image.as_raw(), rgba_image.width(), rgba_image.height(), ExtendedColorType::Rgba8)
        .map_err(|e| Report::new(TextureError::PngEncode(e.to_string())))?;

    Ok(png_buf)
}

/// Bake a tiled camouflage tile texture with color scheme replacement.
///
/// The tile texture is a color-indexed mask where R/G/B/Black zones correspond
/// to color1/color2/color3/color0 from the color scheme. This function replaces
/// each zone with the appropriate color and returns the result as PNG.
pub fn bake_tiled_camo_png(tile_dds_bytes: &[u8], colors: &[[f32; 4]; 4]) -> Result<Vec<u8>, Report<TextureError>> {
    let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(tile_dds_bytes))
        .map_err(|e| Report::new(TextureError::DdsParse(e.to_string())))?;

    let mut rgba_image =
        image_dds::image_from_dds(&dds, 0).map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;

    for pixel in rgba_image.pixels_mut() {
        let [r, g, b, _a] = pixel.0;
        // Determine zone by dominant channel. DXT1 compression may blend
        // edge pixels, but dominant-channel detection handles this well.
        let color = if r > g && r > b && r > 30 {
            &colors[1] // Red zone → color1
        } else if g > r && g > b && g > 30 {
            &colors[2] // Green zone → color2
        } else if b > r && b > g && b > 30 {
            &colors[3] // Blue zone → color3
        } else {
            &colors[0] // Black/dark zone → color0
        };
        // Convert linear float [0,1] to sRGB [0,255]
        pixel.0 = [
            (linear_to_srgb(color[0]) * 255.0) as u8,
            (linear_to_srgb(color[1]) * 255.0) as u8,
            (linear_to_srgb(color[2]) * 255.0) as u8,
            (color[3].clamp(0.0, 1.0) * 255.0) as u8,
        ];
    }

    let mut png_buf = Vec::new();
    PngEncoder::new(&mut png_buf)
        .write_image(rgba_image.as_raw(), rgba_image.width(), rgba_image.height(), ExtendedColorType::Rgba8)
        .map_err(|e| Report::new(TextureError::PngEncode(e.to_string())))?;

    Ok(png_buf)
}

/// Convert a linear-space color component to sRGB.
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

const TEXTURE_BASE: &str = "content/gameplay/common/camouflage/textures";

/// Load raw DDS bytes from an absolute VFS path.
pub fn load_dds_from_vfs(vfs: &vfs::VfsPath, path: &str) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut file = vfs.join(path).ok()?.open_file().ok()?;
    std::io::Read::read_to_end(&mut file, &mut data).ok()?;
    if data.is_empty() { None } else { Some(data) }
}

/// WG splits the mip pyramid across files: `.dd0` = top mip (e.g. 4096²),
/// `.dd1` = one level down (2048²), `.dd2` = two levels down (1024²),
/// `.dds` = bundled mip tail (≤512²). See CHANGELOG in
/// `tools/toolkit_integration/pbr_textures.md` for the investigation.
pub const DDS_MIP_SUFFIXES: [&str; 4] = [".dd0", ".dd1", ".dd2", ".dds"];

/// Dumps raw WG DDS files (every available mip level) to a sibling
/// directory, preserving the original filenames. Intended for Unity-side
/// Texture Streaming pipelines that want the full mip chain in BC-
/// compressed form rather than the decoded PNG top-mip that glTF sees.
///
/// Usage pattern: every texture-load function takes
/// `Option<&mut RawDdsDumper>`; when set, as soon as the function
/// identifies a VFS path that resolves, it calls
/// `dumper.dump_all_mips(vfs, resolved_path)` to persist every
/// available mip-level file to the dump directory.
pub struct RawDdsDumper {
    dir: std::path::PathBuf,
    /// Deduplication: output filenames already written this session.
    written: std::collections::HashSet<String>,
}

impl RawDdsDumper {
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        let dir = dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self { dir, written: std::collections::HashSet::new() }
    }

    /// Given a VFS path that resolved to one mip-level file (e.g. `_a.dd0`),
    /// enumerate and dump every available mip level under the same base
    /// (`.dd0`, `.dd1`, `.dd2`, `.dds`) into the dump directory.
    ///
    /// Writes each file once per session (dedup'd by output filename).
    /// Silently skips suffixes that don't exist in the VFS.
    pub fn dump_all_mips(&mut self, vfs: &vfs::VfsPath, resolved_path: &str) {
        // Strip the mip-level suffix from `resolved_path`. If it doesn't end
        // in a known mip suffix, the caller gave us an unexpected path;
        // skip.
        let stem = match DDS_MIP_SUFFIXES.iter().find_map(|s| resolved_path.strip_suffix(s)) {
            Some(s) => s,
            None => return,
        };

        for suffix in DDS_MIP_SUFFIXES {
            let full_path = format!("{stem}{suffix}");
            let filename = full_path.rsplit_once('/').map(|(_, n)| n.to_string()).unwrap_or(full_path.clone());
            if self.written.contains(&filename) {
                continue;
            }
            if let Some(bytes) = load_dds_from_vfs(vfs, &full_path) {
                let out = self.dir.join(&filename);
                if let Err(e) = std::fs::write(&out, &bytes) {
                    eprintln!("  Warning: failed to write raw DDS {}: {e}", out.display());
                }
                self.written.insert(filename.clone());

                // Emit glTF-conformant siblings derived from this mip. WG packs
                // `_n` with B = categorical camo gate (not Z) and `_mg` with
                // non-glTF channel order; the siblings let stock Unity / Three.js
                // / Blender / glTF viewers consume the textures without per-
                // consumer decode logic. See `tools/reference/shared/
                // texture_conventions.md` §Decision.
                self.write_conformant_siblings(&filename, &bytes, suffix);
            }
        }
    }

    /// Write conformant siblings for a just-dumped WG file. Quietly noops
    /// for filenames that don't match the `_n` / `_mg` patterns and for
    /// any per-sibling encode failure (warns to stderr; doesn't abort the
    /// dump — non-conformant siblings are independent of the original
    /// dump succeeding).
    fn write_conformant_siblings(&mut self, filename: &str, bytes: &[u8], mip_suffix: &str) {
        let stem_no_mip = match filename.strip_suffix(mip_suffix) {
            Some(s) => s,
            None => return,
        };

        if let Some(stem) = stem_no_mip.strip_suffix("_n") {
            // Normal map: split into _normal + _nbmask siblings.
            let normal_name = format!("{stem}_normal{mip_suffix}");
            let mask_name = format!("{stem}_nbmask{mip_suffix}");

            let need_normal = !self.written.contains(&normal_name);
            let need_mask = !self.written.contains(&mask_name);
            if !need_normal && !need_mask {
                return;
            }

            match split_wg_normal_dds(bytes) {
                Ok((normal_dds, mask_dds)) => {
                    if need_normal {
                        let out = self.dir.join(&normal_name);
                        if let Err(e) = std::fs::write(&out, &normal_dds) {
                            eprintln!("  Warning: failed to write {}: {e}", out.display());
                        } else {
                            self.written.insert(normal_name);
                        }
                    }
                    if need_mask {
                        let out = self.dir.join(&mask_name);
                        if let Err(e) = std::fs::write(&out, &mask_dds) {
                            eprintln!("  Warning: failed to write {}: {e}", out.display());
                        } else {
                            self.written.insert(mask_name);
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to split WG normal {filename}: {e:?} \
                         (consumer must fall back to raw _n.dd?)"
                    );
                }
            }
        } else if let Some(stem) = stem_no_mip.strip_suffix("_mg") {
            // Metallic-gloss map: emit swizzled _mr sibling.
            let mr_name = format!("{stem}_mr{mip_suffix}");
            if self.written.contains(&mr_name) {
                return;
            }
            match swizzle_wg_mg_dds_to_mr(bytes) {
                Ok(mr_dds) => {
                    let out = self.dir.join(&mr_name);
                    if let Err(e) = std::fs::write(&out, &mr_dds) {
                        eprintln!("  Warning: failed to write {}: {e}", out.display());
                    } else {
                        self.written.insert(mr_name);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to swizzle WG MG {filename}: {e:?} \
                         (consumer must swizzle on bind from raw _mg.dd?)"
                    );
                }
            }
        }
    }
}

/// MFM name suffixes that don't appear in texture filenames.
///
/// E.g. MFM `AGM034_16in50_Mk7_skinned.mfm` → texture `AGM034_16in50_Mk7_camo_01.dds`.
const MFM_STRIP_SUFFIXES: &[&str] = &["_skinned", "_wire", "_dead", "_blaze", "_alpha"];

/// Strip a `_<year>` token (4 ASCII digits surrounded by `_` boundaries)
/// from a stem, preserving anything after.
///
/// Used by [`texture_base_names`] to handle WG's inconsistent year
/// suffixing in camo file names. Returns `None` when no year token is
/// present.
///
/// Examples:
/// - `"ASC017_Baltimore_1944_Bow"` → `"ASC017_Baltimore_Bow"`
/// - `"ASB017_Montana_1945"` → `"ASB017_Montana"` (year at end of string)
/// - `"AGM034_16in50_Mk7"` → `None` (no 4-digit `_NNNN_` token)
fn strip_year_token(stem: &str) -> Option<String> {
    let bytes = stem.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i + 5 <= n {
        if bytes[i] == b'_' && bytes[i + 1..i + 5].iter().all(|b| b.is_ascii_digit()) {
            // Boundary check: at position i+5 must be EOF or another '_'.
            if i + 5 == n || bytes[i + 5] == b'_' {
                let mut out = String::with_capacity(n - 5);
                out.push_str(&stem[..i]);
                out.push_str(&stem[i + 5..]);
                return Some(out);
            }
        }
        i += 1;
    }
    None
}

/// Truncate a stem at the first `_<year>` token, returning the prefix.
///
/// Used by [`texture_base_names`] to handle WG ships that ship one
/// camo file for the whole hull rather than per-part. Returns `None`
/// when no year token is present, and an empty result is treated as
/// no-match by callers.
///
/// Examples:
/// - `"ASC017_Baltimore_1944_Bow"` → `"ASC017_Baltimore"`
/// - `"ASB017_Montana_1945_Deckhouse"` → `"ASB017_Montana"`
/// - `"JSB039_Yamato_1945"` → `"JSB039_Yamato"`
fn truncate_at_year_token(stem: &str) -> Option<String> {
    let bytes = stem.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i + 5 <= n {
        if bytes[i] == b'_' && bytes[i + 1..i + 5].iter().all(|b| b.is_ascii_digit()) {
            if i + 5 == n || bytes[i + 5] == b'_' {
                return Some(stem[..i].to_string());
            }
        }
        i += 1;
    }
    None
}

/// Derive texture base names from an MFM stem.
///
/// Returns the original stem first, then the stem with various WG
/// authoring inconsistencies normalised:
///
/// 1. **`MFM_STRIP_SUFFIXES`** — strips `_skinned`/`_wire`/`_dead`/...
///    so turret-style stems like `AGM034_16in50_Mk7_skinned` resolve
///    via the texture name `AGM034_16in50_Mk7`.
/// 2. **Year-token stripped** — drops a `_<year>` token (4 ASCII
///    digits surrounded by `_` boundaries) anywhere in the stem.
///    Handles ships like Baltimore where WG names camo files
///    `ASC017_Baltimore_camo_01.dd0` even though the MFM stems carry
///    `_1944` (e.g. `ASC017_Baltimore_1944_Bow`).
/// 3. **Truncated at year-token** — produces the prefix before any
///    year token. Handles ships that ship one camo file for the whole
///    hull rather than per-part.
///
/// All variants are tried by the consumers ([`load_texture_bytes`],
/// [`discover_texture_schemes`]); the first to produce a hit wins.
pub fn texture_base_names(mfm_stem: &str) -> Vec<String> {
    let mut names = vec![mfm_stem.to_string()];

    // Add MFM-suffix-stripped variants (operate on the ORIGINAL stem, not
    // on the year-stripped variants, so we don't combinatorially explode).
    for suffix in MFM_STRIP_SUFFIXES {
        if let Some(stripped) = mfm_stem.strip_suffix(suffix)
            && !names.iter().any(|n| n == stripped)
        {
            names.push(stripped.to_string());
        }
    }

    // Year-stripped variant: `ASC017_Baltimore_1944_Bow` →
    // `ASC017_Baltimore_Bow`. Catches WG camos that omit the year.
    if let Some(without_year) = strip_year_token(mfm_stem)
        && !without_year.is_empty()
        && !names.iter().any(|n| n == &without_year)
    {
        names.push(without_year);
    }

    // Truncated-at-year variant: `ASC017_Baltimore_1944_Bow` →
    // `ASC017_Baltimore`. Catches WG camos that ship one file per hull
    // (not per part) under the index-only name.
    if let Some(at_year) = truncate_at_year_token(mfm_stem)
        && !at_year.is_empty()
        && !names.iter().any(|n| n == &at_year)
    {
        names.push(at_year);
    }

    names
}

/// Texture channel suffixes that indicate a multi-channel camo scheme.
///
/// When a scheme is discovered as e.g. `GW_a`, the `_a` suffix means it's the albedo
/// channel of scheme `GW`. The `_mg` and `_mgn` suffixes are metallic/gloss channels.
/// These are stripped during discovery to group channels into a single scheme.
const TEXTURE_CHANNEL_SUFFIXES: &[&str] = &["_a", "_mg", "_mgn"];

/// Load the albedo texture for a given MFM stem and camo scheme from the VFS.
///
/// Given an MFM leaf like `JSB039_Yamato_1945_Hull` and scheme like `GW`,
/// tries multiple naming conventions in order:
/// 1. `{stem}_{scheme}_a.dd0/dds` — explicit albedo channel (e.g. `Hull_GW_a.dds`)
/// 2. `{stem}_{scheme}.dd0/dds` — direct replacement (e.g. `Hull_camo_01.dds`)
///
/// Also tries with known MFM suffixes stripped (e.g. `_skinned`) to handle
/// turret models where the texture name differs from the MFM name.
///
/// Returns `(base_name, dds_bytes)` if found, or `None`.
///
/// If `raw_dds_dumper` is `Some`, also dumps every available mip level of
/// the resolved texture to the dumper's directory as a side effect.
pub fn load_texture_bytes(
    vfs: &vfs::VfsPath,
    mfm_stem: &str,
    scheme: &str,
    raw_dds_dumper: Option<&mut RawDdsDumper>,
) -> Option<(String, Vec<u8>)> {
    for base in texture_base_names(mfm_stem) {
        // Try explicit albedo channel first ({base}_{scheme}_a), then direct ({base}_{scheme}).
        let candidates = [
            format!("{TEXTURE_BASE}/{base}_{scheme}_a.dd0"),
            format!("{TEXTURE_BASE}/{base}_{scheme}_a.dds"),
            format!("{TEXTURE_BASE}/{base}_{scheme}.dd0"),
            format!("{TEXTURE_BASE}/{base}_{scheme}.dds"),
        ];

        for path in &candidates {
            if let Ok(vfs_path) = vfs.join(path)
                && let Ok(mut file) = vfs_path.open_file()
            {
                let mut data = Vec::new();
                if std::io::Read::read_to_end(&mut file, &mut data).is_ok() && !data.is_empty() {
                    if let Some(d) = raw_dds_dumper {
                        d.dump_all_mips(vfs, path);
                    }
                    return Some((base, data));
                }
            }
        }
    }

    None
}

/// Load the base albedo texture for a hull mesh from the VFS.
///
/// The base albedo is the "default" ship appearance — gray/weathered paint without
/// any camouflage applied. Textures live in a `textures/` sibling directory next to
/// the ship folder, e.g.:
/// `content/gameplay/japan/ship/battleship/textures/JSB039_Yamato_1945_Hull_a.dd0`
///
/// Prefers `.dd0` (highest resolution, typically 4096x4096) over `.dds` (low-res
/// 512x512 mip tail). Falls back to searching the MFM's own directory.
///
/// `mfm_full_path` is the full VFS path to the MFM file (e.g. ending in `.mfm`).
/// Returns DDS bytes if found.
///
/// If `raw_dds_dumper` is `Some`, also dumps every available mip level of
/// the resolved texture to the dumper's directory as a side effect.
pub fn load_base_albedo_bytes(
    vfs: &vfs::VfsPath,
    mfm_full_path: &str,
    raw_dds_dumper: Option<&mut RawDdsDumper>,
) -> Option<Vec<u8>> {
    let dir = mfm_full_path.rsplit_once('/')?.0;
    let mfm_filename = mfm_full_path.rsplit_once('/')?.1;
    let stem = mfm_filename.strip_suffix(".mfm")?;

    // The textures/ sibling directory: go up from the ship dir to the species dir,
    // then into textures/. E.g. .../cruiser/JSC010_Mogami_1944/ -> .../cruiser/textures/
    let tex_sibling_dir = dir.rsplit_once('/').map(|(parent, _)| format!("{parent}/textures"));

    // Albedo suffix priority: `_a` (standard PBS), `_od` (TILEDLAND overlay diffuse).
    let albedo_suffixes = ["_a", "_od"];

    // Search directories: textures/ sibling, MFM's dir, and TILED/ subdirectory
    // (underwater TILEDLAND materials store textures in a TILED/ subdirectory).
    let tiled_subdir = format!("{dir}/TILED");

    for base in texture_base_names(stem) {
        // Build candidate paths: prefer dd0 (high-res) over dds (low-res mip tail).
        let mut candidates = Vec::new();
        for suffix in &albedo_suffixes {
            if let Some(tex_dir) = &tex_sibling_dir {
                candidates.push(format!("{tex_dir}/{base}{suffix}.dd0"));
                candidates.push(format!("{tex_dir}/{base}{suffix}.dds"));
            }
            candidates.push(format!("{dir}/{base}{suffix}.dd0"));
            candidates.push(format!("{dir}/{base}{suffix}.dds"));
            candidates.push(format!("{tiled_subdir}/{base}{suffix}.dd0"));
            candidates.push(format!("{tiled_subdir}/{base}{suffix}.dds"));
        }

        for path in &candidates {
            if let Ok(vfs_path) = vfs.join(path)
                && let Ok(mut file) = vfs_path.open_file()
            {
                let mut data = Vec::new();
                if std::io::Read::read_to_end(&mut file, &mut data).is_ok() && !data.is_empty() {
                    if let Some(d) = raw_dds_dumper {
                        d.dump_all_mips(vfs, path);
                    }
                    return Some(data);
                }
            }
        }
    }

    None
}

/// Strip texture channel suffixes (`_a`, `_mg`, `_mgn`) from a raw scheme name.
///
/// E.g. `GW_a` → `GW`, `camo_01` → `camo_01` (no channel suffix).
fn strip_channel_suffix(scheme: &str) -> &str {
    for suffix in TEXTURE_CHANNEL_SUFFIXES {
        if let Some(stripped) = scheme.strip_suffix(suffix)
            && !stripped.is_empty()
        {
            return stripped;
        }
    }
    scheme
}

/// Discover available texture schemes for a set of MFM stems by scanning the VFS.
///
/// Multi-channel schemes (e.g. `GW_a` + `GW_mg`) are grouped into a single scheme
/// name (`GW`). Returns sorted, deduplicated scheme names.
pub fn discover_texture_schemes(vfs: &vfs::VfsPath, mfm_stems: &[String]) -> Vec<String> {
    let mut schemes = std::collections::BTreeSet::new();

    let Ok(tex_dir) = vfs.join(TEXTURE_BASE) else {
        return Vec::new();
    };
    let Ok(entries) = tex_dir.read_dir() else {
        return Vec::new();
    };

    // Collect filenames ending in .dds (base mip level — avoids counting .dd0/.dd1/.dd2 dupes).
    let dds_names: Vec<String> = entries
        .filter_map(|entry| {
            let name = entry.filename();
            if name.ends_with(".dds") { Some(name) } else { None }
        })
        .collect();

    for stem in mfm_stems {
        for base in texture_base_names(stem) {
            let prefix = format!("{base}_");
            for name in &dds_names {
                if let Some(rest) = name.strip_prefix(&prefix)
                    && let Some(raw_scheme) = rest.strip_suffix(".dds")
                    && !raw_scheme.is_empty()
                {
                    let scheme = strip_channel_suffix(raw_scheme);
                    schemes.insert(scheme.to_string());
                }
            }
        }
    }

    schemes.into_iter().collect()
}

// ---------------------------------------------------------------------------
// TILEDLAND terrain texture baking
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use image_dds::SurfaceRgba8;
use image_dds::image::RgbaImage;

use crate::models::assets_bin::PrototypeDatabase;
use crate::models::material;
use crate::models::material::MaterialPrototype;

/// Resolve a texture selfId hash from an MFM property to a VFS path,
/// load the DDS bytes, and return them.
fn load_texture_by_hash(
    vfs: &vfs::VfsPath,
    db: &PrototypeDatabase<'_>,
    self_id_index: &HashMap<u64, usize>,
    texture_hash: u64,
    raw_dds_dumper: Option<&mut RawDdsDumper>,
) -> Option<Vec<u8>> {
    let &path_idx = self_id_index.get(&texture_hash)?;
    let full_path = db.reconstruct_path(path_idx, self_id_index);

    // Try .dd0 (high-res) first, then the path as-is (.dds).
    let dd0_path = if full_path.ends_with(".dds") { Some(full_path.replace(".dds", ".dd0")) } else { None };

    for path in dd0_path.iter().chain(std::iter::once(&full_path)) {
        if let Some(data) = load_dds_from_vfs(vfs, path) {
            if let Some(d) = raw_dds_dumper {
                d.dump_all_mips(vfs, path);
            }
            return Some(data);
        }
    }
    None
}

/// Parse an MFM material from assets.bin given its selfId (material_mfm_path_id).
///
/// Returns the parsed material if the MFM is found and parses successfully.
pub fn parse_mfm_from_db(db: &PrototypeDatabase<'_>, mfm_path_id: u64) -> Option<MaterialPrototype> {
    let r2p_value = db.lookup_r2p(mfm_path_id)?;
    let location = db.decode_r2p_value(r2p_value).ok()?;
    if location.blob_index != material::MATERIAL_BLOB_INDEX {
        return None;
    }
    let record_data = db.get_prototype_data(location, material::MATERIAL_ITEM_SIZE).ok()?;
    material::parse_material(record_data).ok()
}

/// Check if a material is a TILEDLAND terrain material.
///
/// TILEDLAND materials have `AHArray` (tile atlas), `blendMap`, and `g_tilesIndex`.
pub fn is_tiledland_material(mat: &MaterialPrototype) -> bool {
    mat.get_texture_hash("AHArray").is_some()
        && mat.get_texture_hash("blendMap").is_some()
        && mat.get_vec4("g_tilesIndex").is_some()
}

/// Bake a TILEDLAND terrain albedo texture from MFM material properties.
///
/// The TILEDLAND shader composites 4 tile layers from a shared atlas texture,
/// weighted by the RGBA channels of a blend map. Parameters:
/// - `AHArray`: texture array atlas (Albedo/Height), each layer is a tile material
/// - `blendMap`: per-pixel RGBA blend weights selecting which atlas layers to use
/// - `g_tilesIndex`: vec4 of 4 atlas layer indices (one per blend channel)
/// - `g_tilesScale`: float UV tiling scale for atlas sampling
///
/// Note: ODMap is intentionally NOT applied — it requires g_overlayOpacity/g_overlayDepth
/// shader parameters for correct blending, and naive multiplication darkens the result.
///
/// Returns PNG bytes of the baked albedo texture at the blend map's resolution.
pub fn bake_tiledland_albedo(
    mat: &MaterialPrototype,
    vfs: &vfs::VfsPath,
    db: &PrototypeDatabase<'_>,
    self_id_index: &HashMap<u64, usize>,
    max_size: Option<u32>,
) -> Option<Vec<u8>> {
    // Extract material properties.
    let ah_hash = mat.get_texture_hash("AHArray")?;
    let blend_hash = mat.get_texture_hash("blendMap")?;
    let tiles_index = mat.get_vec4("g_tilesIndex")?;
    let tiles_scale = mat.get_float("g_tilesScale").unwrap_or(16.0);

    // Optional sheen tint color — the TILEDLAND shader uses this to add vegetation
    // coloring (e.g. green tint) on top of the otherwise earth-tone atlas tiles.
    let sheen_tint = mat.get_vec4("addSheenTintColor");
    let sheen_amount = mat.get_float("sheen").unwrap_or(0.0);

    // Load and decode the tile atlas (array texture).
    let ah_dds_bytes = load_texture_by_hash(vfs, db, self_id_index, ah_hash, None)?;
    let ah_dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(&ah_dds_bytes)).ok()?;
    let num_layers = ah_dds.get_num_array_layers().max(1);
    // Decode only mip 0 of all layers.
    let ah_surface = SurfaceRgba8::decode_layers_mipmaps_dds(&ah_dds, 0..num_layers, 0..1).ok()?;

    // Extract the 4 tile layers we need.
    let layer_indices: [u32; 4] =
        [tiles_index[0] as u32, tiles_index[1] as u32, tiles_index[2] as u32, tiles_index[3] as u32];
    let tile_w = ah_surface.width;
    let tile_h = ah_surface.height;

    let tile_layers: Vec<Option<RgbaImage>> =
        layer_indices.iter().map(|&idx| ah_surface.get_image(idx, 0, 0)).collect();

    // Load and decode the blend map.
    let blend_dds_bytes = load_texture_by_hash(vfs, db, self_id_index, blend_hash, None)?;
    let blend_img = {
        let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(&blend_dds_bytes)).ok()?;
        image_dds::image_from_dds(&dds, 0).ok()?
    };
    let blend_w = blend_img.width();
    let blend_h = blend_img.height();

    // Determine output size: use blend map resolution (typically 512-1024).
    let out_w = blend_w;
    let out_h = blend_h;

    // Bake: for each output pixel, sample blend weights and composite tile layers.
    let mut output = RgbaImage::new(out_w, out_h);

    for py in 0..out_h {
        for px in 0..out_w {
            let blend_pixel = blend_img.get_pixel(px, py);
            let weights = [
                blend_pixel[0] as f32 / 255.0, // R → layer 0
                blend_pixel[1] as f32 / 255.0, // G → layer 1
                blend_pixel[2] as f32 / 255.0, // B → layer 2
                blend_pixel[3] as f32 / 255.0, // A → layer 3
            ];

            // Normalize weights so they sum to 1. If all zero, use equal weights.
            let sum: f32 = weights.iter().sum();
            let norm = if sum > 0.001 {
                [weights[0] / sum, weights[1] / sum, weights[2] / sum, weights[3] / sum]
            } else {
                [0.25, 0.25, 0.25, 0.25]
            };

            // UV in blend map space [0..1], then tile with g_tilesScale.
            let u = px as f32 / out_w as f32;
            let v = py as f32 / out_h as f32;
            let tile_u = (u * tiles_scale).fract();
            let tile_v = (v * tiles_scale).fract();

            // Sample each tile layer and blend.
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;

            for (i, layer_img) in tile_layers.iter().enumerate() {
                if norm[i] < 0.001 {
                    continue;
                }
                if let Some(img) = layer_img {
                    let tx = ((tile_u * tile_w as f32) as u32).min(tile_w - 1);
                    let ty = ((tile_v * tile_h as f32) as u32).min(tile_h - 1);
                    let p = img.get_pixel(tx, ty);
                    r += p[0] as f32 * norm[i];
                    g += p[1] as f32 * norm[i];
                    b += p[2] as f32 * norm[i];
                }
            }

            // Apply addSheenTintColor — the TILEDLAND shader uses this to add
            // vegetation coloring (green tint) on top of the earth-tone atlas tiles.
            // We lerp toward the tint color by the sheen amount.
            if let Some(tint) = sheen_tint
                && sheen_amount > 0.0
            {
                let t = sheen_amount;
                r = r * (1.0 - t) + (tint[0] * 255.0) * t;
                g = g * (1.0 - t) + (tint[1] * 255.0) * t;
                b = b * (1.0 - t) + (tint[2] * 255.0) * t;
            }

            output.put_pixel(
                px,
                py,
                image_dds::image::Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    255,
                ]),
            );
        }
    }

    // Downsample if needed.
    let (final_w, final_h, pixels) = if let Some(max) = max_size
        && (out_w > max || out_h > max)
    {
        let scale = (max as f32 / out_w as f32).min(max as f32 / out_h as f32);
        let nw = ((out_w as f32 * scale) as u32).max(1);
        let nh = ((out_h as f32 * scale) as u32).max(1);
        let src = output.as_raw();
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        for dy in 0..nh {
            let sy0 = (dy as f64 * out_h as f64 / nh as f64) as u32;
            let sy1 = (((dy + 1) as f64 * out_h as f64 / nh as f64) as u32).min(out_h);
            for dx in 0..nw {
                let sx0 = (dx as f64 * out_w as f64 / nw as f64) as u32;
                let sx1 = (((dx + 1) as f64 * out_w as f64 / nw as f64) as u32).min(out_w);
                let mut ra = 0u32;
                let mut ga = 0u32;
                let mut ba = 0u32;
                let mut count = 0u32;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let i = (sy * out_w + sx) as usize * 4;
                        ra += src[i] as u32;
                        ga += src[i + 1] as u32;
                        ba += src[i + 2] as u32;
                        count += 1;
                    }
                }
                if count > 0 {
                    let di = (dy * nw + dx) as usize * 4;
                    dst[di] = (ra / count) as u8;
                    dst[di + 1] = (ga / count) as u8;
                    dst[di + 2] = (ba / count) as u8;
                    dst[di + 3] = 255;
                }
            }
        }
        (nw, nh, dst)
    } else {
        (out_w, out_h, output.into_raw())
    };

    // Encode to PNG.
    let mut png_buf = Vec::new();
    PngEncoder::new(&mut png_buf).write_image(&pixels, final_w, final_h, ExtendedColorType::Rgba8).ok()?;

    Some(png_buf)
}

// ---------------------------------------------------------------------------
// PBR auxiliary channels (normal / metallicGloss / ambientOcclusion)
// ---------------------------------------------------------------------------

/// PBR auxiliary channels loaded for a material alongside the albedo.
///
/// Each field is `Some(png_bytes)` when the corresponding MFM property
/// (`normalMap`, `metallicGlossMap`, `ambientOcclusionMap`) resolves to a
/// readable DDS in the VFS.
///
/// **Phase A (current)**: PNG bytes are the raw DDS→PNG conversion with
/// WG's original channel layout — no swizzling to match glTF conventions.
/// Downstream consumers (Unity, Blender shader graphs) need to remap
/// channels themselves. See `tools/toolkit_integration/pbr_textures.md`
/// in the parent repo for the channel-packing plan and the Phase B
/// swizzler that will make these glTF-correct.
#[derive(Default, Debug)]
pub struct PbrChannels {
    /// `normalMap` — tangent-space normal map. WG typically uses BC5/DXN
    /// (XY in RG, B reconstructed), which matches glTF convention.
    pub normal: Option<Vec<u8>>,
    /// `metallicGlossMap` — packed metallic + gloss. glTF expects G=roughness
    /// B=metallic; WG packing differs and passes through raw in Phase A.
    pub metallic_roughness: Option<Vec<u8>>,
    /// `ambientOcclusionMap` — usually R-channel grayscale. Passes through
    /// raw; glTF reads R for occlusion so this path is already correct.
    pub occlusion: Option<Vec<u8>>,
}

/// Resolve PBR auxiliary texture hashes from an MFM and load their DDS
/// bytes, converting each to PNG.
///
/// `mfm_path_id` is the MFM's `selfId` hash (i.e. `render_set.material_mfm_path_id`).
/// Returns `PbrChannels::default()` (all `None`) if the MFM can't be parsed
/// or the properties aren't present.
///
/// The `metallicGlossMap` channel is swizzled from WG's packing
/// (R=cavity/specOcc, G=metallic, B=gloss, A=unused) to the glTF
/// metallicRoughnessTexture convention (R=unused, G=roughness, B=metallic,
/// A=unused) via [`repack_wg_mg_to_gltf_mr`]. `normalMap` and
/// `ambientOcclusionMap` pass through unchanged (already glTF-compatible).
///
/// Packing authority: WoWS `PBS_ship_metallic` shader fxo metadata,
/// cross-referenced with empirical channel statistics on 4 PBS
/// materials. See `tools/toolkit_integration/pbr_textures.md` in the
/// pipeline repo for the full derivation.
pub fn load_pbr_channels(
    vfs: &vfs::VfsPath,
    db: &PrototypeDatabase<'_>,
    self_id_index: &HashMap<u64, usize>,
    mfm_path_id: u64,
    mut raw_dds_dumper: Option<&mut RawDdsDumper>,
) -> PbrChannels {
    let Some(mat) = parse_mfm_from_db(db, mfm_path_id) else {
        return PbrChannels::default();
    };

    // Closure-local loader — threads the dumper by reborrowing so each
    // property resolution can independently trigger mip dumps.
    let mut load_raw = |property: &str| -> Option<Vec<u8>> {
        let hash = mat.get_texture_hash(property)?;
        let dds = load_texture_by_hash(vfs, db, self_id_index, hash, raw_dds_dumper.as_deref_mut())?;
        dds_to_png(&dds).ok()
    };

    // Normal map: WG packs (R,G) = tangent X,Y but the B channel is a
    // categorical no-camo region marker (not Z). Rewrite B as
    // sqrt(1 - X² - Y²) so the embedded PNG is a glTF-conformant normal
    // map — the original B is preserved on disk via the
    // `_nbmask.dd?` siblings emitted by `RawDdsDumper` (see
    // `tools/reference/shared/texture_conventions.md` §Decision).
    let normal = load_raw("normalMap")
        .and_then(|png| replace_normal_b_with_reconstructed_z_png(&png).ok());
    let occlusion = load_raw("ambientOcclusionMap");
    let metallic_roughness = load_raw("metallicGlossMap")
        .and_then(|png| repack_wg_mg_to_gltf_mr(&png).ok());

    PbrChannels { normal, metallic_roughness, occlusion }
}

/// Repack WG's `metallicGlossMap` PNG into a glTF-conformant
/// `metallicRoughnessTexture` PNG.
///
/// WG packing (observed + confirmed against PBS_ship_metallic.fxo strings
/// in the SEA-group shader backup):
///   R = cavity / specular occlusion (continuous, panel-line detail)
///   G = metallic mask (bimodal 0/1)
///   B = gloss (continuous, higher = smoother)
///   A = unused (BC1 no-alpha path, always 255)
///
/// glTF metallicRoughnessTexture packing (spec):
///   R = unused (or occlusion when using combined ORM)
///   G = roughness (linear; higher = rougher)
///   B = metallic
///   A = unused
///
/// Swizzle:
///   out.R = in.R         (preserve cavity; no-op if consumer ignores R)
///   out.G = 255 - in.B   (roughness = 1 - gloss)
///   out.B = in.G         (metallic passes through)
///   out.A = 255
///
/// The gloss→roughness inversion is *linear* here. The WG shader may
/// apply a `g_legacyGlossRemap` power curve before the inversion; when
/// that property is present the output won't perfectly match in-engine
/// appearance, but the mapping will be qualitatively correct (dielectric
/// vs conductor, smooth vs rough).
pub fn repack_wg_mg_to_gltf_mr(png_bytes: &[u8]) -> Result<Vec<u8>, Report<TextureError>> {
    use image_dds::image::ImageReader;

    let reader = ImageReader::new(Cursor::new(png_bytes))
        .with_guessed_format()
        .map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;
    let img = reader.decode().map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;
    let mut rgba = img.into_rgba8();

    for pixel in rgba.pixels_mut() {
        let [r, g, b, _a] = pixel.0;
        // R: pass-through cavity/specOcc (glTF readers that use combined
        // ORM pack occlusion in R; dedicated consumers can ignore it).
        // G: roughness = 255 - gloss.
        // B: metallic from WG's G channel.
        pixel.0 = [r, 255u8.saturating_sub(b), g, 255];
    }

    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(rgba.as_raw(), rgba.width(), rgba.height(), ExtendedColorType::Rgba8)
        .map_err(|e| Report::new(TextureError::PngEncode(e.to_string())))?;
    Ok(out)
}

/// Replace WG's categorical-mask `B` channel in a normal map PNG with the
/// reconstructed Z (`sqrt(1 - X² - Y²)`), so the result is a glTF-conformant
/// tangent-space normal map.
///
/// Why: WG packs (R,G) = (X,Y) in the standard way, but the `B` channel
/// carries a per-pixel "no-camo region" categorical marker (119 = apply
/// camo, 153 = deck-skip, 34 = detail-skip), NOT Z. Standard glTF / Unity
/// `UnpackNormal` paths expect B = Z; if they read WG's B as Z the
/// resulting normals are visually wrong.
///
/// The original B is preserved by the disk-dump path as a sibling
/// `_nbmask.dds` (see [`split_wg_normal_dds`]) so the camo gate has its
/// own input texture independent of the normal map.
///
/// See `tools/reference/shared/texture_conventions.md` §Normal in the
/// pipeline repo for the full convention writeup.
pub fn replace_normal_b_with_reconstructed_z_png(
    png_bytes: &[u8],
) -> Result<Vec<u8>, Report<TextureError>> {
    use image_dds::image::ImageReader;

    let reader = ImageReader::new(Cursor::new(png_bytes))
        .with_guessed_format()
        .map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;
    let img = reader
        .decode()
        .map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))?;
    let mut rgba = img.into_rgba8();

    for pixel in rgba.pixels_mut() {
        // Decode tangent-space (X, Y) from R, G in [-1, 1].
        let x = (pixel.0[0] as f32 / 255.0) * 2.0 - 1.0;
        let y = (pixel.0[1] as f32 / 255.0) * 2.0 - 1.0;
        // Z = sqrt(1 - X² - Y²); clamp for numeric safety.
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        // Re-encode Z to [0, 255].
        let bz = (((z + 1.0) * 0.5 * 255.0).round() as i32).clamp(0, 255) as u8;
        pixel.0[2] = bz;
        pixel.0[3] = 255;
    }

    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            ExtendedColorType::Rgba8,
        )
        .map_err(|e| Report::new(TextureError::PngEncode(e.to_string())))?;
    Ok(out)
}

/// Decode a single-mip DDS to RGBA8 (image_dds wrapper).
fn decode_single_mip_dds(
    dds_bytes: &[u8],
) -> Result<image_dds::image::RgbaImage, Report<TextureError>> {
    let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(dds_bytes))
        .map_err(|e| Report::new(TextureError::DdsParse(e.to_string())))?;
    image_dds::image_from_dds(&dds, 0)
        .map_err(|e| Report::new(TextureError::DdsDecode(e.to_string())))
}

/// Encode an RGBA8 image as a single-mip DDS in the given format.
fn encode_single_mip_dds(
    rgba: &image_dds::image::RgbaImage,
    format: image_dds::ImageFormat,
) -> Result<Vec<u8>, Report<TextureError>> {
    let surface = image_dds::SurfaceRgba8::from_image(rgba);
    let dds = surface
        .encode_dds(format, image_dds::Quality::Fast, image_dds::Mipmaps::Disabled)
        .map_err(|e| Report::new(TextureError::DdsEncode(e.to_string())))?;
    let mut buf = Vec::new();
    dds.write(&mut buf)
        .map_err(|e| Report::new(TextureError::DdsEncode(e.to_string())))?;
    Ok(buf)
}

/// Swizzle a single-mip WG `_mg` DDS into a glTF-conformant `_mr` DDS.
///
/// The pixel-level swizzle is identical to [`repack_wg_mg_to_gltf_mr`]
/// (R cavity pass-through, G = 255 - B (gloss → roughness), B = G
/// (metallic), A = 255). Output format is `BC1RgbaUnorm` to match the
/// WG MG packing (BC1 no-alpha is canonical for `_mg` files; A is
/// always unused).
///
/// Single mip — WG splits the mip pyramid across `.dd0` / `.dd1` /
/// `.dd2` / `.dds` files, each carrying one mip level. Callers iterate
/// per-file.
pub fn swizzle_wg_mg_dds_to_mr(dds_bytes: &[u8]) -> Result<Vec<u8>, Report<TextureError>> {
    let mut rgba = decode_single_mip_dds(dds_bytes)?;
    for pixel in rgba.pixels_mut() {
        let [r, g, b, _a] = pixel.0;
        pixel.0 = [r, 255u8.saturating_sub(b), g, 255];
    }
    encode_single_mip_dds(&rgba, image_dds::ImageFormat::BC1RgbaUnorm)
}

/// Split a single-mip WG `_n` DDS into two glTF-conformant siblings:
///
/// 1. `_normal.dds` — BC7 with `B := sqrt(1 - X² - Y²)` (reconstructed
///    Z baked in). Standard tangent-space normal map.
/// 2. `_nbmask.dds` — BC4 single-channel, original B preserved (the
///    categorical no-camo region marker that drives the camo gate).
///
/// The split is the disk-side counterpart to
/// [`replace_normal_b_with_reconstructed_z_png`] (which fixes the
/// GLB-embed path). Callers in `RawDdsDumper` invoke this once per
/// mip-level file so the conformant siblings mirror WG's per-mip-file
/// convention.
///
/// See `tools/reference/shared/texture_conventions.md` §Decision in the
/// pipeline repo for the architectural rationale.
pub fn split_wg_normal_dds(
    dds_bytes: &[u8],
) -> Result<(Vec<u8> /* normal */, Vec<u8> /* mask */), Report<TextureError>> {
    let rgba = decode_single_mip_dds(dds_bytes)?;
    let (w, h) = (rgba.width(), rgba.height());

    let mut normal_buf: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    let mut mask_buf: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);

    for pixel in rgba.pixels() {
        let [r, g, b, _a] = pixel.0;

        // Reconstructed-Z normal: keep R, G; replace B with z; force A=255.
        let x = (r as f32 / 255.0) * 2.0 - 1.0;
        let y = (g as f32 / 255.0) * 2.0 - 1.0;
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        let bz = (((z + 1.0) * 0.5 * 255.0).round() as i32).clamp(0, 255) as u8;
        normal_buf.extend_from_slice(&[r, g, bz, 255]);

        // BC4 mask: encoder reads R only; pad G/B/A to keep the RGBA8
        // surface shape consistent (encoder ignores G/B/A for BC4R).
        mask_buf.extend_from_slice(&[b, 0, 0, 255]);
    }

    let normal_img = image_dds::image::RgbaImage::from_raw(w, h, normal_buf)
        .ok_or_else(|| Report::new(TextureError::DdsEncode("normal buf size mismatch".into())))?;
    let mask_img = image_dds::image::RgbaImage::from_raw(w, h, mask_buf)
        .ok_or_else(|| Report::new(TextureError::DdsEncode("mask buf size mismatch".into())))?;

    let normal_dds = encode_single_mip_dds(&normal_img, image_dds::ImageFormat::BC7RgbaUnorm)?;
    let mask_dds = encode_single_mip_dds(&mask_img, image_dds::ImageFormat::BC4RUnorm)?;

    Ok((normal_dds, mask_dds))
}

/// Try to load a texture for a model mesh, with TILEDLAND baking support.
///
/// Walk an arbitrary directory of WG-pack DDS files (player-authored
/// loose-mod skin packs are the canonical case) and emit glTF-conformant
/// siblings alongside each non-conformant file:
///
///   `<stem>_n.dd?`  → `<stem>_normal.dd?` + `<stem>_nbmask.dd?`
///   `<stem>_mg.dd?` → `<stem>_mr.dd?`
///
/// Other DDS files (`_a`, `_ao`, decorative atlases, etc.) are skipped.
/// The directory walk is non-recursive — callers that need recursion
/// pass each subdirectory separately or use `swizzle_dir_recursive`.
///
/// `output_dir` defaults to `input_dir` (in-place). Returns
/// `(processed, written)`: how many `_n` / `_mg` source files were
/// recognised, and how many sibling files were actually written
/// (existing siblings are skipped — idempotent).
///
/// This is the bulk-disk counterpart to [`RawDdsDumper`]'s
/// per-VFS-extract emit. The toolkit's `--raw-dds-dir` path on
/// `export-ship` / `export-model` already runs the equivalent emit
/// implicitly; the standalone `swizzle-dir` CLI subcommand exposes it
/// for arbitrary directories of pre-extracted DDS bundles (e.g.
/// content-SDK mod folders that bypass the VFS pipeline). See
/// `tools/toolkit_integration/skins_and_camos.md` §4 in the pipeline
/// repo for the loose-mod ingest context.
pub fn swizzle_dir(
    input_dir: &std::path::Path,
    output_dir: Option<&std::path::Path>,
) -> Result<(usize, usize), Report<TextureError>> {
    let output_dir = output_dir.unwrap_or(input_dir);
    let _ = std::fs::create_dir_all(output_dir);

    let mut processed: usize = 0;
    let mut written:   usize = 0;

    let entries = std::fs::read_dir(input_dir)
        .map_err(|e| {
            Report::new(TextureError::IoError(format!(
                "read_dir {} failed: {e}", input_dir.display()
            )))
        })?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(s) => s,
            None => continue,
        };

        // Identify the mip-level suffix (`.dd0` / `.dd1` / `.dd2` / `.dds`).
        let mip_suffix = match DDS_MIP_SUFFIXES
            .iter()
            .find(|s| filename.to_lowercase().ends_with(*s))
        {
            Some(s) => *s,
            None => continue,
        };
        let stem_no_mip = &filename[..filename.len() - mip_suffix.len()];

        // Read source bytes.
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  Warning: failed to read {}: {e}", path.display());
                continue;
            }
        };

        if let Some(stem) = stem_no_mip.strip_suffix("_n") {
            processed += 1;
            let normal_name = format!("{stem}_normal{mip_suffix}");
            let mask_name   = format!("{stem}_nbmask{mip_suffix}");
            let normal_out  = output_dir.join(&normal_name);
            let mask_out    = output_dir.join(&mask_name);
            let need_normal = !normal_out.exists();
            let need_mask   = !mask_out.exists();
            if !need_normal && !need_mask { continue; }
            match split_wg_normal_dds(&bytes) {
                Ok((normal_dds, mask_dds)) => {
                    if need_normal {
                        if let Err(e) = std::fs::write(&normal_out, &normal_dds) {
                            eprintln!("  Warning: failed to write {}: {e}", normal_out.display());
                        } else { written += 1; }
                    }
                    if need_mask {
                        if let Err(e) = std::fs::write(&mask_out, &mask_dds) {
                            eprintln!("  Warning: failed to write {}: {e}", mask_out.display());
                        } else { written += 1; }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to split WG normal {filename}: {e:?}"
                    );
                }
            }
        } else if let Some(stem) = stem_no_mip.strip_suffix("_mg") {
            processed += 1;
            let mr_name = format!("{stem}_mr{mip_suffix}");
            let mr_out  = output_dir.join(&mr_name);
            if mr_out.exists() { continue; }
            match swizzle_wg_mg_dds_to_mr(&bytes) {
                Ok(mr_dds) => {
                    if let Err(e) = std::fs::write(&mr_out, &mr_dds) {
                        eprintln!("  Warning: failed to write {}: {e}", mr_out.display());
                    } else { written += 1; }
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to swizzle WG MG {filename}: {e:?}"
                    );
                }
            }
        }
    }

    Ok((processed, written))
}

/// Recursive variant of [`swizzle_dir`]. Walks `input_dir` and every
/// subdirectory. When `output_dir` is provided the relative tree is
/// preserved under it; when `None` siblings land in-place beside their
/// originals.
pub fn swizzle_dir_recursive(
    input_dir: &std::path::Path,
    output_dir: Option<&std::path::Path>,
) -> Result<(usize, usize), Report<TextureError>> {
    let mut total_processed: usize = 0;
    let mut total_written:   usize = 0;

    // Stack-based walk, deterministic per filesystem order.
    let mut stack = vec![input_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let out_for_dir = output_dir.map(|root| {
            let rel = dir.strip_prefix(input_dir).unwrap_or(std::path::Path::new(""));
            root.join(rel)
        });
        let (p, w) = swizzle_dir(&dir, out_for_dir.as_deref())?;
        total_processed += p;
        total_written += w;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() { stack.push(path); }
            }
        }
    }
    Ok((total_processed, total_written))
}

/// If assets.bin is available, first parses the MFM to check if it's a TILEDLAND
/// terrain material. If so, bakes a composite albedo from the tile atlas + blend
/// map (the correct rendering). Otherwise falls back to simple filename-based
/// texture lookup via `load_base_albedo_bytes`.
///
/// Returns PNG bytes if successful.
pub fn load_or_bake_albedo(
    vfs: &vfs::VfsPath,
    mfm_full_path: &str,
    mfm_path_id: u64,
    db: Option<&PrototypeDatabase<'_>>,
    self_id_index: Option<&HashMap<u64, usize>>,
    max_size: Option<u32>,
) -> Option<Vec<u8>> {
    // Try MFM-based TILEDLAND baking first (terrain materials).
    // This must come before filename-based lookup because _od files exist for
    // TILEDLAND tiles but are overlay maps, not standalone albedo textures.
    if let Some(db) = db
        && let Some(idx) = self_id_index
        && mfm_path_id != 0
        && let Some(mat) = parse_mfm_from_db(db, mfm_path_id)
        && is_tiledland_material(&mat)
    {
        eprintln!("  Baking TILEDLAND texture for: {mfm_full_path}");
        if let Some(png) = bake_tiledland_albedo(&mat, vfs, db, idx, max_size) {
            return Some(png);
        }
        eprintln!("    Warning: TILEDLAND bake failed, falling back to filename lookup");
    }

    // Fall back to simple filename-based lookup (works for standard PBS materials).
    // Force alpha=255 since model albedo textures often store non-opacity data
    // (height, roughness) in the alpha channel which would cause unwanted transparency.
    let dds_bytes = load_base_albedo_bytes(vfs, mfm_full_path, None)?;
    let mut png = dds_to_png_resized(&dds_bytes, max_size).ok()?;
    force_png_opaque(&mut png);
    Some(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_year_token_middle() {
        // Baltimore: year sits between ship + part.
        assert_eq!(
            strip_year_token("ASC017_Baltimore_1944_Bow").as_deref(),
            Some("ASC017_Baltimore_Bow"),
        );
        // Yamato: year between ship + part.
        assert_eq!(
            strip_year_token("JSB039_Yamato_1945_Hull").as_deref(),
            Some("JSB039_Yamato_Hull"),
        );
    }

    #[test]
    fn strip_year_token_trailing() {
        // Year at the end of the stem (no part suffix).
        assert_eq!(
            strip_year_token("ASB017_Montana_1945").as_deref(),
            Some("ASB017_Montana"),
        );
    }

    #[test]
    fn strip_year_token_no_year() {
        // No 4-digit token in stem.
        assert_eq!(strip_year_token("ASB017_Montana_Deckhouse"), None);
        // 2- and 3-digit numbers must not match (calibers, model codes).
        assert_eq!(strip_year_token("AGM034_16in50_Mk7_skinned"), None);
        assert_eq!(strip_year_token("AGS010_5in38_Mk32_Mod12"), None);
        assert_eq!(strip_year_token("AGM019_8in55_CA68"), None);
    }

    #[test]
    fn strip_year_token_5digit_not_year() {
        // 5-digit number is not a year, must not match.
        assert_eq!(strip_year_token("AB12345_Test"), None);
    }

    #[test]
    fn truncate_at_year_token_basic() {
        assert_eq!(
            truncate_at_year_token("ASC017_Baltimore_1944_Bow").as_deref(),
            Some("ASC017_Baltimore"),
        );
        assert_eq!(
            truncate_at_year_token("JSB039_Yamato_1945_Hull").as_deref(),
            Some("JSB039_Yamato"),
        );
        // Year at end: still produces the prefix.
        assert_eq!(
            truncate_at_year_token("JSB039_Yamato_1945").as_deref(),
            Some("JSB039_Yamato"),
        );
    }

    #[test]
    fn truncate_at_year_token_no_year() {
        assert_eq!(truncate_at_year_token("ASB017_Montana_Deckhouse"), None);
    }

    #[test]
    fn texture_base_names_baltimore_camo_pattern() {
        // The motivating case: WG ships Baltimore's camo as
        // `ASC017_Baltimore_camo_01.dd0` even though the MFM stems
        // carry `_1944`. The candidate list must include the bare
        // index prefix `ASC017_Baltimore` so the camo file is found.
        let names = texture_base_names("ASC017_Baltimore_1944_Bow");
        assert!(
            names.contains(&"ASC017_Baltimore".to_string()),
            "expected ASC017_Baltimore in candidate list, got {names:?}",
        );
        assert!(
            names.contains(&"ASC017_Baltimore_Bow".to_string()),
            "expected ASC017_Baltimore_Bow in candidate list, got {names:?}",
        );
        // Original is always first.
        assert_eq!(names.first().map(String::as_str), Some("ASC017_Baltimore_1944_Bow"));
    }

    #[test]
    fn texture_base_names_montana_passthrough() {
        // Montana has no `_1945` in its texture stems, so the
        // year-strippers don't fire. Original stem stays as the only
        // candidate.
        let names = texture_base_names("ASB017_Montana_Deckhouse");
        assert_eq!(names, vec!["ASB017_Montana_Deckhouse".to_string()]);
    }

    #[test]
    fn texture_base_names_yamato_full() {
        // Yamato's hull MFM carries `_1945`. Year-strip should produce
        // both `JSB039_Yamato_Hull` and `JSB039_Yamato` so any of WG's
        // naming variants resolve.
        let names = texture_base_names("JSB039_Yamato_1945_Hull");
        assert!(names.contains(&"JSB039_Yamato_1945_Hull".to_string()));
        assert!(names.contains(&"JSB039_Yamato_Hull".to_string()));
        assert!(names.contains(&"JSB039_Yamato".to_string()));
    }

    #[test]
    fn texture_base_names_skinned_turret() {
        // Existing _skinned strip still fires for accessory MFMs.
        let names = texture_base_names("AGM034_16in50_Mk7_skinned");
        assert_eq!(names.first().map(String::as_str), Some("AGM034_16in50_Mk7_skinned"));
        assert!(names.contains(&"AGM034_16in50_Mk7".to_string()));
    }

    #[test]
    fn texture_base_names_no_dup_when_year_strip_matches_original() {
        // Pathological: stem with no year shouldn't grow the list.
        let names = texture_base_names("ASC017_Baltimore");
        assert_eq!(names, vec!["ASC017_Baltimore".to_string()]);
    }
}
