//! High-level ship model export API.
//!
//! Provides [`ShipAssets`] (shared expensive resources, created once) and
//! [`ShipModelContext`] (a fully-loaded ship, ready for GLB export).
//!
//! # Quick start
//! ```no_run
//! use wowsunpack::export::ship::{ShipAssets, ShipExportOptions};
//! # fn main() -> rootcause::Result<()> {
//! # let vfs: vfs::VfsPath = todo!();
//! let assets = ShipAssets::load(&vfs)?;
//! let ctx = assets.load_ship("Yamato", &ShipExportOptions::default())?;
//! let mut file = std::fs::File::create("yamato.glb")?;
//! ctx.export_glb(&mut file)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use rootcause::prelude::*;
use vfs::VfsPath;

use crate::data::ResourceLoader;
use crate::game_params::keys;
use crate::game_params::provider::GameMetadataProvider;
use crate::game_params::types::ArmorMap;
use crate::game_params::types::GameParamProvider;
use crate::game_params::types::MountPoint;
use crate::game_params::types::Vehicle;
use crate::models::assets_bin;
use crate::models::assets_bin::PrototypeDatabase;
use crate::models::geometry;
use crate::models::model;
use crate::models::visual;
use crate::models::visual::VisualPrototype;

use super::camouflage;
use super::camouflage::CamouflageDb;
use super::gltf_export;
use super::gltf_export::InteractiveArmorMesh;
use super::gltf_export::SubModel;
use super::gltf_export::TextureSet;
use super::texture;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which mounted accessories to embed in a ship-level export.
///
/// Driven by `MountSpecies::display_group`; the matching group names are in
/// [`mount_group`]. Hull sub-models are always emitted regardless of mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AccessoryMode {
    /// Embed every mount's mesh (and its per-mount armor zones) in the GLB.
    /// Historical default — produces a fully-self-contained ship with every
    /// secondary turret, AA gun, director, etc. baked in.
    Embed,
    /// Embed only Main Battery mounts. Use when Main guns are manually rigged
    /// per-ship in Blender and everything else comes from a shared-asset
    /// library at Unity import time.
    MainOnly,
    /// Emit only hull parts (and their armor) — no mount meshes, no per-mount
    /// armor. Companion placement data for the sidecar's `accessories[]`
    /// section is assumed to come from a separate call (or from the visual's
    /// node tree walk downstream).
    Exclude,
}

impl AccessoryMode {
    /// Return true if this group should be embedded under the mode.
    /// `display_group` is [`MountSpecies::display_group`]'s string.
    pub fn includes_group(self, display_group: &str) -> bool {
        match self {
            AccessoryMode::Embed => true,
            AccessoryMode::MainOnly => display_group == "Main Battery",
            AccessoryMode::Exclude => false,
        }
    }
}

/// Options controlling ship model export.
#[derive(Debug, Clone)]
pub struct ShipExportOptions {
    /// LOD level (0 = highest detail). Default: 0. Ignored when
    /// `all_render_sets` is true.
    pub lod: usize,
    /// Hull upgrade selection. `None` = first/stock hull.
    /// Accepts full upgrade name (e.g. "PJUH911_Yamato_1944") or a prefix
    /// match against the hull component name (e.g. "B").
    pub hull: Option<String>,
    /// Whether to embed textures in the GLB. Default: true.
    pub textures: bool,
    /// Export the damaged/destroyed hull state instead of intact.
    /// When true, crack geometry is included and patch geometry is excluded.
    /// Default: false (intact hull). Ignored when `all_render_sets` is true.
    pub damaged: bool,
    /// Bundle every render set in the visual as its own named glTF mesh —
    /// all LODs + both damage states at once. Render-set names encode LOD
    /// level (`_lod1` / `_lod2` / `_lod3`) and damage state (`_crack_` /
    /// `_patch_`); downstream consumers filter by name. `_hide` is still
    /// excluded. Default: false (preserves historical single-LOD behaviour).
    pub all_render_sets: bool,
    /// Which accessory mounts to embed in the ship GLB. Default:
    /// [`AccessoryMode::Embed`] — every mount is baked in. Use
    /// [`AccessoryMode::MainOnly`] when main turrets are rigged per-ship
    /// (Blender side) but secondaries / AAs / directors come from a
    /// shared-asset library at import time; use [`AccessoryMode::Exclude`]
    /// for a pure hull+armor export.
    pub accessory_mode: AccessoryMode,
    /// Module overrides: component type key (e.g. "artillery") to component name.
    /// Overrides the default component for specific types.
    pub module_overrides: std::collections::HashMap<crate::game_params::keys::ComponentType, String>,
    /// If set, write a JSON manifest of every resolved mount placement to this
    /// path alongside the GLB. Emitted regardless of [`accessory_mode`] — the
    /// placements JSON is the canonical source for the Unity sidecar's typed
    /// sections (turrets / secondaries / antiair / torpedoes / accessories),
    /// so it stays complete even when the GLB omits accessory meshes.
    ///
    /// Field contract: see `run_export_ship` docs in the CLI.
    pub placements_json_path: Option<PathBuf>,
    /// If set, write textures as PNG files to this directory and reference
    /// them via URIs in the glTF, instead of embedding them in the GLB's
    /// BIN chunk. Useful for the shared accessory library where many ships
    /// should reference one on-disk copy of each texture. Default: `None`
    /// (embedded, historical behaviour).
    pub textures_dir: Option<PathBuf>,
    /// Prefix prepended to every texture filename when emitted as a URI in
    /// the glTF (e.g. `"textures/"`). Must include the trailing slash if a
    /// subdirectory is intended. Only used when `textures_dir` is `Some`.
    /// The default `"textures/"` works when the textures dir is placed as a
    /// `textures/` sub-directory next to the GLB.
    pub textures_uri_prefix: String,
    /// If set, dump raw WG DDS files (every available mip level —
    /// `.dd0`, `.dd1`, `.dd2`, `.dds`) to this directory as a parallel
    /// data stream. Preserves WG filenames verbatim so downstream
    /// consumers (Unity's DDS importer, Texture Streaming) see the full
    /// mip chain in BC-compressed form. Independent of `textures_dir`
    /// (PNG export for glTF): both can be set simultaneously.
    pub raw_dds_dir: Option<PathBuf>,
    /// If set, write a JSON manifest of every material → texture stem
    /// mapping to this path. Walks each hull-part `.visual`'s render sets,
    /// resolves the `.mfm` for each, and resolves every texture-typed
    /// MFM property to a VFS path. Lets downstream pipelines bind
    /// materials to DDS stems deterministically rather than via
    /// filename heuristics. Emits regardless of [`accessory_mode`] /
    /// [`textures`].
    ///
    /// Field contract: see `ShipModelContext::write_material_mappings_json`.
    pub material_mappings_json_path: Option<PathBuf>,
}

impl Default for ShipExportOptions {
    fn default() -> Self {
        Self {
            lod: 0,
            hull: None,
            textures: true,
            damaged: false,
            all_render_sets: false,
            accessory_mode: AccessoryMode::Embed,
            module_overrides: std::collections::HashMap::new(),
            placements_json_path: None,
            textures_dir: None,
            textures_uri_prefix: "textures/".to_string(),
            raw_dds_dir: None,
            material_mappings_json_path: None,
        }
    }
}

/// Resolved ship identity information.
#[derive(Debug, Clone)]
pub struct ShipInfo {
    /// Model directory name, e.g. "JSB039_Yamato_1945".
    pub model_dir: String,
    /// Translated display name if translations are loaded, e.g. "Yamato".
    pub display_name: Option<String>,
    /// GameParam index key, e.g. "PJSB018".
    pub param_index: String,
    /// Nation tag from GameParams typeinfo (e.g. "usa", "japan"). Empty if unresolved.
    pub nation: String,
    /// Species / class name from GameParams (e.g. "Destroyer", "Battleship",
    /// "Cruiser", "AirCarrier"). Empty if unresolved.
    pub species: String,
    /// Tier / level from GameParams (1-11 for current WoWS tiers; 0 if unresolved).
    pub tier: u32,
}

// ---------------------------------------------------------------------------
// Skel_ext helpers — shared between `export-ship` and `export-model`
// ---------------------------------------------------------------------------

/// Find all `.skel_ext` VFS paths whose parent directory matches `model_dir`.
///
/// For ships, `model_dir` is the hull directory (e.g. `ASB017_Montana_1945`)
/// and a typical match yields 8 entries: 4 base sections (Bow / MidFront /
/// MidBack / Stern) + 4 `_ep` variants. For accessories, `model_dir` is the
/// asset stem (e.g. `AGM034_16in50_Mk7`) and a typical match yields 1 entry.
///
/// The returned tuples are `(segment_name, vfs_path)`. Segment naming:
/// the stem with `model_dir + "_"` stripped — for ships this gives `Bow`,
/// `MidFront_ep`, etc.; for accessories the strip is a no-op (the stem
/// equals `model_dir`) and the segment becomes the full stem (e.g.
/// `AGM034_16in50_Mk7`).
pub fn find_skel_ext_paths(
    db: &PrototypeDatabase<'_>,
    self_id_index: &HashMap<u64, usize>,
    model_dir: &str,
) -> Vec<(String, String)> {
    let needle = format!("/{model_dir}/");
    let mut result = Vec::new();
    for (i, entry) in db.paths_storage.iter().enumerate() {
        if !entry.name.ends_with(".skel_ext") {
            continue;
        }
        let full_path = db.reconstruct_path(i, self_id_index);
        if full_path.contains(&needle) {
            let stem_with_ext = entry.name.as_str();
            let stem = stem_with_ext
                .strip_suffix(".skel_ext")
                .unwrap_or(stem_with_ext);
            let prefix = format!("{model_dir}_");
            let segment = stem.strip_prefix(&prefix).unwrap_or(stem).to_string();
            result.push((segment, full_path));
        }
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Load `.skel_ext` bytes for each `(segment, vfs_path)` pair.
///
/// Per-file failures (bad VFS join, missing file, read error) emit a
/// warning to stderr and are skipped — the caller gets a possibly-shorter
/// vec rather than a hard failure. This matches the existing `export-ship`
/// behaviour where one bad segment shouldn't doom the whole export.
pub fn load_skel_ext_files(
    vfs: &VfsPath,
    skel_ext_paths: &[(String, String)],
) -> Result<Vec<OwnedSkelExt>, Report> {
    let mut out = Vec::new();
    for (segment, vfs_path) in skel_ext_paths {
        let vfs_joined = match vfs.join(vfs_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: skipping skel_ext '{vfs_path}' (bad VFS join): {e}");
                continue;
            }
        };
        let mut file = match vfs_joined.open_file() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Warning: skipping skel_ext '{vfs_path}' (could not open): {e}");
                continue;
            }
        };
        let mut bytes = Vec::new();
        if let Err(e) = file.read_to_end(&mut bytes) {
            eprintln!("Warning: failed to read skel_ext '{vfs_path}': {e}");
            continue;
        }
        out.push(OwnedSkelExt {
            segment: segment.clone(),
            vfs_path: vfs_path.clone(),
            bytes,
        });
    }
    Ok(out)
}

/// Write a companion `<subject>.skel_ext_candidates.json` from a slice of
/// loaded `.skel_ext` files.
///
/// Free-function variant of the writer used by both `export-ship` and
/// `export-model`. The output schema is documented at length on
/// [`ShipModelContext::write_skel_ext_candidates_json`]; this function
/// produces the same per-record dedupe + manifest header, with the
/// subject-specific metadata block driven by `subject` (ship-style
/// nation/species/tier or model-style stem).
///
/// `bone_visual` is optional context used to compose each placement's
/// parent-bone rest pose into the emitted matrix (schema_version=2,
/// principled emit). When provided, each placement's `p1_hash` is
/// resolved against the visual's node tree and the matched bone's
/// composed-to-root rest pose is composed onto the placement matrix
/// before basis conversion. When `None`, falls back to the legacy
/// emit (schema_version=1) which left bone-rest composition to the
/// consumer. New consumers should require `schema_version >= 2`.
#[cfg(feature = "json")]
pub fn write_skel_ext_candidates_json(
    skel_ext_files: &[OwnedSkelExt],
    subject: &SkelExtSubject<'_>,
    path: &Path,
    bone_visual: Option<(&VisualPrototype, &crate::models::assets_bin::StringsSection<'_>)>,
) -> Result<(), Report> {
    use serde_json::json;

    let subject_block = match subject {
        SkelExtSubject::Ship(info) => json!({
            "model_dir":    info.model_dir,
            "display_name": info.display_name,
            "nation":       info.nation,
            "species":      info.species,
            "tier":         info.tier,
        }),
        SkelExtSubject::Model { stem, display_name } => json!({
            "model_dir":    stem,
            "display_name": display_name,
        }),
    };
    let subject_key = match subject {
        SkelExtSubject::Ship(_) => "ship",
        SkelExtSubject::Model { .. } => "model",
    };

    // schema_version=3 (2026-05-10): the Ry(180°) factor on each placement
    // is now conditional on the asset's `Rotate_Y_BlendBone` rest pose. WG
    // ships two valid (bone, mesh) authoring conventions:
    //   A. Z-mirror BlendBone (`det < 0`) + +Z mesh barrels  → POST-multiply Ry(180°)
    //   B. Identity BlendBone (`det >= 0`) + -Z mesh barrels → PRE-multiply  Ry(180°)
    //
    // The schema_v2 unconditional POST-Ry(180°) was calibrated on convention A
    // (covers Iowa/Baltimore/Atago and most stock turrets) but over-flips
    // accessories on convention B (Ohio's AGM182_18in48_MK1_Twin, Fletcher's
    // AGS001 — anywhere the host's BlendBone is identity, since
    // `mount.transform = HP · ensure_proper(inv(M_BB))` is then identity-on-M_BB
    // and absorbs no Ry(180°)).
    //
    // Why pre-multiply for convention B (not just drop the factor):
    //
    // The WG runtime model is `world_sub = mount.transform · bone_composed
    // · pl.matrix · pl_local`. Convention A has `mount.transform = HP · Ry(180°)`,
    // which mirrors the placement's authored (-X, -Z) values through the turret
    // pivot, landing decoratives on the breech side. Convention B has
    // `mount.transform = HP · I`, which carries no such mirror. WG authors
    // both conventions' `.skel_ext` data with the same (-X, -Z) sign convention
    // (verified empirically: AGM034 / AGM019 / AGM182 all author boats at
    // negative composed_pos) — so without an explicit Ry(180°) at the toolkit
    // emit, convention-B accessories land on the muzzle side instead of the
    // breech side. Pre-multiply restores the mirror through the pivot Y axis.
    //
    // A first cut of schema_v3 (2026-05-10 03:00) DROPPED the Ry(180°) entirely
    // for convention B — that fixed orientation but left positions on the
    // muzzle side. The pre-multiply correction is the universal form that
    // addresses both at once (mirror = position + orientation flipped).
    //
    // The det<0 detection is a single bit of conditional, fed by the asset's
    // own `.visual` data — no per-asset list, no swap-pair table. Output is
    // byte-identical to schema_v2 for every convention-A asset.
    let schema_version = if bone_visual.is_some() { 3 } else { 1 };

    if skel_ext_files.is_empty() {
        let manifest = json!({
            "schema_version": schema_version,
            subject_key: subject_block,
            "stats": { "file_count": 0, "candidate_count": 0 },
            "candidates": [],
        });
        let file = std::fs::File::create(path)
            .context_with(|| format!("Failed to create skel_ext candidates at {}", path.display()))?;
        serde_json::to_writer(std::io::BufWriter::new(file), &manifest)
            .map_err(|e| rootcause::report!("Failed to serialize skel_ext candidates: {e}"))?;
        return Ok(());
    }

    let mut global_placements: Vec<(String, crate::models::skel_ext::SkelExtPlacement)> = Vec::new();
    for skel_file in skel_ext_files {
        let placements = match crate::models::skel_ext::parse_skel_ext(&skel_file.bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "Warning: could not parse skel_ext '{}' (segment {}): {}",
                    skel_file.vfs_path, skel_file.segment, e
                );
                continue;
            }
        };
        for pl in placements {
            global_placements.push((skel_file.segment.clone(), pl));
        }
    }

    // Dedupe globally at 1 cm on (p0_hash, native-coord position). See
    // `ShipModelContext::write_skel_ext_candidates_json` for rationale.
    let mut seen: HashSet<(u32, i64, i64, i64)> = HashSet::new();
    let mut candidates: Vec<serde_json::Value> = Vec::new();

    // schema_version=2/3 emit cache: per-bone composed rest poses (avoids
    // re-walking the parent chain for every placement that shares a
    // bone — typical .skel_ext has 99% of placements rooted on the
    // same `Rotate_Y`).
    let mut bone_rest_cache: HashMap<u32, Option<[f32; 16]>> = HashMap::new();
    let mut unresolved_p1: HashMap<u32, u32> = HashMap::new();

    // schema_version=3: the post-Ry(180°) is gated on the asset's
    // `Rotate_Y_BlendBone` det sign. Read it once per asset (None when
    // the asset has no BB — AA mounts, torpedo tubes, depth charges —
    // which fall back to identity → no Ry(180°), matching their
    // `mount.transform = HP` parent placement).
    let bb_is_z_mirror: bool = bone_visual
        .and_then(|(vp, strings)| vp.find_node_local_matrix("Rotate_Y_BlendBone", strings))
        .map(|bb_local| mat3_determinant(&bb_local) < 0.0)
        .unwrap_or(false);

    for (segment, pl) in &global_placements {
        let p = pl.position();
        let key = (
            pl.p0_hash,
            (p[0] * 100.0).round() as i64,
            (p[1] * 100.0).round() as i64,
            (p[2] * 100.0).round() as i64,
        );
        if !seen.insert(key) {
            continue;
        }

        let emit_matrix = if let Some((vp, strings)) = bone_visual {
            // schema_version=2/3: compose parent-bone rest pose into the
            // placement, conditionally post-multiply by Ry(180°) (v3:
            // only when the asset's `Rotate_Y_BlendBone` is Z-mirror;
            // v2 was unconditional), scale translation ×15. This produces
            // a matrix the consumer can decompose verbatim — no further
            // Z-mirror, Ry(180°) flip, or Y bone-rest offset needed at
            // the consumer side.
            //
            // Why these steps fold into one toolkit-side emit:
            //   - bone composition: WG runtime parents the placement
            //     under the parent bone (typically `Rotate_Y` for guns,
            //     the asset's own root node for directors). Without
            //     composing the bone rest, the placement lands at the
            //     bone's position rather than at the asset root.
            //   - Conditional Ry(180°) post-multiply: when the asset's
            //     `mount.transform = HP · ensure_proper(inv(M_BB))` is
            //     `HP · Ry(180°)` (M_BB is Z-mirror), the post-Ry(180°)
            //     on the sub-matrix composes with the host's Ry(180°)
            //     rotation to produce the visually-correct orientation.
            //     When `mount.transform = HP` (M_BB is identity), there's
            //     no host Ry(180°) to compose with, so the sub-matrix
            //     must NOT carry one either or directional accessories
            //     end up 180° flipped. The det<0 test is the single bit
            //     that distinguishes the two cases.
            //   - Skip negate_z entirely: the previous schema_version=1
            //     emit applied `S·M·S` which the consumer then had to
            //     undo with another `S·M·S` (composing to identity).
            //     Eliminating both halves of the cancellation lets us
            //     incorporate the `Ry(180°)` cleanly without the
            //     consumer needing handedness-conversion math.
            let bone_rest = bone_rest_cache
                .entry(pl.p1_hash)
                .or_insert_with(|| vp.find_composed_matrix_by_hash(pl.p1_hash, strings));
            match bone_rest {
                Some(rest) => {
                    // composed_native = bone_rest @ pl.matrix
                    let composed = mat4_mul_col_major(rest, &pl.matrix);
                    // Convention-A (Z-mirror M_BB): POST-multiply Ry(180°).
                    // Matches schema_v2 byte-equal — mount.transform carries
                    // its own Ry(180°) (from HP · ensure_proper(inv(M_BB))),
                    // and the post-Ry on sub composes with that to land
                    // boats / periscopes on the breech side of the turret.
                    //
                    // Convention-B (identity M_BB): mirror only the X+Z
                    // translation through the turret pivot Y axis; leave
                    // rotation untouched.
                    //
                    //   composed = R_c | t_c    →    emit = R_c | (−t_c.x, t_c.y, −t_c.z)
                    //
                    // Why position-only and not pre-multiply Ry(180°):
                    //
                    // The WG-authored .skel_ext data uses the same negative-
                    // X/negative-Z sign convention for both A and B turrets
                    // (verified: AGM034 boat composed_pos (-5.63, 1.64, -4.56);
                    // AGM182 boat composed_pos (-4.70, 1.82, -3.24)). Convention
                    // A's `mount.transform = HP · Ry(180°)` mirrors this through
                    // the pivot, putting boats on the breech side; convention B's
                    // `mount.transform = HP · I` carries no such mirror, so the
                    // placement-side has to do it. Post-mul on the sub matrix
                    // doesn't affect col_3 (translation), so we cannot use the
                    // schema_v2 form — we have to reach into col_3 directly.
                    //
                    // Pre-multiplying Ry(180°) (an earlier candidate) WOULD
                    // mirror col_3, but it ALSO negates rotation rows 0+2,
                    // which over-rotates assets whose mesh-local origin is
                    // not at the bone center. Example: AM788_Rangefinder is
                    // authored with mesh extents at Z ∈ [+6.24, +8.54] (the
                    // optic head sits 6-9 m forward of the placement origin)
                    // while composed_pos = (0, 0.09, 0). Pre-mul would leave
                    // its position at the pivot but rotate it 180° around Y,
                    // sending the mesh into world −Z (muzzle side). Position-
                    // only flip leaves the rotation as identity, so the mesh
                    // extent stays in world +Z (breech side, correct).
                    //
                    // For Y-symmetric decoratives (AM055 boats are X+Z
                    // mirror-symmetric, AM777 periscopes are X+Z symmetric)
                    // the rotation choice is invisible — both pre-mul and
                    // position-only render the same. Position-only is the
                    // strict refinement that doesn't break the Z-asymmetric
                    // assets like AM788.
                    let final_unscaled = if bb_is_z_mirror {
                        mat4_mul_col_major(&composed, &RY_180_4X4)
                    } else {
                        let mut m = composed;
                        m[12] = -m[12]; // negate world-x of col_3
                        m[14] = -m[14]; // negate world-z of col_3
                        m
                    };
                    // Scale translation × 15 (native → metres). No
                    // basis conversion: the matrix is in WG-native LH
                    // coordinates, but the asset's GLB vertices were
                    // also negate_z'd, so the LH placement matches the
                    // negate_z'd GLB.
                    let mut m = final_unscaled;
                    m[12] *= crate::models::skel_ext::NATIVE_TO_METRES;
                    m[13] *= crate::models::skel_ext::NATIVE_TO_METRES;
                    m[14] *= crate::models::skel_ext::NATIVE_TO_METRES;
                    m
                }
                None => {
                    // Unresolved p1_hash — bone not in this visual.
                    // Track for diagnostics; fall back to the legacy
                    // basis conversion so the placement at least lands
                    // at the right position for assets where the
                    // unknown bone happens to be at identity rest.
                    *unresolved_p1.entry(pl.p1_hash).or_insert(0) += 1;
                    crate::models::skel_ext::to_metric_glft(pl.matrix)
                }
            }
        } else {
            // schema_version=1 legacy emit: just basis-convert the raw
            // placement matrix. Consumers must compose bone rests
            // themselves (or apply the historical 3-correction stack).
            crate::models::skel_ext::to_metric_glft(pl.matrix)
        };
        let position = [emit_matrix[12], emit_matrix[13], emit_matrix[14]];
        candidates.push(json!({
            "segment":       segment,
            "record_offset": format!("0x{:X}", pl.record_offset),
            "record_type":   pl.record_type,
            "record_count":  pl.record_count,
            "matrix_index":  pl.matrix_index,
            "p0_hash":       format!("0x{:08X}", pl.p0_hash),
            "p1_hash":       format!("0x{:08X}", pl.p1_hash),
            "transform": {
                "matrix":    emit_matrix.as_slice(),
                "position":  position.as_slice(),
            },
        }));
    }

    let now = format_rfc3339_utc(SystemTime::now());
    let toolkit_version = env!("CARGO_PKG_VERSION");

    // Sort unresolved p1 hashes by count (desc) for diagnostics. Most
    // commonly the unresolved set is empty (every bone the placements
    // reference exists in the visual) or a small number of legacy/
    // unused hashes.
    let mut unresolved_vec: Vec<(u32, u32)> = unresolved_p1.into_iter().collect();
    unresolved_vec.sort_by(|a, b| b.1.cmp(&a.1));
    let unresolved_json: Vec<serde_json::Value> = unresolved_vec
        .iter()
        .take(16)
        .map(|(hash, count)| json!({
            "p1_hash": format!("0x{:08X}", hash),
            "count":   count,
        }))
        .collect();

    let manifest = json!({
        "schema_version": schema_version,
        subject_key: subject_block,
        "pipeline": {
            "toolkit_version": toolkit_version,
            "generated_at":    now,
            "skel_ext_emit": match schema_version {
                3 => "bone_composed + {Ry(180°) post if det(M_BB)<0 else col_3 X+Z negate} + ×15 (post-2026-05-10)",
                2 => "bone_composed + Ry(180°) post + ×15 (2026-05-08; over-flips identity-BB hosts)",
                _ => "to_metric_glft (legacy: S·M·S + ×15)",
            },
            "host_bone_z_mirror": bb_is_z_mirror,
        },
        "stats": {
            "file_count":         skel_ext_files.len(),
            "raw_matrix_count":   global_placements.len(),
            "candidate_count":    candidates.len(),
            "position_tolerance_cm": 1,
            "segments": skel_ext_files.iter().map(|f| f.segment.as_str()).collect::<Vec<_>>(),
            "unresolved_p1_top": unresolved_json,
        },
        "candidates": candidates,
    });

    let file = std::fs::File::create(path)
        .context_with(|| format!("Failed to create skel_ext candidates at {}", path.display()))?;
    serde_json::to_writer(std::io::BufWriter::new(file), &manifest)
        .map_err(|e| rootcause::report!("Failed to serialize skel_ext candidates: {e}"))?;
    Ok(())
}

/// Summary of a hull upgrade for listing purposes.
#[derive(Debug, Clone)]
pub struct HullUpgradeInfo {
    /// Upgrade name (GameParam key), e.g. "PJUH911_Yamato_1944".
    pub name: String,
    /// Components in this upgrade: (type_key, component_name, mount_count).
    pub components: Vec<(String, String, usize)>,
}

// ---------------------------------------------------------------------------
// ShipAssets — shared expensive resources (created once)
// ---------------------------------------------------------------------------

/// Shared game assets for ship export operations.
///
/// Creating this is the expensive step (~18 seconds for GameParams parsing).
/// Reuse a single instance across multiple ship exports.
pub struct ShipAssets {
    assets_bin_bytes: Vec<u8>,
    vfs: VfsPath,
    metadata: Arc<GameMetadataProvider>,
    camo_db: Option<CamouflageDb>,
}

impl ShipAssets {
    /// Load shared assets from the VFS.
    ///
    /// This is expensive (~18 seconds) because it parses GameParams.
    /// Create once and reuse for multiple ships.
    pub fn load(vfs: &VfsPath) -> Result<Self, Report> {
        let mut assets_bin_bytes = Vec::new();
        vfs.join("content/assets.bin")
            .context("VFS path error")?
            .open_file()
            .context("Could not find content/assets.bin in VFS")?
            .read_to_end(&mut assets_bin_bytes)?;

        let metadata = Arc::new(GameMetadataProvider::from_vfs(vfs).context("Failed to load GameParams")?);

        let camo_db = CamouflageDb::load(vfs);

        Ok(Self { assets_bin_bytes, vfs: vfs.clone(), metadata, camo_db })
    }

    /// Load shared assets from the VFS, reusing an already-loaded [`GameMetadataProvider`].
    ///
    /// This skips the expensive GameParams parse that [`Self::load`] performs,
    /// making it suitable when the caller already has metadata available.
    pub fn from_vfs_with_metadata(vfs: &VfsPath, metadata: Arc<GameMetadataProvider>) -> Result<Self, Report> {
        let mut assets_bin_bytes = Vec::new();
        vfs.join("content/assets.bin")
            .context("VFS path error")?
            .open_file()
            .context("Could not find content/assets.bin in VFS")?
            .read_to_end(&mut assets_bin_bytes)?;

        let camo_db = CamouflageDb::load(vfs);

        Ok(Self { assets_bin_bytes, vfs: vfs.clone(), metadata, camo_db })
    }

    /// Load shared assets directly from a World of Warships installation directory.
    ///
    /// This is a convenience wrapper that builds the VFS (idx files + assets.bin overlay)
    /// from the game directory, then calls [`Self::load`]. It uses the latest build
    /// found in the `bin/` directory.
    ///
    /// For callers who already have a VFS, use [`Self::load`] instead.
    pub fn from_game_dir(game_dir: &Path) -> Result<Self, Report> {
        let vfs = crate::game_data::build_game_vfs(game_dir)?;
        Self::load(&vfs)
    }

    /// Set translations for display name resolution.
    pub fn set_translations(&self, catalog: gettext::Catalog) {
        self.metadata.set_translations(catalog);
    }

    /// Access the underlying `GameMetadataProvider`.
    pub fn metadata(&self) -> &GameMetadataProvider {
        &self.metadata
    }

    /// Access the underlying VFS root.
    pub fn vfs(&self) -> &VfsPath {
        &self.vfs
    }

    /// Find a ship by name (fuzzy display-name match or exact model dir).
    pub fn find_ship(&self, name: &str) -> Result<ShipInfo, Report> {
        let db = self.db()?;
        let self_id_index = db.build_self_id_index();

        // Strategy 1: try direct match against assets.bin paths.
        let needle = format!("/{name}/");
        let has_direct = db.paths_storage.iter().any(|e| {
            e.name.ends_with(".visual") && {
                // Reconstruct is expensive; just check if any visual file's
                // full path contains the needle. We only need one hit.
                let idx = db.paths_storage.iter().position(|x| std::ptr::eq(x, e)).unwrap();
                db.reconstruct_path(idx, &self_id_index).contains(&needle)
            }
        });

        if has_direct {
            // Direct model dir match — no GameParams needed for identity.
            // Try to find the GameParam for richer info.
            let param = self
                .metadata
                .params()
                .iter()
                .find(|p| p.vehicle().and_then(|v| v.model_path()).map(|mp| mp.contains(name)).unwrap_or(false));

            let (nation, species, tier) = ship_identity_from_param(param.map(|p| p.as_ref()));
            return Ok(ShipInfo {
                model_dir: name.to_string(),
                display_name: param.and_then(|p| self.metadata.localized_name_from_param(p)),
                param_index: param.map(|p| p.index().to_string()).unwrap_or_default(),
                nation,
                species,
                tier,
            });
        }

        // Strategy 2: exact param index match via GameParams.
        if let Some(param) = self.metadata.game_param_by_index(name)
            && let Some(vehicle) = param.vehicle()
            && let Some(model_path) = vehicle.model_path()
        {
            let dir = model_path.rsplit_once('/').map(|(d, _)| d).unwrap_or(model_path);
            let model_dir = dir.rsplit('/').next().unwrap_or(dir);
            let (nation, species, tier) = ship_identity_from_param(Some(&param));
            return Ok(ShipInfo {
                model_dir: model_dir.to_string(),
                display_name: self.metadata.localized_name_from_param(&param),
                param_index: param.index().to_string(),
                nation,
                species,
                tier,
            });
        }

        // Strategy 3: fuzzy display name match via GameParams.
        let normalized_input = unidecode::unidecode(name).to_lowercase();
        let mut matches: Vec<(String, String, String)> = Vec::new();

        for param in self.metadata.params() {
            let vehicle = match param.vehicle() {
                Some(v) => v,
                None => continue,
            };
            let model_path = match vehicle.model_path() {
                Some(p) => p,
                None => continue,
            };

            let display_name =
                self.metadata.localized_name_from_param(param).unwrap_or_else(|| param.index().to_string());

            let normalized_display = unidecode::unidecode(&display_name).to_lowercase();
            if normalized_display.contains(&normalized_input) {
                let dir = model_path.rsplit_once('/').map(|(d, _)| d).unwrap_or(model_path);
                let dir_name = dir.rsplit('/').next().unwrap_or(dir);
                matches.push((display_name, param.index().to_string(), dir_name.to_string()));
            }
        }

        // Helper closure: build a ShipInfo from a pick-one match + re-resolve
        // the param for nation/species/tier.
        let build_from_match = |display: &str, idx: &str, dir: &str| -> ShipInfo {
            let param = self.metadata.game_param_by_index(idx);
            let (nation, species, tier) = ship_identity_from_param(param.as_deref());
            ShipInfo {
                model_dir: dir.to_string(),
                display_name: Some(display.to_string()),
                param_index: idx.to_string(),
                nation,
                species,
                tier,
            }
        };

        match matches.len() {
            0 => bail!(
                "No ship found matching '{name}'. Try using the model directory name \
                 (e.g. 'JSB039_Yamato_1945')."
            ),
            1 => Ok(build_from_match(&matches[0].0, &matches[0].1, &matches[0].2)),
            _ => {
                // If all matches share the same model dir, use it.
                let unique_dirs: HashSet<&str> = matches.iter().map(|(_, _, d)| d.as_str()).collect();
                if unique_dirs.len() == 1 {
                    return Ok(build_from_match(&matches[0].0, &matches[0].1, &matches[0].2));
                }

                let listing: Vec<String> =
                    matches.iter().map(|(display, idx, dir)| format!("  {display} ({idx}) -> {dir}")).collect();
                bail!(
                    "Multiple ships match '{name}':\n{}\nPlease refine your search \
                     or use the model directory name directly.",
                    listing.join("\n")
                );
            }
        }
    }

    /// List hull upgrades for a ship.
    pub fn list_hull_upgrades(&self, name: &str) -> Result<Vec<HullUpgradeInfo>, Report> {
        let info = self.find_ship(name)?;
        let vehicle = self.find_vehicle(&info.model_dir)?;

        let Some(upgrades) = vehicle.hull_upgrades() else {
            return Ok(Vec::new());
        };

        let mut result = Vec::new();
        let mut sorted: Vec<_> = upgrades.iter().collect();
        sorted.sort_by_key(|(k, _)| (*k).clone());

        for (upgrade_name, config) in sorted {
            let mut components = Vec::new();
            for ct in keys::ComponentType::ALL {
                let comp = config.component_name(*ct).unwrap_or("(none)").to_string();
                let mount_count = config.mounts(*ct).map(|m| m.len()).unwrap_or(0);
                components.push((ct.to_string(), comp, mount_count));
            }
            result.push(HullUpgradeInfo { name: upgrade_name.clone(), components });
        }

        Ok(result)
    }

    /// List available camouflage texture schemes for a ship.
    pub fn list_texture_schemes(&self, name: &str) -> Result<Vec<String>, Report> {
        let info = self.find_ship(name)?;
        let db = self.db()?;
        let self_id_index = db.build_self_id_index();

        // Collect visuals for the ship model dir.
        let visual_paths = self.find_visual_paths(&db, &self_id_index, &info.model_dir);
        let sub_models = self.load_sub_models(&db, &self_id_index, &visual_paths)?;

        // Also load turret models to include their stems.
        let vehicle = self.find_vehicle(&info.model_dir).ok();
        let mount_points = vehicle
            .and_then(|v| self.select_hull_mount_points(v, None, &std::collections::HashMap::new()))
            .unwrap_or_default();
        let turret_data = self.load_turret_models(&db, &self_id_index, &mount_points)?;

        let mut all_stems = Vec::new();
        for smd in &sub_models {
            for mfm in collect_mfm_info(&smd.visual, &db) {
                all_stems.push(mfm.stem);
            }
        }
        for tmd in &turret_data {
            for mfm in collect_mfm_info(&tmd.visual, &db) {
                all_stems.push(mfm.stem);
            }
        }

        let mut schemes = texture::discover_texture_schemes(&self.vfs, &all_stems);

        // Also include material-based camo scheme display names.
        let ship_index = self.find_ship_index(&info.model_dir);
        let ship_idx = ship_index.as_deref();
        let mat_camos = self.discover_mat_camo_schemes(&info.model_dir, ship_idx);
        for scheme in &mat_camos {
            let tag = if scheme.tiled { "tiled" } else { "mat_camo" };
            schemes.push(format!("{} ({})", scheme.display_name, tag));
        }

        // Include universal camos (PCEC entries available to all ships).
        let universal = self.discover_universal_camo_schemes(ship_idx);
        for scheme in &universal {
            let tag = if scheme.tiled { "tiled" } else { "mat_camo" };
            schemes.push(format!("{} (universal/{})", scheme.display_name, tag));
        }

        Ok(schemes)
    }

    /// Load a complete ship model, ready for export.
    pub fn load_ship(&self, name: &str, options: &ShipExportOptions) -> Result<ShipModelContext, Report> {
        let info = self.find_ship(name)?;
        let vehicle = self.find_vehicle(&info.model_dir).ok();
        self.load_ship_inner(info, vehicle, options)
    }

    /// Load a ship using a [`Vehicle`] reference instead of a name lookup.
    ///
    /// This is useful when the caller already has a `Vehicle` from their own
    /// GameParams processing and wants to skip the name-based search.
    pub fn load_ship_from_vehicle(
        &self,
        vehicle: &Vehicle,
        options: &ShipExportOptions,
    ) -> Result<ShipModelContext, Report> {
        let model_path = vehicle.model_path().ok_or_else(|| rootcause::report!("Vehicle has no model_path"))?;
        // model_path is like "content/gameplay/nation/ship/DIR_NAME/file.model"
        let dir = model_path.rsplit_once('/').map(|(d, _)| d).unwrap_or(model_path);
        let model_dir = dir.rsplit('/').next().unwrap_or(dir);

        let param = self
            .metadata
            .params()
            .iter()
            .find(|p| p.vehicle().and_then(|v| v.model_path()).map(|mp| mp.contains(model_dir)).unwrap_or(false));

        let (nation, species, tier) = ship_identity_from_param(param.map(|p| p.as_ref()));
        let info = ShipInfo {
            model_dir: model_dir.to_string(),
            display_name: param.and_then(|p| self.metadata.localized_name_from_param(p)),
            param_index: param.map(|p| p.index().to_string()).unwrap_or_default(),
            nation,
            species,
            tier,
        };

        self.load_ship_inner(info, Some(vehicle), options)
    }

    fn load_ship_inner(
        &self,
        info: ShipInfo,
        vehicle: Option<&Vehicle>,
        options: &ShipExportOptions,
    ) -> Result<ShipModelContext, Report> {
        let db = self.db()?;
        let self_id_index = db.build_self_id_index();

        // Find all .visual files in the model directory.
        let visual_paths = self.find_visual_paths(&db, &self_id_index, &info.model_dir);
        if visual_paths.is_empty() {
            bail!("No .visual files found for '{}'.", info.model_dir);
        }

        // Load hull sub-models.
        let hull_parts = self.load_sub_models(&db, &self_id_index, &visual_paths)?;

        // Load per-segment `.skel_ext` files for decorative placements.
        // Silently yields an empty vec if a ship has no `.skel_ext` assets.
        let skel_ext_paths = self.find_skel_ext_paths(&db, &self_id_index, &info.model_dir);
        let skel_ext_files = self.load_skel_ext_files(&skel_ext_paths)?;

        // Load turret/mount models from GameParams.
        let mount_points: Vec<MountPoint> = vehicle
            .and_then(|v| self.select_hull_mount_points(v, options.hull.as_deref(), &options.module_overrides))
            .unwrap_or_default();

        let loaded = self.load_mounts(&db, &self_id_index, &mount_points, &hull_parts, &info.model_dir)?;
        let turret_models = loaded.turret_models;
        let mounts = loaded.mounts;

        // Resolve material-based camouflage schemes (ship-specific + universal).
        let ship_index = self.find_ship_index(&info.model_dir);
        let ship_idx = ship_index.as_deref();
        let mut mat_camo_schemes = self.discover_mat_camo_schemes(&info.model_dir, ship_idx);
        mat_camo_schemes.extend(self.discover_universal_camo_schemes(ship_idx));

        // Extract armor thickness map and hit locations from GameParams.
        let armor_map = vehicle.and_then(|v| v.armor().cloned());
        let hit_locations = vehicle.and_then(|v| v.hit_locations().cloned());

        Ok(ShipModelContext {
            vfs: self.vfs.clone(),
            assets_bin_bytes: self.assets_bin_bytes.clone(),
            hull_parts,
            turret_models,
            mounts,
            skel_ext_files,
            info,
            options: options.clone(),
            mat_camo_schemes,
            armor_map,
            hit_locations,
        })
    }

    // --- Internal helpers ---

    /// Discover material-based camo schemes available for a ship via GameParams.
    ///
    /// Follows: Vehicle.permoflages → Exterior.camouflage → camouflages.xml entry.
    /// Returns owned `MatCamoScheme` data (no lifetimes).
    fn discover_mat_camo_schemes(&self, model_dir: &str, ship_index: Option<&str>) -> Vec<MatCamoScheme> {
        let camo_db = match &self.camo_db {
            Some(db) => db,
            None => return Vec::new(),
        };
        let vehicle = match self.find_vehicle(model_dir) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let mut result = Vec::new();
        let mut seen_camo_names = HashSet::new();

        for permo_name in vehicle.permoflages() {
            // permoflages entries are param names (e.g. "PCEM017_Steel_10lvl"), not indices.
            let param =
                self.metadata.game_param_by_name(permo_name).or_else(|| self.metadata.game_param_by_index(permo_name));
            let Some(param) = param else {
                continue;
            };
            let Some(exterior) = param.exterior() else {
                continue;
            };
            let Some(camo_name) = exterior.camouflage() else {
                continue;
            };

            // Deduplicate by camo name (multiple exteriors can share the same camo).
            if !seen_camo_names.insert(camo_name.to_string()) {
                continue;
            }

            let Some(entry) = camo_db.get(camo_name, ship_index) else {
                continue;
            };
            if entry.textures.is_empty() {
                continue;
            }

            // Build display name from translation.
            // Exterior entries use IDS_{NAME_UPPER} as the translation key
            // (e.g. "PCEM017_Steel_10lvl" → "IDS_PCEM017_STEEL_10LVL").
            let ids_key = format!("IDS_{}", permo_name.to_uppercase());
            let display_name = self
                .metadata
                .localized_name_from_id(&ids_key)
                .or_else(|| {
                    // Fallback: try IDS_{index}
                    self.metadata.localized_name_from_param(&param)
                })
                .unwrap_or_else(|| camo_name.to_string());

            // Collect unique texture paths (most mat_camos reuse one texture for all parts).
            let mut unique_paths: Vec<String> = Vec::new();
            let mut seen_paths = HashSet::new();
            for path in entry.textures.values() {
                if seen_paths.insert(path.clone()) {
                    unique_paths.push(path.clone());
                }
            }

            // For tiled camos, resolve color scheme.
            let color_scheme_colors = if entry.tiled {
                entry.color_scheme.as_ref().and_then(|cs_name| camo_db.color_scheme(cs_name)).map(|cs| cs.colors)
            } else {
                None
            };

            result.push(MatCamoScheme {
                display_name,
                texture_paths: unique_paths,
                tiled: entry.tiled,
                color_scheme_colors,
                uv_transforms: entry.uv_transforms.clone(),
            });
        }

        result
    }

    /// Discover universal camouflage schemes (PCEC entries available to all ships).
    ///
    /// These are not referenced by any ship's `permoflages` list — they're
    /// universally applicable. Deduplicated by camouflage name.
    fn discover_universal_camo_schemes(&self, ship_index: Option<&str>) -> Vec<MatCamoScheme> {
        let camo_db = match &self.camo_db {
            Some(db) => db,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        let mut seen_camo_names = HashSet::new();

        for param in self.metadata.params() {
            let name = param.name();
            if !name.starts_with("PCEC") {
                continue;
            }
            let Some(exterior) = param.exterior() else {
                continue;
            };
            let Some(camo_name) = exterior.camouflage() else {
                continue;
            };
            if !seen_camo_names.insert(camo_name.to_string()) {
                continue;
            }

            let Some(entry) = camo_db.get(camo_name, ship_index) else {
                continue;
            };
            if entry.textures.is_empty() {
                continue;
            }

            let display_name = self
                .metadata
                .localized_name_from_id(&format!("IDS_{}", name.to_uppercase()))
                .unwrap_or_else(|| camo_name.to_string());

            let color_scheme_colors = if entry.tiled {
                entry.color_scheme.as_ref().and_then(|cs_name| camo_db.color_scheme(cs_name)).map(|cs| cs.colors)
            } else {
                None
            };

            let mut unique_paths: Vec<String> = Vec::new();
            let mut seen_paths = HashSet::new();
            for path in entry.textures.values() {
                if seen_paths.insert(path.clone()) {
                    unique_paths.push(path.clone());
                }
            }

            result.push(MatCamoScheme {
                display_name,
                texture_paths: unique_paths,
                tiled: entry.tiled,
                color_scheme_colors,
                uv_transforms: entry.uv_transforms.clone(),
            });
        }

        result
    }

    /// Re-parse the PrototypeDatabase from owned bytes.
    fn db(&self) -> Result<PrototypeDatabase<'_>, Report> {
        Ok(assets_bin::parse_assets_bin(&self.assets_bin_bytes).context("Failed to parse assets.bin")?)
    }

    /// Find a Vehicle by model directory name.
    fn find_vehicle(&self, model_dir: &str) -> Result<&crate::game_params::types::Vehicle, Report> {
        self.metadata
            .params()
            .iter()
            .filter_map(|p| p.vehicle())
            .find(|v| v.model_path().map(|mp| mp.contains(model_dir)).unwrap_or(false))
            .ok_or_else(|| rootcause::report!("Ship '{}' not found in GameParams", model_dir))
    }

    /// Find the ship param's index name (e.g. "PJSB018_Yamato_1944") from model directory.
    fn find_ship_index(&self, model_dir: &str) -> Option<String> {
        self.metadata
            .params()
            .iter()
            .find(|p| p.vehicle().and_then(|v| v.model_path()).map(|mp| mp.contains(model_dir)).unwrap_or(false))
            .map(|p| p.index().to_string())
    }

    /// Scan paths_storage for .visual files in a directory matching the name.
    fn find_visual_paths(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        model_dir: &str,
    ) -> Vec<(String, String)> {
        let needle = format!("/{model_dir}/");
        let mut result = Vec::new();

        for (i, entry) in db.paths_storage.iter().enumerate() {
            if !entry.name.ends_with(".visual") {
                continue;
            }
            let full_path = db.reconstruct_path(i, self_id_index);
            if full_path.contains(&needle) {
                let sub_name = entry.name.strip_suffix(".visual").unwrap_or(&entry.name).to_string();
                result.push((sub_name, full_path));
            }
        }

        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Load and parse all sub-models from (name, full_path) pairs.
    fn load_sub_models(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        visual_paths: &[(String, String)],
    ) -> Result<Vec<OwnedSubModel>, Report> {
        let mut result = Vec::new();

        for (sub_name, _) in visual_paths {
            let visual_suffix = format!("{sub_name}.visual");
            let vis_data = match resolve_visual_data(db, &visual_suffix, self_id_index) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Warning: skipping '{visual_suffix}': {e}");
                    continue;
                }
            };
            let vp = visual::parse_visual(vis_data).context("Failed to parse VisualPrototype")?;

            let geom_path_idx = self_id_index.get(&vp.merged_geometry_path_id).ok_or_else(|| {
                rootcause::report!(
                    "Could not resolve mergedGeometryPathId 0x{:016X} for {}",
                    vp.merged_geometry_path_id,
                    sub_name
                )
            })?;
            let geom_full_path = db.reconstruct_path(*geom_path_idx, self_id_index);

            let mut geom_bytes = Vec::new();
            self.vfs
                .join(&geom_full_path)
                .context("VFS path error")?
                .open_file()
                .context_with(|| format!("Could not open geometry: {geom_full_path}"))?
                .read_to_end(&mut geom_bytes)?;

            // Try loading the .splash file (same directory, same base name).
            let splash_bytes = if geom_full_path.ends_with(".geometry") {
                let splash_path = format!("{}.splash", &geom_full_path[..geom_full_path.len() - ".geometry".len()]);
                let mut buf = Vec::new();
                match self.vfs.join(&splash_path).and_then(|p| p.open_file()) {
                    Ok(mut f) => {
                        let _ = f.read_to_end(&mut buf);
                        Some(buf)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            result.push(OwnedSubModel { name: sub_name.clone(), visual: vp, geom_bytes, splash_bytes });
        }

        Ok(result)
    }

    /// Find all `.skel_ext` VFS paths whose parent directory is the ship's
    /// model_dir. Method form — delegates to the free
    /// [`find_skel_ext_paths`] for re-use from `export-model`.
    fn find_skel_ext_paths(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        model_dir: &str,
    ) -> Vec<(String, String)> {
        find_skel_ext_paths(db, self_id_index, model_dir)
    }

    /// Load `.skel_ext` bytes for each segment path. Method form —
    /// delegates to the free [`load_skel_ext_files`] for re-use.
    fn load_skel_ext_files(
        &self,
        skel_ext_paths: &[(String, String)],
    ) -> Result<Vec<OwnedSkelExt>, Report> {
        load_skel_ext_files(&self.vfs, skel_ext_paths)
    }

    /// Select mount points for the chosen hull upgrade, with optional module overrides.
    fn select_hull_mount_points(
        &self,
        vehicle: &crate::game_params::types::Vehicle,
        hull_selection: Option<&str>,
        module_overrides: &std::collections::HashMap<crate::game_params::keys::ComponentType, String>,
    ) -> Option<Vec<MountPoint>> {
        let upgrades = vehicle.hull_upgrades()?;
        let mut sorted: Vec<_> = upgrades.iter().collect();
        sorted.sort_by_key(|(k, _)| (*k).clone());

        let selected = if let Some(sel) = hull_selection {
            sorted
                .iter()
                .find(|(name, _)| *name == sel || name.to_lowercase().contains(&sel.to_lowercase()))
                .or_else(|| {
                    let prefix = format!("{sel}_");
                    sorted.iter().find(|(_, config)| {
                        config
                            .component_name(keys::ComponentType::Hull)
                            .map(|n| n.starts_with(&prefix))
                            .unwrap_or(false)
                    })
                })
                .copied()
        } else {
            sorted.first().copied()
        };

        selected.map(|(_, config)| {
            if module_overrides.is_empty() {
                config.all_mount_points().cloned().collect()
            } else {
                config.mount_points_with_overrides(module_overrides).cloned().collect()
            }
        })
    }

    /// Load turret models and build mount resolution data.
    fn load_mounts(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        mount_points: &[MountPoint],
        hull_parts: &[OwnedSubModel],
        model_dir: &str,
    ) -> Result<LoadedMounts, Report> {
        // Collect hardpoint transforms from hull visuals. Track which hull
        // sub-model each HP came from so downstream consumers (sidecar +
        // Unity) can group placements by section for damage/sinking
        // animations. The section is encoded in the sub-model's filename:
        // `<model_dir>_<Section>` for the 4 split files (Bow / MidFront /
        // MidBack / Stern), bare `<model_dir>` for the top-level visual.
        let model_prefix = format!("{model_dir}_");
        let mut hp_transforms: HashMap<String, [f32; 16]> = HashMap::new();
        let mut hp_section: HashMap<String, String> = HashMap::new();
        for smd in hull_parts {
            let section = smd.name.strip_prefix(&model_prefix).unwrap_or("Full").to_string();
            for &name_id in &smd.visual.nodes.name_map_name_ids {
                if let Some(name) = db.strings.get_string_by_id(name_id)
                    && name.starts_with("HP_")
                    && let Some(xform) = smd.visual.find_hardpoint_transform(name, &db.strings)
                {
                    hp_transforms.insert(name.to_string(), xform);
                    hp_section.insert(name.to_string(), section.clone());
                }
            }
        }

        // Load unique turret models.
        let (turret_models, turret_model_index) = self.load_turret_models_deduped(db, self_id_index, mount_points)?;

        // Map hull HP names to turret model paths so we can find the parent
        // turret visual for compound hardpoints.
        let hp_to_model_path: HashMap<&str, &str> = mount_points
            .iter()
            .filter(|mi| !mi.model_path().is_empty() && hp_transforms.contains_key(mi.hp_name()))
            .map(|mi| (mi.hp_name(), mi.model_path()))
            .collect();

        // Build resolved mounts.
        let mut mounts = Vec::new();
        for mi in mount_points {
            let Some(&model_idx) = turret_model_index.get(mi.model_path()) else {
                continue;
            };

            // Resolve transform: simple HP from hull directly, compound HP
            // by composing parent (hull) and child (turret visual) transforms.
            // A mount is compound iff its HP name is NOT in the hull node tree.
            // `parent_section` is the hull section that owns the (parent) HP —
            // for direct HPs that's the section the HP node lives in; for
            // compound HPs we inherit it from the parent hull HP.
            let (hull_transform, child_hp_transform, parent_section) =
                if let Some(&xform) = hp_transforms.get(mi.hp_name()) {
                    (xform, None, hp_section.get(mi.hp_name()).cloned())
                } else {
                    match resolve_compound_hp(
                        mi.hp_name(),
                        &hp_transforms,
                        &hp_to_model_path,
                        &turret_model_index,
                        &turret_models,
                        &db.strings,
                    ) {
                        Some((parent_hp, parent_xform, child)) => {
                            (parent_xform, child, hp_section.get(parent_hp).cloned())
                        }
                        None => {
                            eprintln!("Warning: could not resolve hardpoint '{}'", mi.hp_name());
                            continue;
                        }
                    }
                };
            let hp_transform = match child_hp_transform {
                None => hull_transform,
                Some(child_xform) => mat4_mul_col_major(&hull_transform, &child_xform),
            };

            // The turret model's Rotate_Y_BlendBone encodes its rest-pose
            // facing direction. WG bakes a Z-mirror (`diag(1,1,-1)`,
            // det = -1) into gun bones — a Maya/Blender export quirk
            // that we MUST undo so the gun's local +Z aligns with the
            // hardpoint's forward; otherwise the muzzle ends up facing
            // backwards. We post-multiply by `inverse(bone_local)` to
            // strip it out:
            //     visual_transform = hp_transform * inverse(bone_rotation).
            // Armor geometry is already aligned with the hardpoint, so
            // it uses the raw transform without any correction.
            //
            // Catapults (HP_JC_* / HP_AC_*, GameParams `typeinfo.type ==
            // "Catapult"`, no MountSpecies value) need a different rule:
            // their parent-ship `.visual` HPs carry a Ry(±90°) baked in,
            // and the catapult model's `Rotate_Y_BlendBone` has the
            // same `diag(1,1,-1)` as guns (det = -1). Following the
            // gun rule produces a Ry(±90°) net rotation — catapult
            // rails perpendicular to ship-forward — but the rendered
            // game places them ALONG ship-forward (legacy gmconvert
            // harvest captures identity). Empirically, catapults render
            // correctly when we keep ONLY the translation from the
            // composed HP transform and discard its rotation. Per-HP
            // rotation that's actually meaningful (catapult yaw for
            // launch) is animated at runtime; the rest pose is identity
            // ship-forward.
            let armor_transform = Some(hp_transform);
            let turret_visual = &turret_models[model_idx].visual;
            let is_catapult = mi.model_path().contains("/catapult/");
            let base_transform = if is_catapult {
                // Translation-only — strip rotation/scale, keep position.
                let mut t = IDENTITY_4X4;
                t[12] = hp_transform[12];
                t[13] = hp_transform[13];
                t[14] = hp_transform[14];
                t
            } else {
                let yaw_correction = turret_visual
                    .find_node_local_matrix("Rotate_Y_BlendBone", &db.strings)
                    .map(|bone_local| mat4_rotation_inverse(&bone_local));
                match yaw_correction {
                    Some(inv) => mat4_mul_col_major(&hp_transform, &inv),
                    None => hp_transform,
                }
            };

            let visual_transform = Some(base_transform);

            // Build per-vertex barrel pitch if pitchDeadZones applies.
            let min_pitch = mi.min_pitch_at_yaw(0.0);
            let barrel_pitch =
                if min_pitch > 0.0 { build_barrel_pitch(turret_visual, &db.strings, min_pitch) } else { None };

            mounts.push(ResolvedMount {
                hp_name: mi.hp_name().to_string(),
                parent_section,
                turret_model_index: model_idx,
                transform: visual_transform,
                armor_transform,
                mount_armor: mi.mount_armor().cloned(),
                species: mi.species(),
                barrel_pitch,
                model_path: mi.model_path().to_string(),
                ammo_list: mi.ammo_list().to_vec(),
            });
        }

        Ok(LoadedMounts { turret_models, turret_model_index, mounts })
    }

    /// Load unique turret models, deduplicating by model path.
    fn load_turret_models_deduped(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        mount_points: &[MountPoint],
    ) -> Result<(Vec<OwnedSubModel>, HashMap<String, usize>), Report> {
        let mut index_map: HashMap<String, usize> = HashMap::new();
        let mut models = Vec::new();

        for mi in mount_points {
            if mi.model_path().is_empty() || index_map.contains_key(mi.model_path()) {
                continue;
            }

            match self.load_single_turret(db, self_id_index, mi.model_path()) {
                Ok(smd) => {
                    let idx = models.len();
                    index_map.insert(mi.model_path().to_string(), idx);
                    models.push(smd);
                }
                Err(e) => {
                    eprintln!("Warning: could not load turret '{}': {e}", mi.model_path());
                }
            }
        }

        Ok((models, index_map))
    }

    /// Load turret models (non-deduplicating variant for texture listing).
    fn load_turret_models(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        mount_points: &[MountPoint],
    ) -> Result<Vec<OwnedSubModel>, Report> {
        let (models, _) = self.load_turret_models_deduped(db, self_id_index, mount_points)?;
        Ok(models)
    }

    /// Load a single turret model from its .model path.
    fn load_single_turret(
        &self,
        db: &PrototypeDatabase<'_>,
        self_id_index: &HashMap<u64, usize>,
        model_path: &str,
    ) -> Result<OwnedSubModel, Report> {
        let visual_path = model_path.replace(".model", ".visual");
        let visual_suffix = visual_path.rsplit('/').next().unwrap_or(&visual_path).to_string();

        let vis_data = resolve_visual_data(db, &visual_suffix, self_id_index)?;
        let vp = visual::parse_visual(vis_data).context("Failed to parse turret visual")?;

        let geom_path_idx = self_id_index
            .get(&vp.merged_geometry_path_id)
            .ok_or_else(|| rootcause::report!("Could not resolve geometry for turret '{}'", visual_suffix))?;
        let geom_full_path = db.reconstruct_path(*geom_path_idx, self_id_index);

        let mut geom_bytes = Vec::new();
        self.vfs
            .join(&geom_full_path)
            .context("VFS path error")?
            .open_file()
            .context_with(|| format!("Could not open turret geometry: {geom_full_path}"))?
            .read_to_end(&mut geom_bytes)?;

        let model_short_name =
            model_path.rsplit('/').next().unwrap_or(model_path).strip_suffix(".model").unwrap_or(model_path);

        Ok(OwnedSubModel { name: model_short_name.to_string(), visual: vp, geom_bytes, splash_bytes: None })
    }
}

// ---------------------------------------------------------------------------
// ShipModelContext — fully-loaded ship, ready for export
// ---------------------------------------------------------------------------

/// A fully-loaded ship model. Owns all bytes and parsed visuals.
///
/// Created via [`ShipAssets::load_ship()`]. Call [`export_glb()`](Self::export_glb)
/// to write the model to a file or buffer.
pub struct ShipModelContext {
    vfs: VfsPath,
    assets_bin_bytes: Vec<u8>,
    hull_parts: Vec<OwnedSubModel>,
    turret_models: Vec<OwnedSubModel>,
    mounts: Vec<ResolvedMount>,
    /// Per-segment `.skel_ext` files (decorative placements, loaded from VFS).
    /// Empty if the ship's model_dir has no `.skel_ext` files.
    skel_ext_files: Vec<OwnedSkelExt>,
    info: ShipInfo,
    options: ShipExportOptions,
    mat_camo_schemes: Vec<MatCamoScheme>,
    /// Armor thickness map from GameParams.  See [`ArmorMap`].
    armor_map: Option<ArmorMap>,
    /// Hit location zones from GameParams, keyed by zone name (e.g. "Citadel").
    hit_locations: Option<HashMap<String, crate::game_params::types::HitLocation>>,
}

/// Resolve a visual suffix to VisualPrototype record data.
///
/// If the suffix resolves to blob 1 (VisualPrototype), returns the data directly.
/// If it resolves to blob 3 (ModelPrototype), parses the ModelPrototype and follows
/// its `visual_resource_id` to look up the actual VisualPrototype.
fn resolve_visual_data<'a>(
    db: &'a PrototypeDatabase<'a>,
    visual_suffix: &str,
    self_id_index: &HashMap<u64, usize>,
) -> Result<&'a [u8], Report> {
    let (vis_location, _) = db
        .resolve_path(visual_suffix, self_id_index)
        .context_with(|| format!("Could not resolve visual: {visual_suffix}"))?;

    match vis_location.blob_index {
        1 => {
            // Direct VisualPrototype
            Ok(db
                .get_prototype_data(vis_location, visual::VISUAL_ITEM_SIZE)
                .context("Failed to get visual prototype data")?)
        }
        3 => {
            // ModelPrototype -- follow visualResourceId to the actual VisualPrototype
            let model_data = db
                .get_prototype_data(vis_location, model::MODEL_ITEM_SIZE)
                .context("Failed to get model prototype data")?;
            let mp = model::parse_model(model_data)
                .context_with(|| format!("Failed to parse ModelPrototype for {visual_suffix}"))?;

            if mp.visual_resource_id == 0 {
                bail!("ModelPrototype for '{}' has null visualResourceId", visual_suffix);
            }

            // Look up the visual resource by its selfId
            let r2p_value = db.lookup_r2p(mp.visual_resource_id).ok_or_else(|| {
                rootcause::report!(
                    "visualResourceId 0x{:016X} from ModelPrototype '{}' not found in r2p map",
                    mp.visual_resource_id,
                    visual_suffix
                )
            })?;
            let vis_loc = db.decode_r2p_value(r2p_value).context("Failed to decode r2p value for visual resource")?;

            if vis_loc.blob_index != 1 {
                bail!(
                    "ModelPrototype '{}' visualResourceId resolved to blob {} (expected 1)",
                    visual_suffix,
                    vis_loc.blob_index
                );
            }

            Ok(db
                .get_prototype_data(vis_loc, visual::VISUAL_ITEM_SIZE)
                .context("Failed to get visual prototype data via ModelPrototype")?)
        }
        other => {
            bail!("'{}' resolved to blob {} (expected 1=Visual or 3=Model)", visual_suffix, other);
        }
    }
}

impl ShipModelContext {
    /// Ship identity information.
    pub fn info(&self) -> &ShipInfo {
        &self.info
    }

    /// Hull part names (sub-model names).
    pub fn hull_part_names(&self) -> Vec<&str> {
        self.hull_parts.iter().map(|p| p.name.as_str()).collect()
    }

    /// Number of mounted components (turrets, AA, etc.).
    pub fn mount_count(&self) -> usize {
        self.mounts.len()
    }

    /// Number of unique turret/mount 3D models.
    pub fn unique_turret_count(&self) -> usize {
        self.turret_models.len()
    }

    /// Armor thickness map from GameParams.  See [`ArmorMap`].
    pub fn armor_map(&self) -> Option<&ArmorMap> {
        self.armor_map.as_ref()
    }

    /// Raw geometry bytes for hull parts, for inspection.
    pub fn hull_geom_bytes(&self) -> Vec<&[u8]> {
        self.hull_parts.iter().map(|p| p.geom_bytes.as_slice()).collect()
    }

    /// Raw geometry bytes for unique turret/mount models.
    pub fn turret_geom_bytes(&self) -> Vec<&[u8]> {
        self.turret_models.iter().map(|p| p.geom_bytes.as_slice()).collect()
    }

    /// Names of unique turret/mount models.
    pub fn turret_model_names(&self) -> Vec<&str> {
        self.turret_models.iter().map(|p| p.name.as_str()).collect()
    }

    /// Number of LOD levels available for hull meshes.
    pub fn hull_lod_count(&self) -> usize {
        self.hull_parts.iter().map(|p| p.visual.lods.len()).max().unwrap_or(1)
    }

    /// Hit location zones from GameParams (e.g. "Citadel" → HitLocation).
    pub fn hit_locations(&self) -> Option<&HashMap<String, crate::game_params::types::HitLocation>> {
        self.hit_locations.as_ref()
    }

    /// Raw splash file bytes for hull parts (if available).
    pub fn hull_splash_bytes(&self) -> Option<&[u8]> {
        self.hull_parts.iter().find_map(|p| p.splash_bytes.as_deref())
    }

    /// Build interactive armor meshes with per-triangle metadata.
    ///
    /// Returns one [`InteractiveArmorMesh`] per armor model found in the hull
    /// geometry. Each mesh contains the renderable triangle soup plus
    /// [`ArmorTriangleInfo`](gltf_export::ArmorTriangleInfo) entries aligned
    /// 1:1 with triangles, so a viewer can look up material name, thickness,
    /// and zone on hover/click.
    pub fn interactive_armor_meshes(&self) -> Result<Vec<InteractiveArmorMesh>, Report> {
        let mut result = Vec::new();

        // Hull armor (already in world space).
        for part in &self.hull_parts {
            let geom = geometry::parse_geometry(&part.geom_bytes)
                .context("Failed to parse hull geometry for interactive armor")?;
            for armor_model in &geom.armor_models {
                result.push(InteractiveArmorMesh::from_armor_model(armor_model, self.armor_map.as_ref(), None));
            }
        }

        // Turret armor: instance per mount.
        let turret_geoms: Vec<_> = self
            .turret_models
            .iter()
            .map(|part| {
                geometry::parse_geometry(&part.geom_bytes)
                    .context("Failed to parse turret geometry for interactive armor")
            })
            .collect::<Result<_, _>>()?;

        for mount in &self.mounts {
            let geom = &turret_geoms[mount.turret_model_index];
            for armor_model in &geom.armor_models {
                let mut mesh = InteractiveArmorMesh::from_armor_model(
                    armor_model,
                    self.armor_map.as_ref(),
                    mount.mount_armor.as_ref(),
                );
                mesh.transform = mount.armor_transform.map(gltf_export::negate_z_and_scale_to_metres);
                mesh.name = format!("{} [{}]", mesh.name, mount.hp_name);
                result.push(mesh);
            }
        }

        Ok(result)
    }
    /// Collect hull visual meshes for interactive display.
    ///
    /// Returns one [`InteractiveHullMesh`](gltf_export::InteractiveHullMesh) per
    /// render set (hull parts + mounted turrets). LOD 0 is used.
    /// Base albedo textures are baked into per-vertex colors when available.
    pub fn interactive_hull_meshes(&self) -> Result<Vec<gltf_export::InteractiveHullMesh>, Report> {
        use std::io::Cursor;

        let db = assets_bin::parse_assets_bin(&self.assets_bin_bytes)
            .context("Failed to parse assets.bin for hull meshes")?;

        let lod = self.options.lod;
        let damaged = self.options.damaged;
        let mut result = Vec::new();

        // Hull parts (no transform, already in world space).
        for part in &self.hull_parts {
            let geom =
                geometry::parse_geometry(&part.geom_bytes).context("Failed to parse hull geometry for hull meshes")?;
            let meshes = gltf_export::collect_hull_meshes(&part.visual, &geom, &db, lod, damaged, None)?;
            result.extend(meshes);
        }

        // Mounted turrets (with mount transforms).
        for mount in &self.mounts {
            let part = &self.turret_models[mount.turret_model_index];
            let geom = geometry::parse_geometry(&part.geom_bytes)
                .context("Failed to parse turret geometry for hull meshes")?;
            let mut meshes =
                gltf_export::collect_hull_meshes(&part.visual, &geom, &db, lod, damaged, mount.barrel_pitch.as_ref())?;
            for mesh in &mut meshes {
                mesh.transform = mount.transform.map(gltf_export::negate_z_and_scale_to_metres);
                mesh.name = format!("{} [{}]", mesh.name, mount.hp_name);
            }
            result.extend(meshes);
        }

        // Bake base albedo textures into per-vertex colors.
        // Cache decoded images by MFM path to avoid re-loading the same texture.
        let mut texture_cache: HashMap<String, Option<image_dds::image::RgbaImage>> = HashMap::new();

        for mesh in &mut result {
            let mfm_path = match &mesh.mfm_path {
                Some(p) => p.clone(),
                None => continue,
            };
            if mesh.uvs.len() != mesh.positions.len() {
                continue;
            }

            let image = texture_cache.entry(mfm_path.clone()).or_insert_with(|| {
                let dds_bytes = texture::load_base_albedo_bytes(&self.vfs, &mfm_path, None)?;
                let dds = image_dds::ddsfile::Dds::read(&mut Cursor::new(&dds_bytes)).ok()?;
                image_dds::image_from_dds(&dds, 0).ok()
            });

            if let Some(img) = image {
                let width = img.width();
                let height = img.height();
                if width == 0 || height == 0 {
                    continue;
                }

                let mut colors = Vec::with_capacity(mesh.uvs.len());
                for uv in &mesh.uvs {
                    // Wrap UVs into [0, 1) range and sample the image.
                    let u = uv[0].rem_euclid(1.0);
                    let v = uv[1].rem_euclid(1.0);
                    let x = ((u * width as f32) as u32).min(width - 1);
                    let y = ((v * height as f32) as u32).min(height - 1);
                    let pixel = img.get_pixel(x, y);
                    colors.push([
                        pixel[0] as f32 / 255.0,
                        pixel[1] as f32 / 255.0,
                        pixel[2] as f32 / 255.0,
                        1.0, // alpha will be set by the viewer
                    ]);
                }
                mesh.colors = colors;
            }
        }

        Ok(result)
    }

    /// Write a JSON manifest of every resolved mount placement.
    ///
    /// See the `--placements-json` CLI flag for the output contract. The shape
    /// is sidecar-ready: one typed section per [`MountSpecies`] group (main →
    /// `turrets`, secondary → `secondaries`, aaircraft → `antiair`, torpedo →
    /// `torpedoes`, everything else → `accessories`). Each entry carries a
    /// per-asset instance_id, the GameParams scope/category/subcategory parsed
    /// from the mount's model path, and a column-major world-space transform
    /// in metres (native × 15).
    ///
    /// Emits regardless of the configured [`AccessoryMode`] so a hull-only GLB
    /// export can still produce a complete placement manifest.
    #[cfg(feature = "json")]
    pub fn write_placements_json(&self, path: &Path) -> Result<(), Report> {
        use crate::game_params::types::MountSpecies;
        use serde_json::json;

        // --- Sections, keyed by MountSpecies group. ---
        let mut turrets = Vec::new();
        let mut secondaries = Vec::new();
        let mut antiair = Vec::new();
        let mut torpedoes = Vec::new();
        let mut accessories = Vec::new();

        // Per-asset-id running counter for instance_id uniqueness.
        let mut asset_counts: HashMap<String, usize> = HashMap::new();

        for mount in &self.mounts {
            let turret = &self.turret_models[mount.turret_model_index];
            let asset_id = turret.name.clone();
            let counter = asset_counts.entry(asset_id.clone()).or_insert(0);
            let ship_stem =
                self.info.display_name.as_deref().unwrap_or(self.info.model_dir.as_str()).to_string();
            let instance_id = format!("{}_{}_{:02}", ship_stem, asset_id, *counter);
            *counter += 1;

            // Derive scope / category / subcategory from the model VFS path.
            // Expected shape: content/gameplay/<scope>/<category>[/<subcategory>...]/<Asset>/<Asset>.model
            let (scope, category, subcategory) = parse_mount_path_taxonomy(&mount.model_path);

            // Build the world-space transform.
            //
            // Use `mount.transform` (the yaw-corrected variant — see the
            // bone-correction site at line ~1006), NOT `mount.armor_transform`
            // (the raw hardpoint). Downstream consumers instantiate the
            // accessory library GLB (visual mesh) at this placement, and the
            // visual mesh is stored in asset-local frame — so it needs the
            // `inverse(Rotate_Y_BlendBone)` correction applied to the
            // placement matrix. Armor meshes use `armor_transform` because
            // they're stored in hardpoint frame already (see comment at
            // line ~1003), but accessory placements are not armor.
            //
            // For ~half the WoWS fleet's gun assets the bone is a Z-mirror
            // (`diag(1,1,-1)`), so `mount.transform` carries a det=-1
            // (improper) rotation. `ensure_proper_rotation` converts that to
            // a proper `Ry(180°)` — visually identical for the bilaterally
            // symmetric turrets where this comes up, and produces a clean
            // PRS decomposition for consumers (no negative scale, no face-
            // winding flip needed). See `tools/reference/forward_axis_flip_audit.md`
            // for the full investigation.
            let raw = mount.transform.unwrap_or(IDENTITY_4X4);
            let proper = ensure_proper_rotation(raw);
            let matrix = super::gltf_export::negate_z_and_scale_to_metres(proper);
            let position = [matrix[12], matrix[13], matrix[14]];

            let species_str: Option<&'static str> = mount.species.map(|s| match s {
                MountSpecies::Main => "Main",
                MountSpecies::Secondary => "Secondary",
                MountSpecies::AAircraft => "AAircraft",
                MountSpecies::Torpedo => "Torpedo",
                MountSpecies::DCharge => "DCharge",
                MountSpecies::FireControl => "FireControl",
                MountSpecies::Search => "Search",
                MountSpecies::MissileGun => "MissileGun",
                MountSpecies::Decoration => "Decoration",
            });

            // `ammo_ids` is the per-mount link into the per-ship ballistics
            // file (see `wowsunpack ammo`). Emitted only when non-empty so
            // accessory entries stay schema-minimal. Order is preserved
            // from GameParams so consumers can map index → ammo type
            // (e.g. ammo_ids[0] == AP, ammo_ids[1] == HE for many BBs).
            let mut entry = json!({
                "instance_id":    instance_id,
                "asset_id":       asset_id,
                "hp_name":        mount.hp_name,
                "parent_section": mount.parent_section,
                "scope":          scope,
                "category":       category,
                "subcategory":    subcategory,
                "species":        species_str,
                "transform": {
                    "matrix":   matrix.as_slice(),
                    "position": position.as_slice(),
                },
            });
            if !mount.ammo_list.is_empty()
                && let Some(obj) = entry.as_object_mut()
            {
                obj.insert("ammo_ids".to_string(), json!(mount.ammo_list));
            }

            match mount.species {
                Some(MountSpecies::Main) => turrets.push(entry),
                Some(MountSpecies::Secondary) => secondaries.push(entry),
                Some(MountSpecies::AAircraft) => antiair.push(entry),
                Some(MountSpecies::Torpedo) => torpedoes.push(entry),
                // Everything else (FireControl, Search, MissileGun, Decoration,
                // DCharge, None) is a bulk accessory.
                _ => accessories.push(entry),
            }
        }

        // Skel-ext candidates are emitted to a separate companion file
        // (see `write_skel_ext_candidates_json`) rather than inlined here,
        // so the primary placements JSON stays compact for ships with
        // many decorative props (~100k affine matrices → 40+ MB inline).

        let now = format_rfc3339_utc(SystemTime::now());
        let toolkit_version = env!("CARGO_PKG_VERSION");

        let manifest = json!({
            "schema_version": 1,
            "ship": {
                "model_dir":    self.info.model_dir,
                "display_name": self.info.display_name,
                "param_index":  self.info.param_index,
                "nation":       self.info.nation,
                "species":      self.info.species,
                "tier":         self.info.tier,
            },
            "pipeline": {
                "toolkit_version": toolkit_version,
                "generated_at":    now,
            },
            "turrets":     turrets,
            "secondaries": secondaries,
            "antiair":     antiair,
            "torpedoes":   torpedoes,
            "accessories": accessories,
            "skel_ext_summary": {
                "files": self.skel_ext_files.len(),
                "segments": self.skel_ext_files.iter().map(|f| f.segment.as_str()).collect::<Vec<_>>(),
                "note": "Decorative skel_ext placements are emitted to a companion file via `--skel-ext-candidates-json` (see `write_skel_ext_candidates_json`).",
            },
        });

        let file = std::fs::File::create(path)
            .context_with(|| format!("Failed to create placements JSON at {}", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &manifest)
            .map_err(|e| rootcause::report!("Failed to serialize placements JSON: {e}"))?;
        writer.write_all(b"\n").ok();
        Ok(())
    }

    /// Write a companion JSON with every skel_ext affine matrix found in the
    /// ship's `.skel_ext` files.
    ///
    /// This output is **unresolved candidates**, not the final placements:
    /// each entry carries a 4×4 metric glTF transform plus the per-placement
    /// p0 and p1 hashes, but no asset_id (that's downstream). The Python
    /// resolver tool (`tools/skel_ext_resolve.py` in the pipeline repo) reads
    /// this file together with a legacy gmconvert scan (for ships we have
    /// one) or a cross-ship p0→asset_id fingerprint database (future) and
    /// fills in the `accessories[]` section of the main placements JSON.
    ///
    /// Emitted as **compact JSON** (no pretty-printing): for Montana, ~57k
    /// candidates → 6-8 MB, vs. 40+ MB pretty. Still JSON-parseable.
    ///
    /// Global dedupe at 1 cm: a real placement appears ~40-60× across
    /// segments for LOD/render-set variants. Keeping one per 1-cm position
    /// bucket preserves unique placements while cutting noise ~2×.
    ///
    /// Method form — delegates to the free
    /// [`write_skel_ext_candidates_json`] so `export-model` can call the
    /// same writer with a [`SkelExtSubject::Model`].
    #[cfg(feature = "json")]
    pub fn write_skel_ext_candidates_json(&self, path: &Path) -> Result<(), Report> {
        // Ship-side skel_ext placements are parented under the hull's
        // bones (typically `Scene Root` / `Root` at identity rest);
        // bone-rest composition is a no-op so we don't need it. Stay
        // on schema_version=1 emit to keep downstream consumers
        // (skel_ext_resolve.py for ship-side, hull decoratives) on
        // the same path. Asset-side placements (export-model) take
        // the schema_version=2 path with the asset's own visual.
        write_skel_ext_candidates_json(
            &self.skel_ext_files,
            &SkelExtSubject::Ship(&self.info),
            path,
            None,
        )
    }

    /// Write a JSON manifest of every hull material → texture stem mapping.
    ///
    /// For each hull sub-model, walks the `VisualPrototype` render sets,
    /// resolves the `.mfm` for each, parses it, and resolves every
    /// texture-typed property to a VFS path. The output lets downstream
    /// pipelines bind materials to DDS stems deterministically rather
    /// than via filename heuristics (e.g. avoiding cases like Myoko's
    /// `SHIPMAT_PBS_Bulge` → `_bools_a` mapping that needs a hand-curated
    /// alias today).
    ///
    /// One entry per `(sub_model, render_set)` tuple. Same material
    /// identifier (e.g. `SHIPMAT_PBS_Hull`) on multiple sub-models
    /// produces multiple entries, since each sub-model points to its
    /// own `.mfm` (e.g. Bow.mfm vs MidFront.mfm) and may bind different
    /// texture stems.
    ///
    /// Output shape (schema_version 1):
    /// ```json
    /// {
    ///   "schema_version": 1,
    ///   "ship": { "model_dir": "...", "param_index": "...", ... },
    ///   "pipeline": { "toolkit_version": "...", "generated_at": "..." },
    ///   "materials": [
    ///     {
    ///       "material_identifier": "SHIPMAT_PBS_Bulge",
    ///       "sub_model": "JSC008_Myoko_1945_Bow",
    ///       "mfm_stem": "JSC008_Myoko_1945_Bow",
    ///       "mfm_path": "content/.../JSC008_Myoko_1945_Bow.mfm",
    ///       "shader_id": "0x...",
    ///       "render_set": "Bow",
    ///       "skinned": false,
    ///       "textures": {
    ///         "diffuseMap": {
    ///           "stem": "JSC008_Myoko_1945_Bow",
    ///           "channel": "_a",
    ///           "vfs_path": "content/.../JSC008_Myoko_1945_Bow_a.dds",
    ///           "hash": "0x..."
    ///         },
    ///         "normalMap": { ... },
    ///         "metallicGlossMap": { ... },
    ///         "ambientOcclusionMap": { ... }
    ///       }
    ///     }
    ///   ]
    /// }
    /// ```
    ///
    /// Texture properties whose hash doesn't resolve (rare; would
    /// indicate a stale assets.bin index) are silently skipped.
    /// Properties named only by hash (i.e. not in the toolkit's
    /// 174-entry property name dictionary) surface with a hex slot
    /// name like `0xa1b2c3d4` so consumers don't lose data.
    #[cfg(feature = "json")]
    pub fn write_material_mappings_json(&self, path: &Path) -> Result<(), Report> {
        use serde_json::json;

        let db = assets_bin::parse_assets_bin(&self.assets_bin_bytes)
            .context("Failed to re-parse assets.bin for material mappings")?;
        let self_id_index = db.build_self_id_index();

        let mut material_entries: Vec<serde_json::Value> = Vec::new();
        for sub in &self.hull_parts {
            let entries = build_material_entries_for_visual(
                &sub.visual, &db, &self_id_index, &sub.name,
            );
            material_entries.extend(entries);
        }

        let now = format_rfc3339_utc(SystemTime::now());
        let toolkit_version = env!("CARGO_PKG_VERSION");

        let manifest = json!({
            "schema_version": 1,
            "ship": {
                "model_dir":    self.info.model_dir,
                "display_name": self.info.display_name,
                "param_index":  self.info.param_index,
                "nation":       self.info.nation,
                "species":      self.info.species,
                "tier":         self.info.tier,
            },
            "pipeline": {
                "toolkit_version": toolkit_version,
                "generated_at":    now,
            },
            "materials": material_entries,
        });

        let file = std::fs::File::create(path)
            .context_with(|| format!("Failed to create material mappings JSON at {}", path.display()))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(file), &manifest)
            .map_err(|e| rootcause::report!("Failed to serialize material mappings JSON: {e}"))?;
        Ok(())
    }

    /// Export the loaded ship model to GLB format.
    pub fn export_glb(&self, writer: &mut impl Write) -> Result<(), Report> {
        let db = assets_bin::parse_assets_bin(&self.assets_bin_bytes).context("Failed to re-parse assets.bin")?;

        // Parse geometries (scoped borrows — no self-referential issue).
        let hull_geoms: Vec<geometry::MergedGeometry<'_>> = self
            .hull_parts
            .iter()
            .map(|d| geometry::parse_geometry(&d.geom_bytes).expect("Failed to parse geometry"))
            .collect();

        let turret_geoms: Vec<geometry::MergedGeometry<'_>> = self
            .turret_models
            .iter()
            .map(|d| geometry::parse_geometry(&d.geom_bytes).expect("Failed to parse turret geometry"))
            .collect();

        // Build SubModel list.
        let mut sub_models: Vec<SubModel<'_>> = Vec::new();

        // Hull sub-models.
        for (data, geom) in self.hull_parts.iter().zip(hull_geoms.iter()) {
            sub_models.push(SubModel {
                name: data.name.clone(),
                visual: &data.visual,
                geometry: geom,
                transform: None,
                group: "Hull",
                barrel_pitch: None,
            });
        }

        // Mounted components. Filtered by `accessory_mode` so the caller can
        // keep main turrets (or nothing) in the ship GLB while secondaries /
        // AAs / etc. come from a shared-asset library downstream.
        for mount in &self.mounts {
            let group = mount_group(mount.species);
            if !self.options.accessory_mode.includes_group(group) {
                continue;
            }
            let turret_data = &self.turret_models[mount.turret_model_index];
            let turret_geom = &turret_geoms[mount.turret_model_index];

            sub_models.push(SubModel {
                name: format!("{} ({})", mount.hp_name, turret_data.name),
                visual: &turret_data.visual,
                geometry: turret_geom,
                transform: mount.transform,
                group,
                barrel_pitch: mount.barrel_pitch.clone(),
            });
        }

        // Load textures.
        //
        // Two trigger conditions:
        //   - `options.textures`       — textures are wanted in the GLB itself
        //     (either embedded in BIN chunk, or externalized as PNG URIs if
        //      `textures_dir` is set).
        //   - `options.raw_dds_dir`    — raw WG DDS files (every mip) are
        //     dumped as a side effect of `build_texture_set` for Unity
        //     streaming. Does NOT require `options.textures`.
        //
        // When only `raw_dds_dir` is set (common for Unity-authoritative
        // pipelines that want DDS-only), we still run the full texture
        // pipeline to populate the dumper, then discard the TextureSet so
        // no textures land in the GLB.
        let load_textures = self.options.textures || self.options.raw_dds_dir.is_some();
        let texture_set = if load_textures {
            let mut all_mfm_infos = Vec::new();
            for sub in &sub_models {
                all_mfm_infos.extend(collect_mfm_info(sub.visual, &db));
            }
            let mut tex_set = build_texture_set(
                &all_mfm_infos,
                &self.vfs,
                &db,
                self.options.raw_dds_dir.as_deref(),
            );
            let per_ship_count = tex_set.camo_schemes.len();

            // Merge material-based camo textures (mat_Steel, mat_Yamato_KoF, etc.).
            if !self.mat_camo_schemes.is_empty() {
                let stems: Vec<String> = {
                    let mut s = HashSet::new();
                    for info in &all_mfm_infos {
                        s.insert(info.stem.clone());
                    }
                    s.into_iter().collect()
                };

                for scheme in &self.mat_camo_schemes {
                    let mut png_bytes = None;

                    if scheme.tiled {
                        // Tiled camo: load tile DDS and bake with color scheme.
                        if let Some(colors) = &scheme.color_scheme_colors {
                            for path in &scheme.texture_paths {
                                if let Some(dds) = texture::load_dds_from_vfs(&self.vfs, path) {
                                    match texture::bake_tiled_camo_png(&dds, colors) {
                                        Ok(png) => {
                                            png_bytes = Some(png);
                                            break;
                                        }
                                        Err(e) => {
                                            eprintln!("  Warning: failed to bake tiled camo {}: {e}", path);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Non-tiled mat_camo: load DDS and convert to PNG.
                        for path in &scheme.texture_paths {
                            if let Some(dds) = texture::load_dds_from_vfs(&self.vfs, path) {
                                match texture::dds_to_png(&dds) {
                                    Ok(png) => {
                                        png_bytes = Some(png);
                                        break;
                                    }
                                    Err(e) => {
                                        eprintln!("  Warning: failed to decode mat_camo texture {}: {e}", path);
                                    }
                                }
                            }
                        }
                    }

                    if let Some(png) = png_bytes {
                        let mut scheme_textures = HashMap::new();
                        for stem in &stems {
                            scheme_textures.insert(stem.clone(), png.clone());
                        }
                        let scheme_idx = tex_set.camo_schemes.len();
                        tex_set.camo_schemes.push((scheme.display_name.clone(), scheme_textures));
                        if scheme.tiled {
                            // Store per-stem UV transforms for this tiled scheme.
                            for stem in &stems {
                                let cat = camouflage::classify_part_category(stem);
                                if let Some(xform) = scheme.uv_transforms.get(cat) {
                                    tex_set.tiled_uv_transforms.insert(
                                        (scheme_idx, stem.clone()),
                                        [xform.scale[0], xform.scale[1], xform.offset[0], xform.offset[1]],
                                    );
                                }
                            }
                        }
                    }
                }

                let mat_count = tex_set.camo_schemes.len() - per_ship_count;
                eprintln!("  Texture variants: {} per-ship, {} material-based", per_ship_count, mat_count);
            }

            tex_set
        } else {
            TextureSet::empty()
        };

        // DDS dump is a side effect of `build_texture_set`. If the caller
        // only asked for DDS (not glTF-embedded textures), discard the
        // loaded TextureSet so nothing lands in the GLB.
        let texture_set = if self.options.textures {
            texture_set
        } else {
            TextureSet::empty()
        };

        // Collect armor meshes from hull AND turret geometries with thickness data.
        let armor_map = self.armor_map.as_ref();
        let mut armor_meshes: Vec<gltf_export::ArmorSubModel> = Vec::new();
        // Hull armor (already in world space, no transform needed).
        for geom in &hull_geoms {
            for am in &geom.armor_models {
                armor_meshes.extend(gltf_export::armor_sub_models_by_zone(am, armor_map, None));
            }
        }

        // Turret armor: instance per mount with that mount's transform.
        // Follows the same accessory-mode filter as the mount meshes so the
        // two stay consistent.
        for mount in &self.mounts {
            let group = mount_group(mount.species);
            if !self.options.accessory_mode.includes_group(group) {
                continue;
            }
            let turret_geom = &turret_geoms[mount.turret_model_index];
            for am in &turret_geom.armor_models {
                let mut subs = gltf_export::armor_sub_models_by_zone(am, armor_map, mount.mount_armor.as_ref());
                for s in &mut subs {
                    s.transform = mount.armor_transform;
                    s.name = format!("{} [{}]", s.name, mount.hp_name);
                }
                armor_meshes.extend(subs);
            }
        }

        // Hitboxes: parse the hull `.splash` file into named cube AABBs.
        // Default on — data is ~1 KB per ship and gives downstream consumers
        // per-magazine citadel volumes, per-turret barbettes, etc. that the
        // armor-zone-only classification can't express. Silent skip if the
        // ship has no `.splash` file (some older / smaller hulls).
        let hitboxes: Vec<gltf_export::Hitbox> = self
            .hull_splash_bytes()
            .and_then(|bytes| geometry::parse_splash_file(bytes).ok())
            .map(|boxes| boxes.iter().map(gltf_export::hitbox_from_splash).collect())
            .unwrap_or_default();

        let mut tex_out = match &self.options.textures_dir {
            Some(dir) => gltf_export::TextureOutput::external(dir, &self.options.textures_uri_prefix),
            None => gltf_export::TextureOutput::Embedded,
        };
        gltf_export::export_ship_glb(
            &sub_models,
            &armor_meshes,
            &hitboxes,
            &db,
            self.options.lod,
            &texture_set,
            self.options.damaged,
            self.options.all_render_sets,
            &mut tex_out,
            writer,
        )
        .context("Failed to export ship GLB")?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Owns visual + geometry bytes for one sub-model (no lifetime parameters).
struct OwnedSubModel {
    name: String,
    visual: VisualPrototype,
    geom_bytes: Vec<u8>,
    /// Raw `.splash` file bytes (only present for base hull models).
    splash_bytes: Option<Vec<u8>>,
}

/// One `.skel_ext` file owned by some content unit.
///
/// For a ship: 8 per ship (4 base + 4 `_ep`), one per hull segment.
/// For a single accessory model (turret, director, finder, radar,
/// catapult, ...): typically 1, sibling to the asset's `.geometry` /
/// `.visual` files. Contains decorative-placement data that's referenced
/// at runtime via `ModelPrototype.skel_ext_res_ids`. See
/// [`crate::models::skel_ext`] for the format.
pub struct OwnedSkelExt {
    /// Segment stem. For ships this is the section (e.g. "Bow", "Bow_ep",
    /// "MidFront", "MidFront_ep", ...). For accessories there is no
    /// section concept; we use the asset's full stem (e.g.
    /// "AGM034_16in50_Mk7") so downstream resolvers can still group
    /// records by source file.
    pub segment: String,
    /// Full VFS path the bytes came from (for diagnostics).
    pub vfs_path: String,
    /// Raw bytes of the file.
    pub bytes: Vec<u8>,
}

/// What is being exported alongside a `--skel-ext-candidates-json` file.
///
/// Drives the `"ship"` / `"model"` metadata block in the manifest header
/// and lets the same writer serve both `export-ship` and `export-model`.
pub enum SkelExtSubject<'a> {
    /// A full ship, with all GameParams identity available.
    Ship(&'a ShipInfo),
    /// A single accessory model (turret, director, finder, radar,
    /// catapult, ...). The `stem` is the model directory name
    /// (e.g. "AGM034_16in50_Mk7"), which is also the asset_id used by
    /// the pipeline's accessory library.
    Model {
        /// Asset stem (e.g. "AGM034_16in50_Mk7").
        stem: &'a str,
        /// Optional human-readable label (typically the same as `stem`).
        display_name: Option<&'a str>,
    },
}

/// Result of [`ShipAssets::load_mounts`].
struct LoadedMounts {
    turret_models: Vec<OwnedSubModel>,
    #[allow(dead_code)]
    turret_model_index: HashMap<String, usize>,
    mounts: Vec<ResolvedMount>,
}

/// A mount instance with resolved transform.
struct ResolvedMount {
    hp_name: String,
    /// Hull section that owns this HP (`Bow` / `MidFront` / `MidBack` /
    /// `Stern` / `Full` for the top-level visual; `None` if no hull
    /// sub-model declares this HP — shouldn't happen for resolved
    /// mounts, but kept optional defensively). For compound HPs (e.g.
    /// `HP_AGM_3_HP_AGA_4`, an AA gun mounted on a main turret), this
    /// is the section of the **outer parent** HP (`HP_AGM_3` →
    /// `MidFront`/`Stern`/etc), since that's the hull part the whole
    /// stack rides on for damage/sinking purposes.
    parent_section: Option<String>,
    turret_model_index: usize,
    /// Visual transform with yaw correction.
    transform: Option<[f32; 16]>,
    /// Raw hardpoint transform without model rotation (for armor geometry).
    armor_transform: Option<[f32; 16]>,
    /// Per-mount armor map for turret shell surfaces (from `A_Artillery.HP_XXX.armor`).
    mount_armor: Option<crate::game_params::types::ArmorMap>,
    /// Mount species from GameParams `typeinfo.species`.
    species: Option<crate::game_params::types::MountSpecies>,
    /// Per-vertex barrel pitch configuration (if pitchDeadZones applies).
    barrel_pitch: Option<super::gltf_export::BarrelPitch>,
    /// Full GameParams model path, e.g.
    /// `content/gameplay/usa/gun/main/AGM034_16in50_Mk7/AGM034_16in50_Mk7.model`.
    /// Preserved from the `MountPoint` so downstream placement-manifest emission
    /// can derive scope/category/subcategory/asset_id without a second lookup.
    model_path: String,
    /// `Projectile` GameParam names this mount can fire, in declared order.
    /// Empty for non-firing mounts (directors, finders, radars, etc.).
    /// Surfaces in the placements JSON as `ammo_ids: [...]` per turret /
    /// secondary / antiair / torpedo entry; downstream consumers resolve each
    /// against the per-ship ballistics file produced by `wowsunpack ammo`.
    ammo_list: Vec<String>,
}

/// Pre-resolved material-based camouflage scheme (owned data, no lifetimes).
struct MatCamoScheme {
    /// Display name for the variant (translated or fallback).
    display_name: String,
    /// Albedo texture VFS paths from camouflages.xml.
    texture_paths: Vec<String>,
    /// Whether this is a tiled camo (uses UV tiling via KHR_texture_transform).
    tiled: bool,
    /// Resolved color scheme colors for tiled camos (4 RGBA colors, linear space).
    color_scheme_colors: Option<[[f32; 4]; 4]>,
    /// Per-part UV transforms for tiled camos. Key = part category (lowercase).
    uv_transforms: HashMap<String, camouflage::UvTransform>,
}

// ---------------------------------------------------------------------------
// Texture-path stem/channel extraction (used by `write_material_mappings_json`)
// ---------------------------------------------------------------------------

/// Mip-level suffixes WG appends to texture filenames. `.dd0` is the
/// highest-resolution mip; `.dds` carries the bundled mip tail.
const TEXTURE_MIP_SUFFIXES: &[&str] = &[".dd0", ".dd1", ".dd2", ".dds"];

/// Channel suffixes WG appends to texture filenames immediately after the
/// material stem. Order matters: longer / more-specific suffixes must come
/// first so e.g. `_normal` is matched before `_n`, `_emissive` before `_e`,
/// `_nbmask` (4 chars) before its components. Each entry must be a leading
/// `_` plus letters so we can detect it at the end of a filename stem.
const TEXTURE_CHANNEL_SUFFIXES: &[&str] = &[
    "_normal",   // conformant glTF normal map (toolkit-emitted Phase B sibling)
    "_nbmask",   // normal-B-channel mask (toolkit-emitted Phase B sibling)
    "_emissive", // synthesized emissive (Python pipeline-emitted)
    "_mr",       // conformant glTF metallic-roughness (toolkit-emitted Phase B sibling)
    "_mg",       // raw WG metallic-gloss
    "_ao",       // ambient occlusion
    "_od",       // overlay diffuse (TILEDLAND)
    "_n",        // raw WG normal (B = camo gate, not Z)
    "_a",        // albedo
    "_e",        // emissive (legacy short form)
];

/// Decompose a VFS texture path into `(stem, channel)`.
///
/// Strips the mip suffix (`.dd0` / `.dds` / etc.) and the trailing
/// channel suffix (`_a` / `_n` / `_normal` / ...) from the filename leaf.
/// Returns the bare stem and the detected channel separately so consumers
/// can match against either form.
///
/// Examples:
/// - `content/.../JSC008_Myoko_1945_Bow_a.dds` → (`JSC008_Myoko_1945_Bow`, `_a`)
/// - `content/.../JSC008_Myoko_1945_Bow_normal.dd0` → (`JSC008_Myoko_1945_Bow`, `_normal`)
/// - `content/.../no_channel.dds` → (`no_channel`, "") if no channel suffix matches
pub fn derive_texture_stem_and_channel(vfs_path: &str) -> (String, String) {
    let leaf = vfs_path.rsplit_once('/').map(|(_, n)| n).unwrap_or(vfs_path);
    let mut filename_stem = leaf.to_string();
    for sfx in TEXTURE_MIP_SUFFIXES {
        if let Some(s) = filename_stem.strip_suffix(sfx) {
            filename_stem = s.to_string();
            break;
        }
    }
    for sfx in TEXTURE_CHANNEL_SUFFIXES {
        if let Some(s) = filename_stem.strip_suffix(sfx) {
            return (s.to_string(), (*sfx).to_string());
        }
    }
    (filename_stem, String::new())
}

// ---------------------------------------------------------------------------
// Shared helpers (pub so main.rs export-model can use them too)
// ---------------------------------------------------------------------------

/// Build the per-render-set material entries that ship the material
/// mappings JSON, for one visual + one ``sub_model`` label. Used by both
/// the per-ship writer (looping over `hull_parts`) and the per-model
/// writer in `export-model` (single visual).
///
/// Walks each render set whose `material_mfm_path_id != 0`, parses the
/// referenced MFM, and emits a `{material_identifier, sub_model,
/// mfm_stem, mfm_path, shader_id, render_set, skinned, textures}`
/// entry. The `textures` map keys are MFM property names (e.g.
/// `diffuseMap`, `normalMap`, `metallicGlossMap`); values carry the
/// resolved `stem`, `channel`, `vfs_path`, and `hash` so consumers can
/// bind textures per-material instead of falling back to a flat
/// directory walk.
#[cfg(feature = "json")]
pub fn build_material_entries_for_visual(
    vp: &VisualPrototype,
    db: &PrototypeDatabase<'_>,
    self_id_index: &HashMap<u64, usize>,
    sub_model: &str,
) -> Vec<serde_json::Value> {
    use crate::models::material::PropertyType;
    use crate::models::material::PropertyValue;
    use serde_json::json;

    let mut entries: Vec<serde_json::Value> = Vec::new();

    for rs in &vp.render_sets {
        if rs.material_mfm_path_id == 0 {
            continue;
        }

        let material_identifier = db
            .strings
            .get_string_by_id(rs.material_name_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("material_0x{:08X}", rs.material_name_id));

        let render_set_name = db
            .strings
            .get_string_by_id(rs.name_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("rs_0x{:08X}", rs.name_id));

        let Some(&mfm_path_idx) = self_id_index.get(&rs.material_mfm_path_id) else {
            continue;
        };
        let mfm_full_path = db.reconstruct_path(mfm_path_idx, self_id_index);
        let mfm_leaf = &db.paths_storage[mfm_path_idx].name;
        let mfm_stem = mfm_leaf.strip_suffix(".mfm").unwrap_or(mfm_leaf).to_string();

        let mat = match texture::parse_mfm_from_db(db, rs.material_mfm_path_id) {
            Some(m) => m,
            None => {
                entries.push(json!({
                    "material_identifier": material_identifier,
                    "sub_model":           sub_model,
                    "mfm_stem":            mfm_stem,
                    "mfm_path":            mfm_full_path,
                    "shader_id":           serde_json::Value::Null,
                    "render_set":          render_set_name,
                    "skinned":             rs.skinned,
                    "textures":            serde_json::Map::new(),
                    "parse_error":         true,
                }));
                continue;
            }
        };

        let mut textures_obj = serde_json::Map::new();
        let mut floats_obj = serde_json::Map::new();
        for prop in &mat.properties {
            match prop.property_type {
                PropertyType::Texture => {
                    let hash = match &prop.value {
                        Some(PropertyValue::Texture(h)) if *h != 0 => *h,
                        _ => continue,
                    };
                    let Some(&tex_idx) = self_id_index.get(&hash) else {
                        continue;
                    };
                    let vfs_path = db.reconstruct_path(tex_idx, self_id_index);
                    let (stem, channel) = derive_texture_stem_and_channel(&vfs_path);

                    let slot_name = match prop.name {
                        Some(n) => n.to_string(),
                        None => format!("0x{:08x}", prop.name_hash),
                    };

                    textures_obj.insert(
                        slot_name,
                        json!({
                            "stem":     stem,
                            "channel":  channel,
                            "vfs_path": vfs_path,
                            "hash":     format!("0x{:016x}", hash),
                        }),
                    );
                }
                // Capture float scalars too — the engine reads
                // `g_detailNormalInfluence`, `g_detailScaleU`, etc. as
                // per-material blend params for the detail-map normal
                // layer (PBS_ship_metallic chunk012:43-74). Downstream
                // pipeline consumers (sidecar emit) pull these out
                // alongside the texture references.
                PropertyType::FloatA | PropertyType::FloatB => {
                    if let Some(PropertyValue::Float(v)) = prop.value {
                        let name = match prop.name {
                            Some(n) => n.to_string(),
                            None => format!("0x{:08x}", prop.name_hash),
                        };
                        floats_obj.insert(name, json!(v));
                    }
                }
                _ => {}
            }
        }

        entries.push(json!({
            "material_identifier": material_identifier,
            "sub_model":           sub_model,
            "mfm_stem":            mfm_stem,
            "mfm_path":            mfm_full_path,
            "shader_id":           format!("0x{:08x}", mat.shader_id),
            "render_set":          render_set_name,
            "skinned":             rs.skinned,
            "textures":            textures_obj,
            "floats":              floats_obj,
        }));
    }

    entries
}

/// Write a single-model material mappings JSON for `export-model`.
/// Same `materials[]` shape as `write_material_mappings_json` (per-ship)
/// but with a `model { geometry_path, visual_path }` header instead of
/// the `ship { … }` block. `sub_model` is typically the model dir name
/// (e.g. ``JD570_Director_Type_94_1_Arpeggio``); accessory-library
/// consumers key off it for asset_id correlation.
#[cfg(feature = "json")]
pub fn write_model_material_mappings_json(
    vp: &VisualPrototype,
    db: &PrototypeDatabase<'_>,
    geometry_vfs_path: &str,
    visual_vfs_path: &str,
    sub_model: &str,
    out_path: &Path,
) -> Result<(), Report> {
    use serde_json::json;

    let self_id_index = db.build_self_id_index();
    let entries = build_material_entries_for_visual(vp, db, &self_id_index, sub_model);

    let now = format_rfc3339_utc(SystemTime::now());
    let toolkit_version = env!("CARGO_PKG_VERSION");

    let manifest = json!({
        "schema_version": 1,
        "model": {
            "geometry_path": geometry_vfs_path,
            "visual_path":   visual_vfs_path,
            "sub_model":     sub_model,
        },
        "pipeline": {
            "toolkit_version": toolkit_version,
            "generated_at":    now,
        },
        "materials": entries,
    });

    let file = std::fs::File::create(out_path).context_with(|| {
        format!(
            "Failed to create model material mappings JSON at {}",
            out_path.display()
        )
    })?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &manifest)
        .map_err(|e| rootcause::report!("Failed to serialize model material mappings JSON: {e}"))?;
    Ok(())
}

/// Resolved MFM info: stem (leaf name without `.mfm`) and full VFS path.
///
/// `material_mfm_path_id` is the MFM's `selfId` hash (matches
/// `render_set.material_mfm_path_id`) — used to resolve PBR auxiliary
/// texture references via `texture::load_pbr_channels`.
pub struct MfmInfo {
    pub stem: String,
    pub full_path: String,
    pub material_mfm_path_id: u64,
}

/// Collect MFM stems and full paths from a visual's render sets.
pub fn collect_mfm_info(visual: &VisualPrototype, db: &PrototypeDatabase<'_>) -> Vec<MfmInfo> {
    let self_id_index = db.build_self_id_index();
    let mut result = Vec::new();
    let mut seen = HashSet::new();

    for rs in &visual.render_sets {
        if rs.material_mfm_path_id == 0 {
            continue;
        }
        let Some(&path_idx) = self_id_index.get(&rs.material_mfm_path_id) else {
            continue;
        };
        let mfm_name = &db.paths_storage[path_idx].name;
        let stem = mfm_name.strip_suffix(".mfm").unwrap_or(mfm_name);

        if seen.insert(stem.to_string()) {
            let full_path = db.reconstruct_path(path_idx, &self_id_index);
            result.push(MfmInfo {
                stem: stem.to_string(),
                full_path,
                material_mfm_path_id: rs.material_mfm_path_id,
            });
        }
    }

    result
}

/// Build a `TextureSet` from MFM infos: base albedo + PBR auxiliary
/// channels (normal / metallic-gloss / AO) + all camo schemes.
///
/// `db` is required to resolve MFM → `normalMap` / `metallicGlossMap` /
/// `ambientOcclusionMap` texture hashes. The PBR channels populate only
/// the base (default-appearance) maps in Phase A — camo-variant PBR is
/// left for Phase B once per-camo normal/MG paths are confirmed.
///
/// See `tools/toolkit_integration/pbr_textures.md` in the downstream
/// pipeline repo for the channel-packing plan.
pub fn build_texture_set(
    mfm_infos: &[MfmInfo],
    vfs: &VfsPath,
    db: &PrototypeDatabase<'_>,
    raw_dds_dir: Option<&std::path::Path>,
) -> TextureSet {
    let mut base = HashMap::new();
    let mut normal_base = HashMap::new();
    let mut metallic_roughness_base = HashMap::new();
    let mut occlusion_base = HashMap::new();

    let mut seen_stems = HashSet::new();
    let mut unique_infos: Vec<&MfmInfo> = Vec::new();
    for info in mfm_infos {
        if seen_stems.insert(info.stem.clone()) {
            unique_infos.push(info);
        }
    }

    // Build the path-id index once; PBR channel resolution reuses it
    // across every MFM in the ship.
    let self_id_index = db.build_self_id_index();

    // One dumper per ship; session-scoped dedup means the same texture
    // referenced by multiple materials writes to disk only once.
    let mut raw_dds_dumper = raw_dds_dir.map(|d| texture::RawDdsDumper::new(d.to_path_buf()));

    // Load base albedo textures.
    for info in &unique_infos {
        if let Some(dds_bytes) = texture::load_base_albedo_bytes(vfs, &info.full_path, raw_dds_dumper.as_mut()) {
            match texture::dds_to_png(&dds_bytes) {
                Ok(png_bytes) => {
                    base.insert(info.stem.clone(), png_bytes);
                }
                Err(e) => {
                    eprintln!("  Warning: failed to decode base texture for {}: {e}", info.stem);
                }
            }
        }

        // Load PBR auxiliary channels (normal / MG / AO) via MFM property lookup.
        if info.material_mfm_path_id != 0 {
            let channels = texture::load_pbr_channels(
                vfs, db, &self_id_index, info.material_mfm_path_id,
                raw_dds_dumper.as_mut(),
            );
            if let Some(png) = channels.normal {
                normal_base.insert(info.stem.clone(), png);
            }
            if let Some(png) = channels.metallic_roughness {
                metallic_roughness_base.insert(info.stem.clone(), png);
            }
            if let Some(png) = channels.occlusion {
                occlusion_base.insert(info.stem.clone(), png);
            }
        }
    }

    // Discover camo schemes.
    let stems: Vec<String> = unique_infos.iter().map(|i| i.stem.clone()).collect();
    let schemes = texture::discover_texture_schemes(vfs, &stems);

    let mut camo_schemes = Vec::new();
    for scheme in &schemes {
        let mut scheme_textures = HashMap::new();
        for info in &unique_infos {
            if let Some((_base_name, dds_bytes)) =
                texture::load_texture_bytes(vfs, &info.stem, scheme, raw_dds_dumper.as_mut())
            {
                match texture::dds_to_png(&dds_bytes) {
                    Ok(png_bytes) => {
                        scheme_textures.insert(info.stem.clone(), png_bytes);
                    }
                    Err(e) => {
                        eprintln!("  Warning: failed to decode camo texture {}_{}: {e}", info.stem, scheme);
                    }
                }
            }
        }
        if !scheme_textures.is_empty() {
            camo_schemes.push((scheme.clone(), scheme_textures));
        }
    }

    TextureSet {
        base,
        normal_base,
        metallic_roughness_base,
        occlusion_base,
        camo_schemes,
        tiled_uv_transforms: HashMap::new(),
    }
}

/// Resolve a compound hardpoint (e.g. `HP_AGM_3_HP_AGA_1`) by finding the
/// longest hull HP name that prefixes the mount's HP name, then looking up
/// the child HP in the parent turret's visual node tree.
///
/// Returns `(parent_hull_transform, Some(child_turret_transform))` on success.
/// Returns `(parent_hp_name, hull_transform, child_hp_transform)` so the
/// caller can both compose the placement transform and look up the parent
/// hull HP's section for the placements JSON.
fn resolve_compound_hp<'a>(
    hp_name: &str,
    hp_transforms: &'a HashMap<String, [f32; 16]>,
    hp_to_model_path: &HashMap<&str, &str>,
    turret_model_index: &HashMap<String, usize>,
    turret_models: &[OwnedSubModel],
    strings: &assets_bin::StringsSection<'_>,
) -> Option<(&'a str, [f32; 16], Option<[f32; 16]>)> {
    // Find the longest hull HP name that is a proper prefix of hp_name with
    // a '_' separator. This avoids partial matches like HP_AG matching HP_AGM_3.
    let mut best_parent: Option<(&str, &[f32; 16])> = None;
    for (hull_hp, xform) in hp_transforms {
        if hp_name.len() > hull_hp.len()
            && hp_name.starts_with(hull_hp.as_str())
            && hp_name.as_bytes()[hull_hp.len()] == b'_'
            && best_parent.is_none_or(|(bp, _)| hull_hp.len() > bp.len())
        {
            best_parent = Some((hull_hp.as_str(), xform));
        }
    }
    let (parent_hp, parent_xform) = best_parent?;

    // Extract child HP name: everything after "parent_"
    let child_hp = &hp_name[parent_hp.len() + 1..];

    // Find the parent turret model via the hp_to_model_path mapping.
    let &parent_model_path = hp_to_model_path.get(parent_hp)?;
    let &parent_turret_idx = turret_model_index.get(parent_model_path)?;
    let parent_turret = &turret_models[parent_turret_idx];

    // Look up the child HP transform in the parent turret's visual.
    let child_xform = parent_turret.visual.find_hardpoint_transform(child_hp, strings)?;

    Some((parent_hp, *parent_xform, Some(child_xform)))
}

/// Build a [`BarrelPitch`] config for per-vertex barrel rotation.
///
/// Finds the `Rotate_X` pivot point in turret-local space, builds a pitch
/// rotation matrix around it, and identifies which blend bone indices are
/// barrel bones (descendants of `Rotate_X` in the skeleton hierarchy).
fn build_barrel_pitch(
    visual: &VisualPrototype,
    strings: &assets_bin::StringsSection<'_>,
    min_pitch_deg: f32,
) -> Option<super::gltf_export::BarrelPitch> {
    let rotate_x_idx = visual.find_node_index_by_name("Rotate_X", strings)?;

    // Get the Rotate_X world (composed) transform to find pivot position.
    let rotate_x_world = visual.find_hardpoint_transform("Rotate_X", strings)?;
    let pivot = [rotate_x_world[12], rotate_x_world[13], rotate_x_world[14]];

    // Build pitch rotation matrix: T(-pivot) * Rx(-pitch) * T(pivot)
    let pitch_rad = -min_pitch_deg.to_radians();
    let (sin_p, cos_p) = pitch_rad.sin_cos();

    // Rx rotation (column-major):
    //   1     0      0
    //   0   cos_p  -sin_p
    //   0   sin_p   cos_p
    let pitch_matrix = [
        // col 0
        1.0,
        0.0,
        0.0,
        0.0,
        // col 1
        0.0,
        cos_p,
        sin_p,
        0.0,
        // col 2
        0.0,
        -sin_p,
        cos_p,
        0.0,
        // col 3: T(-pivot) * Rx * T(pivot) translation
        0.0,
        pivot[1] - cos_p * pivot[1] + sin_p * pivot[2],
        pivot[2] + sin_p * pivot[1] - cos_p * pivot[2],
        1.0,
    ];

    // Identify barrel bone indices from the first render set's blend bone list.
    // Barrel bones = those that ARE `Rotate_X` or descendants of it.
    let first_rs = visual.render_sets.first()?;
    let mut barrel_bone_indices = Vec::new();
    for (blend_idx, &name_id) in first_rs.node_name_ids.iter().enumerate() {
        // Resolve name_id → node index in skeleton
        let node_idx = visual
            .nodes
            .name_map_name_ids
            .iter()
            .position(|&nid| nid == name_id)
            .map(|i| visual.nodes.name_map_node_ids[i]);
        if let Some(ni) = node_idx
            && (ni == rotate_x_idx || visual.is_descendant_of(ni, rotate_x_idx))
        {
            barrel_bone_indices.push(blend_idx as u8);
        }
    }

    if barrel_bone_indices.is_empty() {
        return None;
    }

    // Conjugate the pitch matrix for Z-negated coordinate space (left→right-handed).
    let pitch_matrix = super::gltf_export::negate_z_transform(pitch_matrix);

    Some(super::gltf_export::BarrelPitch { pitch_matrix, barrel_bone_indices })
}

fn mat4_mul_col_major(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

/// Extract the rotation part of a column-major 4x4 matrix and return its
/// inverse (transpose) as a full 4x4 matrix with zero translation.
///
/// Valid for rigid-body transforms (orthonormal rotation + translation).
/// The inverse of the rotation part is simply its transpose.
fn mat4_rotation_inverse(m: &[f32; 16]) -> [f32; 16] {
    // Column-major layout:
    //   col0 = m[0..4], col1 = m[4..8], col2 = m[8..12], col3 = m[12..16]
    // 3x3 rotation at (row, col) -> m[col*4 + row]
    // Transpose: swap (row, col) -> (col, row)
    [
        m[0], m[4], m[8], 0.0, // col 0 = original row 0
        m[1], m[5], m[9], 0.0, // col 1 = original row 1
        m[2], m[6], m[10], 0.0, // col 2 = original row 2
        0.0, 0.0, 0.0, 1.0, // no translation
    ]
}

/// Determinant of the upper-left 3x3 of a column-major 4x4. Used to detect
/// improper rotations (reflections) that arise when a turret's
/// `Rotate_Y_BlendBone` rest pose is a Z-mirror — common in WoWS main-battery
/// + twin-secondary assets. Such bones produce a `mount.transform` with
/// `det == -1`, which is geometrically correct but inconvenient for
/// consumers (decomposition into PRS yields a negative scale axis, which
/// flips face winding under default culling).
fn mat3_determinant(m: &[f32; 16]) -> f32 {
    m[0] * (m[5] * m[10] - m[6] * m[9])
        - m[4] * (m[1] * m[10] - m[2] * m[9])
        + m[8] * (m[1] * m[6] - m[2] * m[5])
}

/// Convert an improper rotation (`det < 0`) to a proper rotation by
/// negating column 0 of the 3x3 part. For the WoWS-specific case where
/// the impropriety comes from a Z-mirror baked into `Rotate_Y_BlendBone`
/// (col 2's Z component being negative), this yields `Ry(180°)` —
/// visually identical to the original Z-mirror for any bilaterally
/// symmetric asset. Every gun mount in the WoWS fleet is bilaterally
/// symmetric, so the substitution is safe.
///
/// Translation column is preserved verbatim. No-op for proper rotations.
fn ensure_proper_rotation(m: [f32; 16]) -> [f32; 16] {
    if mat3_determinant(&m) >= 0.0 {
        return m;
    }
    let mut out = m;
    out[0] = -out[0];
    out[1] = -out[1];
    out[2] = -out[2];
    out
}

/// Identity 4x4 column-major matrix — translation (0,0,0), no rotation.
/// Used as the fallback transform for a mount whose hardpoint couldn't be
/// resolved (the mount is still emitted so downstream logic sees the asset_id).
const IDENTITY_4X4: [f32; 16] =
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// 180° rotation around the Y axis, column-major. Equivalent to
/// `diag(-1, 1, -1, 1)` in 4x4 form. Post-multiplied onto skel_ext
/// placements (schema_version=2 emit) so consumers don't need a
/// per-asset facing flip — see `write_skel_ext_candidates_json` for
/// the motivation.
const RY_180_4X4: [f32; 16] = [
    -1.0, 0.0, 0.0, 0.0,
     0.0, 1.0, 0.0, 0.0,
     0.0, 0.0,-1.0, 0.0,
     0.0, 0.0, 0.0, 1.0,
];

/// Extract `(scope, category, subcategory)` from a mount's GameParams model path.
///
/// Example: `content/gameplay/usa/gun/main/AGM034_16in50_Mk7/AGM034_16in50_Mk7.model`
/// yields `("usa", "gun", "main")`. Paths with more nesting have the full
/// subtree (joined by `/`) collapsed into `subcategory`; paths with only
/// scope + category yield `subcategory = None`.
///
/// Returns `(None, None, None)` for an unrecognised shape (e.g. empty string)
/// rather than silently inventing fields — the sidecar prefers missing over
/// wrong when the toolkit can't classify.
fn parse_mount_path_taxonomy(model_path: &str) -> (Option<String>, Option<String>, Option<String>) {
    // Strip the root `content/gameplay/` prefix when present. Everything else
    // is kept case-sensitive per CLAUDE.md ("VFS path casing") — we never
    // case-fold these tokens.
    let tail = model_path
        .strip_prefix("content/gameplay/")
        .or_else(|| model_path.strip_prefix("/content/gameplay/"))
        .unwrap_or(model_path);

    // Drop the trailing `<AssetId>/<AssetId>.model` so the remaining segments
    // are just the taxonomy. Walk segments and stop at the first one that
    // matches the following segment's stem (that's the asset dir).
    let mut segs: Vec<&str> = tail.split('/').collect();
    // Drop the `.model` leaf.
    if segs.last().is_some_and(|s| s.ends_with(".model") || s.ends_with(".geometry") || s.ends_with(".visual")) {
        segs.pop();
    }
    // Drop the asset dir (same stem as the popped file).
    if segs.len() >= 2 {
        let last = segs[segs.len() - 1];
        let prev = segs[segs.len() - 2];
        if last == prev {
            segs.pop();
        } else if segs.len() >= 2 {
            // Asset dir without matching leaf stem — still pop one.
            segs.pop();
        }
    }

    let scope = segs.first().map(|s| s.to_string());
    let category = segs.get(1).map(|s| s.to_string());
    let subcategory = if segs.len() > 2 { Some(segs[2..].join("/")) } else { None };

    (scope, category, subcategory)
}

/// Format a `SystemTime` as an RFC 3339 UTC timestamp (seconds precision).
///
/// Avoids pulling `chrono` into the wowsunpack crate; the placements manifest
/// only wants a monotonic "when was this generated" marker for audit.
fn format_rfc3339_utc(time: SystemTime) -> String {
    let secs = time.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    // Civil-date conversion (Howard Hinnant's algorithm, public domain).
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let hour = (time_of_day / 3600) as u32;
    let minute = ((time_of_day % 3600) / 60) as u32;
    let second = (time_of_day % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5) + 1;
    let m = if mp < 10 { mp + 3 } else { mp.wrapping_sub(9) };
    let year = if m <= 2 { y + 1 } else { y };

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, m, d, hour, minute, second)
}

/// Pull nation / species / tier out of a resolved GameParam.
///
/// Returns empty strings and tier=0 if the param is None or doesn't expose
/// the expected Vehicle data. Species is emitted as its canonical
/// GameParams-side string (``"Destroyer"`` / ``"Battleship"`` / ``"Cruiser"``
/// / ``"AirCarrier"`` / ``"Submarine"``); downstream consumers map that to
/// their preferred short form.
fn ship_identity_from_param(
    param: Option<&crate::game_params::types::Param>,
) -> (String, String, u32) {
    let Some(param) = param else {
        return (String::new(), String::new(), 0);
    };
    let nation = param.nation().to_string();
    let species = param
        .species()
        .and_then(|r| r.known())
        .map(|s| {
            use crate::game_params::types::Species::*;
            match s {
                AirCarrier => "AirCarrier",
                Battleship => "Battleship",
                Cruiser => "Cruiser",
                Destroyer => "Destroyer",
                Submarine => "Submarine",
                _ => "",
            }
            .to_string()
        })
        .unwrap_or_default();
    let tier = param.vehicle().map(|v| v.level()).unwrap_or(0);
    (nation, species, tier)
}

/// Map a mount's species to a display group name.
fn mount_group(species: Option<crate::game_params::types::MountSpecies>) -> &'static str {
    match species {
        Some(s) => s.display_group(),
        None => "Other",
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// One-shot: load game data, resolve ship, export GLB.
///
/// For multiple ships, use [`ShipAssets`] directly to amortize the ~18s
/// GameParams parsing cost.
pub fn export_ship_glb(
    vfs: &VfsPath,
    name: &str,
    options: &ShipExportOptions,
    writer: &mut impl Write,
) -> Result<ShipInfo, Report> {
    let assets = ShipAssets::load(vfs)?;
    let ctx = assets.load_ship(name, options)?;
    let info = ctx.info().clone();
    ctx.export_glb(writer)?;
    Ok(info)
}
