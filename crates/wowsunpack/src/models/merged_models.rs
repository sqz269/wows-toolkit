//! Parser for space-level `models.bin` (MergedModels) files.
//!
//! These files contain all model instances for a space/map. Each record packs a
//! ModelPrototype, VisualPrototype, and SkeletonProto into a flat 0xA8-byte
//! record with struct-base-relative relptrs.
//!
//! See MODELS.md § "MergedModels (`models.bin`) Format" for full field layout.

use rootcause::Report;
use thiserror::Error;
use winnow::Parser;
use winnow::binary::le_f32;
use winnow::binary::le_i64;
use winnow::binary::le_u8;
use winnow::binary::le_u16;
use winnow::binary::le_u32;
use winnow::binary::le_u64;
use winnow::error::ContextError;
use winnow::error::ErrMode;
use winnow::token::take;

use crate::data::parser_utils;
use crate::data::parser_utils::BoundingBox;
use crate::data::parser_utils::Matrix4x4;
use crate::data::parser_utils::WResult;
use crate::data::parser_utils::parse_lod_fields;
use crate::data::parser_utils::parse_matrix_array;
use crate::data::parser_utils::parse_render_set_fields;
use crate::data::parser_utils::parse_u16_array;
use crate::data::parser_utils::parse_u32_array;
use crate::data::parser_utils::resolve_relptr;
use crate::data::parser_utils::resolve_relptr_at;
use crate::models::model::ModelPrototype;
use crate::models::visual::Lod;
use crate::models::visual::RenderSet;
use crate::models::visual::VisualNodes;
use crate::models::visual::VisualPrototype;

/// Errors during `models.bin` parsing.
#[derive(Debug, Error)]
pub enum MergedModelsError {
    #[error("data too short: need {need} bytes at offset 0x{offset:X}, have {have}")]
    DataTooShort { offset: usize, need: usize, have: usize },
    #[error("parse error: {0}")]
    ParseError(String),
}

/// Parsed `models.bin` file.
#[derive(Debug)]
pub struct MergedModels {
    pub models: Vec<MergedModelRecord>,
    pub skeletons: Vec<SkeletonProto>,
    pub model_bone_count: u16,
}

/// A single model record from the merged array.
#[derive(Debug)]
pub struct MergedModelRecord {
    /// selfId identifying this model in pathsStorage.
    pub path_id: u64,
    /// Inlined ModelPrototype fields.
    pub model_proto: ModelPrototype,
    /// Inlined VisualPrototype (includes inline SkeletonProto).
    pub visual_proto: VisualPrototype,
    /// Index into the shared skeletons array.
    pub skeleton_proto_index: u32,
    /// First geometry mapping index for this model's render sets.
    pub render_set_geometry_start_idx: u16,
    /// Number of geometry mappings for this model.
    pub render_set_geometry_count: u16,
}

/// Shared skeleton prototype (stride 0x30).
#[derive(Debug)]
pub struct SkeletonProto {
    pub nodes: VisualNodes,
}

/// A per-instance dye override: an 8-byte `{matter_id, replaces_id}` pair.
/// `matter_id` is the new texture/color to apply; `replaces_id` is the
/// prototype slot it targets.
///
/// Engine treats both fields as opaque u32 IDs. Empirically the encoding
/// varies between maps: some carry `MurmurHash3_32` hashes (consistent
/// with ship-side camo per `[[project_mat_camo_hybrid_shipped]]`),
/// others store 4-byte ASCII fragments in the u32 slots directly (e.g.
/// `0x38303134 = "4108"` followed by `0x43363244 = "D26C"`). Consumers
/// must not assume one encoding — preserve the raw u32 and resolve
/// downstream against whichever hash/string table the engine uses for
/// that map.
#[derive(Debug, Clone, Copy)]
pub struct ModelDye {
    pub matter_id: u32,
    pub replaces_id: u32,
}

/// A single PointLightInstance from `space.bin` (0xc0-stride record from
/// the `pointLights[]` sub-array). Engine struct: transform at +0x00
/// (16-float 4×4) → opaque properties array at +0x40 → inline
/// `Lighting::PointLightPrototype` at +0x50. We compute the world position
/// up-front (transform × localPosition) so consumers can drop the value
/// straight into a `THREE.PointLight.position` without re-doing the math.
///
/// Animation tracks (colorAnimation, radiusAnimation) at +0x50 / +0x70 of
/// the instance are skipped — keyframe data lives behind separate
/// relptrs and v1 consumers don't need it.
///
/// See `map_extraction_audit_2026_05_21.md` § "pointLights[]" + RE of
/// FUN_140899190 (instance reader) + FUN_140712390 (prototype reader).
#[derive(Debug, Clone)]
pub struct SpacePointLight {
    /// World-space position (instance transform applied to prototype
    /// `localPosition`). Most prototypes carry `localPosition = (0, 0, 0)`
    /// so this typically equals the transform's translation column.
    pub world_position: [f32; 3],
    /// Engine `color` Vec4 RGBA. Convention: RGB is linear color, alpha is
    /// the intensity multiplier consumers feed into `THREE.PointLight`.
    pub color: [f32; 4],
    /// Falloff radius in metres. Maps to `THREE.PointLight.distance`.
    pub radius: f32,
    /// Engine `Quality` enum minimum: 0=Low, 1=Medium, 2=High, 3=Ultra.
    /// Engine skips the light when the runtime quality is below this.
    pub min_quality: u32,
}

/// A single model instance from `space.bin`, combining a world transform
/// with a reference to the model prototype via `path_id`.
///
/// Per-instance metadata (`is_landscape`, `min_quality_level`) is surfaced
/// from engine offsets +0x58 / +0x5b of the 0x70-stride ModelInstance
/// record. The dye / material-instance override arrays at +0x60 / +0x68
/// are parsed but not yet exported (those need a separate consumer
/// pass — see audit doc Phase 2 backlog).
#[derive(Debug)]
pub struct SpaceInstance {
    /// 4×4 world transform matrix (column-major, row 3 = translation + w=1).
    pub transform: Matrix4x4,
    /// selfId matching a `MergedModelRecord::path_id` in the sibling `models.bin`.
    pub path_id: u64,
    /// Engine `isLandscape` flag — true for LNR* / TILEDLAND backdrop
    /// landmass proxies that the engine renders with a more aggressive
    /// `landscapeBias` LOD policy. Consumers can use this to apply
    /// distance fog or LOD distance gating that matches engine behavior.
    pub is_landscape: bool,
    /// Engine `minimumQualityLevel` (graphics::preferences::Quality enum:
    /// 0=Low, 1=Medium, 2=High, 3=Ultra). The engine skips this instance
    /// when the runtime quality preset is below this value; useful as a
    /// viewer-side detail filter.
    pub min_quality_level: u8,
    /// Per-instance `modelDyes[]` override pairs from +0x60 (relptr) /
    /// +0x59 (u8 count). Resolved at parse time; relptr is rel-to-position
    /// (rec_base+0x60 + i64). Visual impact varies by map — themed event
    /// maps carry hundreds, "plain" maps (Okinawa) carry zero.
    pub model_dyes: Vec<ModelDye>,
    /// Count of `materialInstances[]` override records at +0x68 (relptr)
    /// / +0x5a (u8 count). v1 surfaces the count only; the 0x70-stride
    /// `MaterialInstancePrototype` records carry full per-instance
    /// material property bags (Vec4 tints, texture swaps, shader vars).
    /// Decoding them requires reusing `models/material.rs` with a stride
    /// adjustment — deferred to a follow-up pass.
    pub material_instance_count: u8,
}

/// Parsed `space.bin` instance placements + lighting.
#[derive(Debug)]
pub struct SpaceInstances {
    pub instances: Vec<SpaceInstance>,
    pub point_lights: Vec<SpacePointLight>,
}

// ── Winnow sub-parsers (merged-specific) ────────────────────────────────────

// models.bin header (0x18 bytes)

struct MergedHeader {
    models_count: u32,
    skeletons_count: u16,
    model_bone_count: u16,
    models_relptr: i64,
    skeletons_relptr: i64,
}

fn parse_merged_header(input: &mut &[u8]) -> WResult<MergedHeader> {
    let models_count = le_u32.parse_next(input)?;
    let skeletons_count = le_u16.parse_next(input)?;
    let model_bone_count = le_u16.parse_next(input)?;
    let models_relptr = le_i64.parse_next(input)?;
    let skeletons_relptr = le_i64.parse_next(input)?;
    Ok(MergedHeader { models_count, skeletons_count, model_bone_count, models_relptr, skeletons_relptr })
}

// VisualProto inline fields (0x70 bytes at rec+0x30)

struct VisualProtoInlineFields {
    nodes_count: u32,
    name_map_name_ids_relptr: i64,
    name_map_node_ids_relptr: i64,
    name_ids_relptr: i64,
    matrices_relptr: i64,
    parent_ids_relptr: i64,
    merged_geometry_path_id: u64,
    underwater_model: bool,
    abovewater_model: bool,
    render_sets_count: u16,
    lods_count: u16,
    bounding_box: BoundingBox,
    render_sets_relptr: i64,
    lods_relptr: i64,
}

fn parse_visual_proto_inline_fields(input: &mut &[u8]) -> WResult<VisualProtoInlineFields> {
    // Skeleton sub-struct: +0x00..+0x30
    let nodes_count = le_u32.parse_next(input)?;
    let _pad = le_u32.parse_next(input)?;
    let name_map_name_ids_relptr = le_i64.parse_next(input)?;
    let name_map_node_ids_relptr = le_i64.parse_next(input)?;
    let name_ids_relptr = le_i64.parse_next(input)?;
    let matrices_relptr = le_i64.parse_next(input)?;
    let parent_ids_relptr = le_i64.parse_next(input)?;
    // VisualProto fields: +0x30..+0x70
    let merged_geometry_path_id = le_u64.parse_next(input)?;
    let underwater_model = le_u8.parse_next(input)? != 0;
    let abovewater_model = le_u8.parse_next(input)? != 0;
    let render_sets_count = le_u16.parse_next(input)?;
    let lods_count = le_u16.parse_next(input)?;
    let _ = take(2usize).parse_next(input)?; // padding to +0x40
    let bounding_box = parser_utils::parse_bounding_box(input)?;
    let render_sets_relptr = le_i64.parse_next(input)?;
    let lods_relptr = le_i64.parse_next(input)?;
    Ok(VisualProtoInlineFields {
        nodes_count,
        name_map_name_ids_relptr,
        name_map_node_ids_relptr,
        name_ids_relptr,
        matrices_relptr,
        parent_ids_relptr,
        merged_geometry_path_id,
        underwater_model,
        abovewater_model,
        render_sets_count,
        lods_count,
        bounding_box,
        render_sets_relptr,
        lods_relptr,
    })
}

// space.bin instance entry (engine `SpaceContent::ModelInstance`, stride 0x70)
//
// Layout (Ghidra @ FUN_1408985c0, audit doc 2026-05-21):
//   +0x00  16× f32   transform
//   +0x40  u32       guidCount       — DROPPED
//   +0x44  i64       guids relptr    — DROPPED
//   +0x50  u64       resourceId      — path_id
//   +0x58  u8        isLandscape
//   +0x59  u8        modelDyesCount
//   +0x5a  u8        materialInstanceCount
//   +0x5b  u8        minimumQualityLevel
//   +0x5c  4 bytes   pad
//   +0x60  i64       modelDyes relptr            — DROPPED (Phase 2)
//   +0x68  i64       materialInstances relptr    — DROPPED (Phase 2)

/// Parse a single ModelInstance record (0x70 bytes) into a fully-resolved
/// `SpaceInstance`. Inner relptrs at +0x60 (modelDyes) and +0x68
/// (materialInstances) are resolved using `rec_base` as the position
/// reference — both are rel-to-position-within-file (verified
/// empirically on `20_NE_two_brothers/space.bin`).
fn parse_space_instance_record(
    file_data: &[u8],
    rec_base: usize,
) -> WResult<SpaceInstance> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_INSTANCE_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)?;
    let _ = take(16usize).parse_next(input)?; // +0x40..+0x50: guidCount + guids relptr
    let path_id = le_u64.parse_next(input)?;
    let is_landscape = le_u8.parse_next(input)? != 0;
    let model_dyes_count = le_u8.parse_next(input)?;
    let material_instance_count = le_u8.parse_next(input)?;
    let min_quality_level = le_u8.parse_next(input)?;
    let _ = take(4usize).parse_next(input)?; // +0x5c..+0x60: pad
    let model_dyes_relptr = le_i64.parse_next(input)?;
    let _material_instances_relptr = le_i64.parse_next(input)?;

    // Resolve modelDyes[]: rec_base+0x60 + relptr → start of N×8 dye records.
    // The relptr can legitimately be 0 (no overrides — file has padding here)
    // when count is 0; defensively guard against out-of-bounds + negative
    // resolved offsets.
    let mut model_dyes = Vec::with_capacity(model_dyes_count as usize);
    if model_dyes_count > 0 {
        let dye_pos = rec_base + 0x60;
        let resolved = dye_pos as i64 + model_dyes_relptr;
        if resolved >= 0 {
            let start = resolved as usize;
            let need = (model_dyes_count as usize) * 8;
            if start + need <= file_data.len() {
                let mut dye_input = &file_data[start..start + need];
                for _ in 0..model_dyes_count {
                    let matter_id = le_u32.parse_next(&mut dye_input)?;
                    let replaces_id = le_u32.parse_next(&mut dye_input)?;
                    model_dyes.push(ModelDye { matter_id, replaces_id });
                }
            }
        }
    }

    Ok(SpaceInstance {
        transform,
        path_id,
        is_landscape,
        min_quality_level,
        model_dyes,
        material_instance_count,
    })
}

// ── Helper: parse array at offset, wrapping winnow errors ───────────────────

fn parse_array_at<T>(
    data: &[u8],
    offset: usize,
    count: usize,
    parser: fn(&mut &[u8], usize) -> WResult<Vec<T>>,
) -> Result<Vec<T>, Report<MergedModelsError>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let input = &mut &data[offset..];
    parser(input, count).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("array at 0x{offset:X}: {e}")))
    })
}

// ── Header ───────────────────────────────────────────────────────────────────

const HEADER_SIZE: usize = 0x18;
const MODEL_RECORD_SIZE: usize = 0xA8;
const SKELETON_SIZE: usize = 0x30;
const RENDER_SET_SIZE: usize = 0x28;
const LOD_SIZE: usize = 0x10;

/// Parse a `models.bin` file.
pub fn parse_merged_models(file_data: &[u8]) -> Result<MergedModels, Report<MergedModelsError>> {
    if file_data.len() < HEADER_SIZE {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: 0,
            need: HEADER_SIZE,
            have: file_data.len(),
        }));
    }

    let input = &mut &file_data[..];
    let hdr = parse_merged_header(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("header: {e}"))))?;

    let models_count = hdr.models_count as usize;
    let skeletons_count = hdr.skeletons_count as usize;
    let models_offset = resolve_relptr(0, hdr.models_relptr);
    let skeletons_offset = resolve_relptr(0, hdr.skeletons_relptr);

    // Parse model records
    let need = models_count * MODEL_RECORD_SIZE;
    if models_offset + need > file_data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: models_offset,
            need,
            have: file_data.len(),
        }));
    }
    let mut models = Vec::with_capacity(models_count);
    for i in 0..models_count {
        let rec_base = models_offset + i * MODEL_RECORD_SIZE;
        let record = parse_model_record(file_data, rec_base)
            .map_err(|e| MergedModelsError::ParseError(format!("model[{i}]: {e}")))?;
        models.push(record);
    }

    // Parse shared skeleton prototypes
    let need = skeletons_count * SKELETON_SIZE;
    if skeletons_offset + need > file_data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: skeletons_offset,
            need,
            have: file_data.len(),
        }));
    }
    let mut skeletons = Vec::with_capacity(skeletons_count);
    for i in 0..skeletons_count {
        let skel_base = skeletons_offset + i * SKELETON_SIZE;
        let skeleton = parse_skeleton_proto(file_data, skel_base)
            .map_err(|e| MergedModelsError::ParseError(format!("skeleton[{i}]: {e}")))?;
        skeletons.push(skeleton);
    }

    Ok(MergedModels { models, skeletons, model_bone_count: hdr.model_bone_count })
}

// ── space.bin parser ─────────────────────────────────────────────────────────

const SPACE_HEADER_SIZE: usize = 0x60;
const SPACE_INSTANCE_SIZE: usize = 0x70;
const SPACE_POINT_LIGHT_SIZE: usize = 0xc0;

// Header layout (see audit doc § "8 typed sub-arrays"):
//   +0x00..+0x20  eight u32 counts in declaration order
//   +0x20..+0x60  eight i64 relptrs in declaration order
// Sub-array order: models, obstacles, particles, pointLights, probes,
// staticDecals, userObjects, prefabs.
const SUBARRAY_MODELS: usize = 0;
const SUBARRAY_POINT_LIGHTS: usize = 3;

/// Parse the 8 (count, relptr) pairs out of the space.bin header.
///
/// Header relptrs are absolute byte offsets from the start of the file
/// (base=0), same convention as `parse_merged_models` for models.bin.
/// This contrasts with deeper engine structs where relptrs are typically
/// stored relative to the pointer's own file position; for the
/// outer `SpaceContent::Instances` block the engine resolves them against
/// the file base. Empirically confirmed by inspecting Okinawa's
/// space.bin: `models[]` relptr value is `0x60`, which lands on a clean
/// Matrix4x4 record at file offset `0x60` (not `0x80` as a
/// relative-to-pointer convention would imply).
fn parse_space_header(file_data: &[u8]) -> Result<[(u32, usize); 8], Report<MergedModelsError>> {
    if file_data.len() < SPACE_HEADER_SIZE {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: 0,
            need: SPACE_HEADER_SIZE,
            have: file_data.len(),
        }));
    }
    let input = &mut &file_data[..SPACE_HEADER_SIZE];
    let counts: Vec<u32> = winnow::combinator::repeat(8usize, le_u32)
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("space counts: {e}")))
        })?;
    let relptrs: Vec<i64> = winnow::combinator::repeat(8usize, le_i64)
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("space relptrs: {e}")))
        })?;
    let mut out = [(0u32, 0usize); 8];
    for i in 0..8 {
        // Absolute-from-file-start, NOT base+offset. See doc above.
        out[i] = (counts[i], relptrs[i].max(0) as usize);
    }
    Ok(out)
}

/// Parse a `space.bin` file to extract model placements + point lights.
///
/// The engine's `SpaceContent::Instances` block carries eight typed
/// sub-arrays; we currently consume two: models (visible meshes) and
/// pointLights (atmospheric lighting). The other six are intentionally
/// dropped — see the audit doc backlog.
pub fn parse_space_instances(file_data: &[u8]) -> Result<SpaceInstances, Report<MergedModelsError>> {
    let header = parse_space_header(file_data)?;
    let (instance_count, instances_offset) = header[SUBARRAY_MODELS];
    let (light_count, lights_offset) = header[SUBARRAY_POINT_LIGHTS];

    let instances = if instance_count == 0 {
        Vec::new()
    } else {
        let need = instance_count as usize * SPACE_INSTANCE_SIZE;
        if instances_offset + need > file_data.len() {
            return Err(Report::new(MergedModelsError::DataTooShort {
                offset: instances_offset,
                need,
                have: file_data.len(),
            }));
        }
        let mut out = Vec::with_capacity(instance_count as usize);
        for i in 0..instance_count as usize {
            let rec_base = instances_offset + i * SPACE_INSTANCE_SIZE;
            let rec = parse_space_instance_record(file_data, rec_base).map_err(
                |e: ErrMode<ContextError>| {
                    Report::new(MergedModelsError::ParseError(format!(
                        "space instance[{i}] @ 0x{rec_base:x}: {e}"
                    )))
                },
            )?;
            out.push(rec);
        }
        out
    };

    let point_lights = if light_count == 0 {
        Vec::new()
    } else {
        let need = light_count as usize * SPACE_POINT_LIGHT_SIZE;
        if lights_offset + need > file_data.len() {
            return Err(Report::new(MergedModelsError::DataTooShort {
                offset: lights_offset,
                need,
                have: file_data.len(),
            }));
        }
        let input = &mut &file_data[lights_offset..];
        winnow::combinator::repeat(light_count as usize, parse_space_point_light_entry)
            .parse_next(input)
            .map_err(|e: ErrMode<ContextError>| {
                Report::new(MergedModelsError::ParseError(format!("space point lights: {e}")))
            })?
    };

    Ok(SpaceInstances { instances, point_lights })
}

// PointLightInstance layout (0xc0 stride). Cursor offsets are checked
// against the reflection registrations in FUN_140899190 / FUN_140712390.
//
//   +0x00  Matrix4x4  transform     (16 f32, 4×4 column-major like ModelInstance)
//   +0x40  16 bytes   <unnamed>     (count u32 + pad + relptr — opaque, skip)
//   +0x50  AnimProto  colorAnimation  (0x20 bytes — AnimationPrototype<Vec4>, skip)
//   +0x70  AnimProto  radiusAnimation (0x20 bytes — AnimationPrototype<f32>,  skip)
//   +0x90  Vec4       color          (RGBA f32; A is intensity)
//   +0xa0  Vec3+pad   localPosition  (12 f32 + 4 byte pad)
//   +0xb0  f32        radius
//   +0xb4  u32        minQuality
//   +0xb8  2 bytes    animatedColor / animatedRadius (bools, skip)
//   +0xba  6 bytes    padding to 0xc0
fn parse_space_point_light_entry(input: &mut &[u8]) -> WResult<SpacePointLight> {
    let transform = parser_utils::parse_matrix4x4(input)?;
    let _ = take(16usize).parse_next(input)?; // +0x40 array desc (opaque)
    let _ = take(32usize).parse_next(input)?; // +0x50 colorAnimation
    let _ = take(32usize).parse_next(input)?; // +0x70 radiusAnimation
    let cr = le_f32.parse_next(input)?;
    let cg = le_f32.parse_next(input)?;
    let cb = le_f32.parse_next(input)?;
    let ca = le_f32.parse_next(input)?;
    let lx = le_f32.parse_next(input)?;
    let ly = le_f32.parse_next(input)?;
    let lz = le_f32.parse_next(input)?;
    let _ = take(4usize).parse_next(input)?; // localPosition pad
    let radius = le_f32.parse_next(input)?;
    let min_quality = le_u32.parse_next(input)?;
    let _ = take(8usize).parse_next(input)?; // animatedColor + animatedRadius + pad
    Ok(SpacePointLight {
        world_position: transform_point(&transform, [lx, ly, lz]),
        color: [cr, cg, cb, ca],
        radius,
        min_quality,
    })
}

/// Apply a glTF-style column-major 4×4 transform to a 3D point. Mirrors
/// the convention used by [`SpaceInstance::transform`] — translation lives
/// in elements 12..15.
fn transform_point(m: &Matrix4x4, p: [f32; 3]) -> [f32; 3] {
    let a = &m.0;
    [
        a[0] * p[0] + a[4] * p[1] + a[8] * p[2] + a[12],
        a[1] * p[0] + a[5] * p[1] + a[9] * p[2] + a[13],
        a[2] * p[0] + a[6] * p[1] + a[10] * p[2] + a[14],
    ]
}

// ── Model Record ─────────────────────────────────────────────────────────────

fn parse_model_record(data: &[u8], rec: usize) -> Result<MergedModelRecord, Report<MergedModelsError>> {
    if rec + MODEL_RECORD_SIZE > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: rec,
            need: MODEL_RECORD_SIZE,
            have: data.len(),
        }));
    }

    let input = &mut &data[rec..];
    let path_id = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("path_id: {e}"))))?;

    // ModelProto inlined at rec+0x08 (0x28 bytes, same shape as assets.bin blob 3
    // for the header — but the OOL arrays it points to use a DIFFERENT
    // representation. In assets.bin, animations[] is a packed array of full
    // ModelPrototype records (0x28 each, recursively). In models.bin's
    // inlined form, animations[] is a packed array of u64 selfIds (8 bytes
    // each) referencing animation entries elsewhere. Calling the regular
    // `parse_model` here recurses through what it thinks are nested
    // ModelPrototypes and reads garbage as skel_ext_count, blowing up
    // (e.g. Dock model[14], 14_Atlantic model[81], any record with
    // animations_count > 0).
    //
    // Workaround: parse the 0x28-byte header inline and leave skel_ext /
    // animations / dyes empty. None of those fields are consumed downstream
    // when rendering a map (only path_id, visual_proto, skeleton_proto_index,
    // and the geometry mapping range matter), so dropping them is safe.
    let model_proto_base = rec + 0x08;
    let model_proto = parse_inline_model_proto_header(data, model_proto_base)?;

    // VisualProto at rec+0x30 (0x70 bytes)
    let vp_base = rec + 0x30;
    let visual_proto = parse_visual_proto_inline(data, vp_base)?;

    // Tail fields at rec+0xA0
    let input = &mut &data[rec + 0xA0..];
    let skeleton_proto_index = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("skeleton_proto_index: {e}")))
    })?;
    let render_set_geometry_start_idx = le_u16.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("rs_geom_start: {e}")))
    })?;
    let render_set_geometry_count = le_u16.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("rs_geom_count: {e}")))
    })?;

    Ok(MergedModelRecord {
        path_id,
        model_proto,
        visual_proto,
        skeleton_proto_index,
        render_set_geometry_start_idx,
        render_set_geometry_count,
    })
}

// Read the 0x28-byte inline ModelPrototype header without recursing into the
// OOL arrays. See parse_model_record's call-site comment for why the regular
// `parse_model` is unsafe here: animations in inline form are u64 selfIds,
// not nested ModelPrototype records, so the recursive parser misinterprets
// them. Map-rendering downstream doesn't read these fields, so leaving them
// empty is safe.
fn parse_inline_model_proto_header(
    data: &[u8],
    base: usize,
) -> Result<ModelPrototype, Report<MergedModelsError>> {
    const INLINE_MODEL_PROTO_SIZE: usize = 0x28;
    if base + INLINE_MODEL_PROTO_SIZE > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort {
            offset: base,
            need: INLINE_MODEL_PROTO_SIZE,
            have: data.len(),
        }));
    }
    let input = &mut &data[base..];
    let visual_resource_id = le_u64.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp visual_resource_id: {e}")))
    })?;
    let _skel_ext_count = le_u8.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp skel_ext_count: {e}")))
    })?;
    let misc_type = le_u8.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp misc_type: {e}")))
    })?;
    // animations_count, dyes_count, padding(4), 3× i64 relptrs — read & discard
    let _ = take(2usize + 4 + 24).parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp tail: {e}")))
    })?;
    Ok(ModelPrototype {
        visual_resource_id,
        misc_type,
        skel_ext_res_ids: Vec::new(),
        animations: Vec::new(),
        dyes: Vec::new(),
    })
}

// ── VisualProto (inline at rec+0x30, size 0x70) ─────────────────────────────

fn parse_visual_proto_inline(data: &[u8], vp_base: usize) -> Result<VisualPrototype, Report<MergedModelsError>> {
    if vp_base + 0x70 > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort { offset: vp_base, need: 0x70, have: data.len() }));
    }

    let input = &mut &data[vp_base..];
    let fields = parse_visual_proto_inline_fields(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("visual_proto at 0x{vp_base:X}: {e}")))
    })?;

    let nodes_count = fields.nodes_count as usize;

    let nodes = if nodes_count > 0 {
        let name_map_name_ids = parse_array_at(
            data,
            resolve_relptr(vp_base, fields.name_map_name_ids_relptr),
            nodes_count,
            parse_u32_array,
        )?;
        let name_map_node_ids = parse_array_at(
            data,
            resolve_relptr(vp_base, fields.name_map_node_ids_relptr),
            nodes_count,
            parse_u16_array,
        )?;
        let name_ids =
            parse_array_at(data, resolve_relptr(vp_base, fields.name_ids_relptr), nodes_count, parse_u32_array)?;
        let matrices =
            parse_array_at(data, resolve_relptr(vp_base, fields.matrices_relptr), nodes_count, parse_matrix_array)?;
        let parent_ids =
            parse_array_at(data, resolve_relptr(vp_base, fields.parent_ids_relptr), nodes_count, parse_u16_array)?;

        VisualNodes { name_map_name_ids, name_map_node_ids, name_ids, matrices, parent_ids }
    } else {
        VisualNodes {
            name_map_name_ids: Vec::new(),
            name_map_node_ids: Vec::new(),
            name_ids: Vec::new(),
            matrices: Vec::new(),
            parent_ids: Vec::new(),
        }
    };

    let render_sets_count = fields.render_sets_count as usize;
    let lods_count = fields.lods_count as usize;

    let render_sets = if render_sets_count > 0 {
        let rs_abs = resolve_relptr(vp_base, fields.render_sets_relptr);
        parse_render_sets_merged(data, rs_abs, render_sets_count)?
    } else {
        Vec::new()
    };

    let lods = if lods_count > 0 {
        let lod_abs = resolve_relptr(vp_base, fields.lods_relptr);
        parse_lods_merged(data, lod_abs, lods_count)?
    } else {
        Vec::new()
    };

    Ok(VisualPrototype {
        nodes,
        merged_geometry_path_id: fields.merged_geometry_path_id,
        underwater_model: fields.underwater_model,
        abovewater_model: fields.abovewater_model,
        bounding_box: fields.bounding_box,
        render_sets,
        lods,
    })
}

// ── Skeleton nodes ──────────────────────────────────────────────────────────

fn parse_skeleton_nodes(data: &[u8], skel_base: usize) -> Result<VisualNodes, Report<MergedModelsError>> {
    if skel_base + 0x30 > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort { offset: skel_base, need: 0x30, have: data.len() }));
    }

    let input = &mut &data[skel_base..];
    let nodes_count = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("nodes_count: {e}"))))?
        as usize;

    if nodes_count == 0 {
        return Ok(VisualNodes {
            name_map_name_ids: Vec::new(),
            name_map_node_ids: Vec::new(),
            name_ids: Vec::new(),
            matrices: Vec::new(),
            parent_ids: Vec::new(),
        });
    }

    let name_map_name_ids = {
        let abs = resolve_relptr_at(data, skel_base, 0x08).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("skel relptr: {e}")))
        })?;
        parse_array_at(data, abs, nodes_count, parse_u32_array)?
    };
    let name_map_node_ids = {
        let abs = resolve_relptr_at(data, skel_base, 0x10).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("skel relptr: {e}")))
        })?;
        parse_array_at(data, abs, nodes_count, parse_u16_array)?
    };
    let name_ids = {
        let abs = resolve_relptr_at(data, skel_base, 0x18).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("skel relptr: {e}")))
        })?;
        parse_array_at(data, abs, nodes_count, parse_u32_array)?
    };
    let matrices = {
        let abs = resolve_relptr_at(data, skel_base, 0x20).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("skel relptr: {e}")))
        })?;
        parse_array_at(data, abs, nodes_count, parse_matrix_array)?
    };
    let parent_ids = {
        let abs = resolve_relptr_at(data, skel_base, 0x28).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("skel relptr: {e}")))
        })?;
        parse_array_at(data, abs, nodes_count, parse_u16_array)?
    };

    Ok(VisualNodes { name_map_name_ids, name_map_node_ids, name_ids, matrices, parent_ids })
}

fn parse_skeleton_proto(data: &[u8], skel_base: usize) -> Result<SkeletonProto, Report<MergedModelsError>> {
    let nodes = parse_skeleton_nodes(data, skel_base)?;
    Ok(SkeletonProto { nodes })
}

// ── RenderSet (stride 0x28) ─────────────────────────────────────────────────

fn parse_render_sets_merged(
    data: &[u8],
    offset: usize,
    count: usize,
) -> Result<Vec<RenderSet>, Report<MergedModelsError>> {
    let need = count * RENDER_SET_SIZE;
    if offset + need > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort { offset, need, have: data.len() }));
    }

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let rs_base = offset + i * RENDER_SET_SIZE;
        let input = &mut &data[rs_base..];

        let fields = parse_render_set_fields(input).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("render_set[{i}]: {e}")))
        })?;

        let node_name_ids = if fields.nodes_count > 0 {
            let abs = resolve_relptr(rs_base, fields.node_name_ids_relptr);
            parse_array_at(data, abs, fields.nodes_count as usize, parse_u32_array)?
        } else {
            Vec::new()
        };

        result.push(RenderSet {
            name_id: fields.name_id,
            material_name_id: fields.material_name_id,
            vertices_mapping_id: fields.vertices_mapping_id,
            indices_mapping_id: fields.indices_mapping_id,
            material_mfm_path_id: fields.material_mfm_path_id,
            skinned: fields.skinned,
            node_name_ids,
        });
    }

    Ok(result)
}

// ── LOD (stride 0x10) ───────────────────────────────────────────────────────

fn parse_lods_merged(data: &[u8], offset: usize, count: usize) -> Result<Vec<Lod>, Report<MergedModelsError>> {
    let need = count * LOD_SIZE;
    if offset + need > data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort { offset, need, have: data.len() }));
    }

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let lod_base = offset + i * LOD_SIZE;
        let input = &mut &data[lod_base..];

        let fields = parse_lod_fields(input)
            .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("lod[{i}]: {e}"))))?;

        let render_set_names = if fields.render_set_names_count > 0 {
            let abs = resolve_relptr(lod_base, fields.render_set_names_relptr);
            parse_array_at(data, abs, fields.render_set_names_count as usize, parse_u32_array)?
        } else {
            Vec::new()
        };

        result.push(Lod { extent: fields.extent, casts_shadow: fields.casts_shadow, render_set_names });
    }

    Ok(result)
}
