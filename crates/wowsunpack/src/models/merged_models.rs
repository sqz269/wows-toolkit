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
use winnow::binary::le_i32;
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
use crate::models::material::MaterialPrototype;
use crate::models::material::parse_material_instance;
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

/// A per-instance dye override: an 8-byte `{matter_id, tint_name_id}`
/// selection pair. `matter_id` matches a prototype `DyeEntry.matter_id`;
/// the second u32 (wire name `replaces_id` kept for extras/sidecar schema
/// stability) matches one of that dye's `tint_name_ids`, selecting which
/// pre-baked tint material the engine applies. Corpus-verified on
/// 01_solomon_islands / 20_NE_two_brothers / 54_Faroe (1,025/1,025 joins).
/// The older reading — `matter_id` is the new texture/color, `replaces_id` the
/// prototype slot it targets.
///
/// Engine treats both fields as opaque u32 IDs. Current corpus evidence
/// matches the native hash/key path; preserve the raw u32 values and resolve
/// them downstream against model/material dye tables before applying them.
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
    /// Engine point-light minimum-quality threshold, driven by the native
    /// DYNAMIC_LIGHTING graphics setting. Native quality tokens are descending:
    /// MAX/MAXIMUM=0, VERYHIGH=1, HIGH=2, MEDIUM=3, LOW=4, VERYLOW=5,
    /// MIN/MINIMUM=6. Consumers should draw when
    /// dynamic_lighting_runtime_raw <= min_quality.
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
    /// Engine `minimumQualityLevel`. Native code stages this byte into
    /// ecs::model::ModelPropertiesComponent byte 1 and filters map model ids
    /// against the OBJECT_LOD runtime raw byte. Native quality tokens are
    /// descending:
    /// MAX/MAXIMUM=0, VERYHIGH=1, HIGH=2, MEDIUM=3, LOW=4, VERYLOW=5,
    /// MIN/MINIMUM=6. Consumers should draw when
    /// object_lod_runtime_raw <= this value.
    pub min_quality_level: u8,
    /// Stable authoring GUID string at +0x40/+0x48, when present.
    pub stable_guid: Option<String>,
    /// Per-instance `modelDyes[]` override pairs from +0x60 (relptr) /
    /// +0x59 (u8 count). Resolved at parse time; relptr is relative to the
    /// start of the ModelInstance record. Visual impact varies by map.
    pub model_dyes: Vec<ModelDye>,
    /// Count of `materialInstances[]` override records at +0x68 (relptr)
    /// / +0x5a (u8 count).
    pub material_instance_count: u8,
    /// Decoded 0x70-stride `MaterialInstancePrototype` overrides.
    pub material_instances: Vec<MaterialPrototype>,
}

#[derive(Debug, Clone)]
pub struct SpaceObstacle {
    pub source_offset: usize,
    pub transform: Matrix4x4,
    pub position: [f32; 3],
    pub packed_indices: u32,
    pub candidate_model_instance_index: u16,
    pub candidate_collision_model_index: u16,
    pub field_44: u32,
    pub grid_min: [i32; 2],
    pub grid_max: [i32; 2],
}

#[derive(Debug, Clone)]
pub struct SpaceParticle {
    pub transform: Matrix4x4,
    pub position: [f32; 3],
    pub raw_guid_blob: [u8; 16],
    pub resource_id: u64,
    pub intensity_count: u32,
    pub intensity_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpaceStaticDecalHeader {
    pub technique: u32,
    pub influence: u32,
    pub field_08: u32,
    pub field_0c: u32,
    pub variant: u32,
    pub alpha: f32,
    pub field_18: u32,
    pub field_1c: u32,
}

#[derive(Debug, Clone)]
pub struct SpaceStaticDecal {
    pub transform: Matrix4x4,
    pub position: [f32; 3],
    pub header: SpaceStaticDecalHeader,
    pub texture_paths: Vec<Option<String>>,
}

#[derive(Debug, Clone)]
pub struct SpaceProbe {
    pub transform: Matrix4x4,
    pub position: [f32; 3],
    pub guid: Option<String>,
    pub name: Option<String>,
    pub resolution: u32,
    pub is_main_probe: bool,
    pub draw_full_scene: bool,
}

#[derive(Debug, Clone)]
pub struct SpaceUserObjectPropertyValue {
    pub path: String,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct SpaceUserObject {
    pub transform: Matrix4x4,
    pub position: [f32; 3],
    pub guid: Option<String>,
    pub object_type: Option<String>,
    pub properties_xml: Option<String>,
    pub properties_well_formed: bool,
    pub property_tags: Vec<String>,
    pub property_values: Vec<SpaceUserObjectPropertyValue>,
}

/// Parsed `space.bin` instance placements + lighting.
#[derive(Debug)]
pub struct SpaceInstances {
    pub instances: Vec<SpaceInstance>,
    pub obstacles: Vec<SpaceObstacle>,
    pub particles: Vec<SpaceParticle>,
    pub point_lights: Vec<SpacePointLight>,
    pub probes: Vec<SpaceProbe>,
    pub static_decals: Vec<SpaceStaticDecal>,
    pub user_objects: Vec<SpaceUserObject>,
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
//   +0x40  u64       stableGuid length
//   +0x48  i64       stableGuid relptr
//   +0x50  u64       resourceId      — path_id
//   +0x58  u8        isLandscape
//   +0x59  u8        modelDyesCount
//   +0x5a  u8        materialInstanceCount
//   +0x5b  u8        minimumQualityLevel
//   +0x5c  4 bytes   pad
//   +0x60  i64       modelDyes relptr
//   +0x68  i64       materialInstances relptr

/// Parse a single ModelInstance record (0x70 bytes) into a fully-resolved
/// `SpaceInstance`. Inner relptrs at +0x60 (modelDyes) and +0x68
/// (materialInstances) are resolved relative to the start of the 0x70-byte
/// ModelInstance record. The stable GUID descriptor at +0x40/+0x48 resolves
/// relative to the descriptor base (`rec_base + 0x40`).
fn parse_space_instance_record(file_data: &[u8], rec_base: usize) -> WResult<SpaceInstance> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_INSTANCE_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)?;
    let stable_guid_len = le_u64.parse_next(input)?;
    let stable_guid_relptr = le_i64.parse_next(input)?;
    let path_id = le_u64.parse_next(input)?;
    let is_landscape = le_u8.parse_next(input)? != 0;
    let model_dyes_count = le_u8.parse_next(input)?;
    let material_instance_count = le_u8.parse_next(input)?;
    let min_quality_level = le_u8.parse_next(input)?;
    let _ = take(4usize).parse_next(input)?; // +0x5c..+0x60: pad
    let model_dyes_relptr = le_i64.parse_next(input)?;
    let material_instances_relptr = le_i64.parse_next(input)?;

    let stable_guid = read_string_descriptor(file_data, rec_base + 0x40, stable_guid_len, stable_guid_relptr);

    // Resolve modelDyes[]: rec_base + relptr -> start of N x 8 dye records.
    // The relptr can legitimately be 0 (no overrides — file has padding here)
    // when count is 0; defensively guard against out-of-bounds + negative
    // resolved offsets.
    let mut model_dyes = Vec::with_capacity(model_dyes_count as usize);
    if model_dyes_count > 0 {
        let resolved = rec_base as i64 + model_dyes_relptr;
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

    let mut material_instances = Vec::with_capacity(material_instance_count as usize);
    if material_instance_count > 0 {
        let resolved = rec_base as i64 + material_instances_relptr;
        if resolved >= 0 {
            let start = resolved as usize;
            let need = (material_instance_count as usize) * crate::models::material::MATERIAL_INSTANCE_ITEM_SIZE;
            if start + need <= file_data.len() {
                for i in 0..material_instance_count as usize {
                    let material_offset = start + i * crate::models::material::MATERIAL_INSTANCE_ITEM_SIZE;
                    if let Ok(material) = parse_material_instance(&file_data[material_offset..]) {
                        material_instances.push(material);
                    }
                }
            }
        }
    }

    Ok(SpaceInstance {
        transform,
        path_id,
        is_landscape,
        min_quality_level,
        stable_guid,
        model_dyes,
        material_instance_count,
        material_instances,
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
const SPACE_OBSTACLE_SIZE: usize = 0x58;
const SPACE_PARTICLE_SIZE: usize = 0x68;
const SPACE_POINT_LIGHT_SIZE: usize = 0xc0;
const SPACE_PROBE_SIZE: usize = 0x70;
const SPACE_STATIC_DECAL_SIZE: usize = 0x68;
const SPACE_USER_OBJECT_SIZE: usize = 0x70;

// Header layout (see audit doc § "8 typed sub-arrays"):
//   +0x00..+0x20  eight u32 counts in declaration order
//   +0x20..+0x60  eight i64 relptrs in declaration order
// Sub-array order: models, obstacles, particles, pointLights, probes,
// staticDecals, userObjects, prefabs.
const SUBARRAY_MODELS: usize = 0;
const SUBARRAY_OBSTACLES: usize = 1;
const SUBARRAY_PARTICLES: usize = 2;
const SUBARRAY_POINT_LIGHTS: usize = 3;
const SUBARRAY_PROBES: usize = 4;
const SUBARRAY_STATIC_DECALS: usize = 5;
const SUBARRAY_USER_OBJECTS: usize = 6;

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
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("space counts: {e}"))))?;
    let relptrs: Vec<i64> =
        winnow::combinator::repeat(8usize, le_i64).parse_next(input).map_err(|e: ErrMode<ContextError>| {
            Report::new(MergedModelsError::ParseError(format!("space relptrs: {e}")))
        })?;
    let mut out = [(0u32, 0usize); 8];
    for i in 0..8 {
        // Absolute-from-file-start, NOT base+offset. See doc above.
        out[i] = (counts[i], relptrs[i].max(0) as usize);
    }
    Ok(out)
}

/// Parse a `space.bin` file to extract the typed instance sub-arrays.
///
/// The engine's `SpaceContent::Instances` block carries eight typed
/// sub-arrays; we consume seven: models, obstacles, particles,
/// pointLights, probes, staticDecals, and userObjects. Only `prefabs[]`
/// is skipped (dead data in the shipped game — see the audit doc).
pub fn parse_space_instances(file_data: &[u8]) -> Result<SpaceInstances, Report<MergedModelsError>> {
    let header = parse_space_header(file_data)?;
    let (instance_count, instances_offset) = header[SUBARRAY_MODELS];
    let (obstacle_count, obstacles_offset) = header[SUBARRAY_OBSTACLES];
    let (particle_count, particles_offset) = header[SUBARRAY_PARTICLES];
    let (light_count, lights_offset) = header[SUBARRAY_POINT_LIGHTS];
    let (probe_count, probes_offset) = header[SUBARRAY_PROBES];
    let (static_decal_count, static_decals_offset) = header[SUBARRAY_STATIC_DECALS];
    let (user_object_count, user_objects_offset) = header[SUBARRAY_USER_OBJECTS];

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
            let rec = parse_space_instance_record(file_data, rec_base).map_err(|e: ErrMode<ContextError>| {
                Report::new(MergedModelsError::ParseError(format!("space instance[{i}] @ 0x{rec_base:x}: {e}")))
            })?;
            out.push(rec);
        }
        out
    };

    let obstacles = parse_space_record_array(
        file_data,
        obstacles_offset,
        obstacle_count as usize,
        SPACE_OBSTACLE_SIZE,
        parse_space_obstacle_record,
        "space obstacles",
    )?;

    let particles = parse_space_record_array(
        file_data,
        particles_offset,
        particle_count as usize,
        SPACE_PARTICLE_SIZE,
        parse_space_particle_record,
        "space particles",
    )?;

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
        winnow::combinator::repeat(light_count as usize, parse_space_point_light_entry).parse_next(input).map_err(
            |e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("space point lights: {e}"))),
        )?
    };

    let probes = parse_space_record_array(
        file_data,
        probes_offset,
        probe_count as usize,
        SPACE_PROBE_SIZE,
        parse_space_probe_record,
        "space probes",
    )?;

    let static_decals = parse_space_record_array(
        file_data,
        static_decals_offset,
        static_decal_count as usize,
        SPACE_STATIC_DECAL_SIZE,
        parse_space_static_decal_record,
        "space static decals",
    )?;

    let user_objects = parse_space_record_array(
        file_data,
        user_objects_offset,
        user_object_count as usize,
        SPACE_USER_OBJECT_SIZE,
        parse_space_user_object_record,
        "space user objects",
    )?;

    Ok(SpaceInstances {
        instances,
        obstacles,
        particles,
        point_lights,
        probes,
        static_decals,
        user_objects,
    })
}

fn parse_space_record_array<T>(
    file_data: &[u8],
    offset: usize,
    count: usize,
    stride: usize,
    parser: fn(&[u8], usize) -> Result<T, Report<MergedModelsError>>,
    label: &str,
) -> Result<Vec<T>, Report<MergedModelsError>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let need = count * stride;
    if offset + need > file_data.len() {
        return Err(Report::new(MergedModelsError::DataTooShort { offset, need, have: file_data.len() }));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let rec_base = offset + i * stride;
        let rec = parser(file_data, rec_base)
            .map_err(|e| MergedModelsError::ParseError(format!("{label}[{i}] @ 0x{rec_base:x}: {e}")))?;
        out.push(rec);
    }
    Ok(out)
}

fn read_string_descriptor(file_data: &[u8], descriptor_base: usize, len: u64, relptr: i64) -> Option<String> {
    if len == 0 {
        return None;
    }
    let start_i64 = descriptor_base as i64 + relptr;
    if start_i64 < 0 {
        return None;
    }
    let start = start_i64 as usize;
    let end = start.checked_add(len as usize)?;
    let bytes = file_data.get(start..end)?;
    let bytes = bytes.strip_suffix(&[0]).unwrap_or(bytes);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn matrix_position(transform: &Matrix4x4) -> [f32; 3] {
    [transform.0[12], transform.0[13], transform.0[14]]
}

fn parse_space_obstacle_record(
    file_data: &[u8],
    rec_base: usize,
) -> Result<SpaceObstacle, Report<MergedModelsError>> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_OBSTACLE_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("matrix: {e}"))))?;
    let packed_indices = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("packed: {e}"))))?;
    let field_44 = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("field_44: {e}"))))?;
    let min_x = le_i32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("grid min x: {e}"))))?;
    let min_y = le_i32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("grid min y: {e}"))))?;
    let max_x = le_i32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("grid max x: {e}"))))?;
    let max_y = le_i32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("grid max y: {e}"))))?;

    Ok(SpaceObstacle {
        source_offset: rec_base,
        position: matrix_position(&transform),
        transform,
        packed_indices,
        candidate_model_instance_index: (packed_indices & 0xffff) as u16,
        candidate_collision_model_index: (packed_indices >> 16) as u16,
        field_44,
        grid_min: [min_x, min_y],
        grid_max: [max_x, max_y],
    })
}

fn parse_space_particle_record(
    file_data: &[u8],
    rec_base: usize,
) -> Result<SpaceParticle, Report<MergedModelsError>> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_PARTICLE_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("matrix: {e}"))))?;
    let guid_bytes: &[u8] = take(16usize)
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("guid blob: {e}"))))?;
    let mut raw_guid_blob = [0u8; 16];
    raw_guid_blob.copy_from_slice(guid_bytes);
    let resource_id = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("resource id: {e}"))))?;
    let intensity_count = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("intensity count: {e}"))))?;
    let _pad = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("intensity pad: {e}"))))?;
    let intensity_relptr = le_i64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("intensity relptr: {e}"))))?;

    let mut intensity_values = Vec::with_capacity(intensity_count as usize);
    if intensity_count > 0 {
        let start_i64 = rec_base as i64 + intensity_relptr;
        if start_i64 >= 0 {
            let start = start_i64 as usize;
            let need = intensity_count as usize * 4;
            if start + need <= file_data.len() {
                let mut value_input = &file_data[start..start + need];
                for _ in 0..intensity_count {
                    intensity_values.push(le_f32.parse_next(&mut value_input).map_err(|e: ErrMode<ContextError>| {
                        Report::new(MergedModelsError::ParseError(format!("intensity value: {e}")))
                    })?);
                }
            }
        }
    }

    Ok(SpaceParticle {
        position: matrix_position(&transform),
        transform,
        raw_guid_blob,
        resource_id,
        intensity_count,
        intensity_values,
    })
}

fn parse_space_probe_record(file_data: &[u8], rec_base: usize) -> Result<SpaceProbe, Report<MergedModelsError>> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_PROBE_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("matrix: {e}"))))?;
    let guid_len = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("guid len: {e}"))))?;
    let guid_relptr = le_i64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("guid relptr: {e}"))))?;
    let name_len = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("name len: {e}"))))?;
    let name_relptr = le_i64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("name relptr: {e}"))))?;
    let resolution = le_u32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("resolution: {e}"))))?;
    let _pad = take(4usize)
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("probe pad: {e}"))))?;
    let is_main_probe = le_u8
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("main probe: {e}"))))?
        != 0;
    let draw_full_scene = le_u8
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("draw full scene: {e}"))))?
        != 0;

    Ok(SpaceProbe {
        position: matrix_position(&transform),
        transform,
        guid: read_string_descriptor(file_data, rec_base + 0x40, guid_len, guid_relptr),
        name: read_string_descriptor(file_data, rec_base + 0x50, name_len, name_relptr),
        resolution,
        is_main_probe,
        draw_full_scene,
    })
}

fn parse_space_static_decal_record(
    file_data: &[u8],
    rec_base: usize,
) -> Result<SpaceStaticDecal, Report<MergedModelsError>> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_STATIC_DECAL_SIZE];
    let technique = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("technique: {e}")))
    })?;
    let influence = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("influence: {e}")))
    })?;
    let field_08 = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("field_08: {e}")))
    })?;
    let field_0c = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("field_0c: {e}")))
    })?;
    let variant = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("variant: {e}")))
    })?;
    let alpha = le_f32
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("alpha: {e}"))))?;
    let field_18 = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("field_18: {e}")))
    })?;
    let field_1c = le_u32.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("field_1c: {e}")))
    })?;
    let texture_block_relptr = le_i64.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("texture block: {e}")))
    })?;
    let transform = parser_utils::parse_matrix4x4(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("matrix: {e}"))))?;

    let texture_paths = read_static_decal_texture_paths(file_data, rec_base, texture_block_relptr);

    Ok(SpaceStaticDecal {
        position: matrix_position(&transform),
        transform,
        header: SpaceStaticDecalHeader { technique, influence, field_08, field_0c, variant, alpha, field_18, field_1c },
        texture_paths,
    })
}

fn read_static_decal_texture_paths(file_data: &[u8], rec_base: usize, texture_block_relptr: i64) -> Vec<Option<String>> {
    let block_i64 = rec_base as i64 + texture_block_relptr;
    if block_i64 < 0 {
        return vec![None, None, None];
    }
    let block = block_i64 as usize;
    let mut paths = Vec::with_capacity(3);
    for i in 0..3 {
        let descriptor_base = block + i * 0x10;
        if descriptor_base + 0x10 > file_data.len() {
            paths.push(None);
            continue;
        }
        let len = u64::from_le_bytes(file_data[descriptor_base..descriptor_base + 8].try_into().unwrap());
        let relptr = i64::from_le_bytes(file_data[descriptor_base + 8..descriptor_base + 0x10].try_into().unwrap());
        paths.push(read_string_descriptor(file_data, descriptor_base, len, relptr));
    }
    paths
}

fn parse_space_user_object_record(
    file_data: &[u8],
    rec_base: usize,
) -> Result<SpaceUserObject, Report<MergedModelsError>> {
    let input = &mut &file_data[rec_base..rec_base + SPACE_USER_OBJECT_SIZE];
    let transform = parser_utils::parse_matrix4x4(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("matrix: {e}"))))?;
    let guid_len = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("guid len: {e}"))))?;
    let guid_relptr = le_i64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("guid relptr: {e}"))))?;
    let type_len = le_u64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("type len: {e}"))))?;
    let type_relptr = le_i64
        .parse_next(input)
        .map_err(|e: ErrMode<ContextError>| Report::new(MergedModelsError::ParseError(format!("type relptr: {e}"))))?;
    let properties_len = le_u64.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("properties len: {e}")))
    })?;
    let properties_relptr = le_i64.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("properties relptr: {e}")))
    })?;

    let properties_xml = read_string_descriptor(file_data, rec_base + 0x60, properties_len, properties_relptr);
    let (properties_well_formed, property_tags, property_values) = properties_xml
        .as_deref()
        .map(parse_user_object_properties)
        .unwrap_or((false, Vec::new(), Vec::new()));

    Ok(SpaceUserObject {
        position: matrix_position(&transform),
        transform,
        guid: read_string_descriptor(file_data, rec_base + 0x40, guid_len, guid_relptr),
        object_type: read_string_descriptor(file_data, rec_base + 0x50, type_len, type_relptr),
        properties_xml,
        properties_well_formed,
        property_tags,
        property_values,
    })
}

fn parse_user_object_properties(xml: &str) -> (bool, Vec<String>, Vec<SpaceUserObjectPropertyValue>) {
    let Ok(doc) = roxmltree::Document::parse(xml) else {
        return (false, Vec::new(), Vec::new());
    };
    let root = doc.root_element();
    let mut tags = Vec::new();
    let mut values = Vec::new();
    let mut path = Vec::new();
    for child in root.children().filter(|node| node.is_element()) {
        walk_user_object_xml(child, &mut path, &mut tags, &mut values);
    }
    (true, tags, values)
}

fn walk_user_object_xml(
    node: roxmltree::Node<'_, '_>,
    path: &mut Vec<String>,
    tags: &mut Vec<String>,
    values: &mut Vec<SpaceUserObjectPropertyValue>,
) {
    let tag = node.tag_name().name().to_string();
    if !tags.contains(&tag) {
        tags.push(tag.clone());
    }
    path.push(tag);
    let mut has_element_child = false;
    for child in node.children().filter(|child| child.is_element()) {
        has_element_child = true;
        walk_user_object_xml(child, path, tags, values);
    }
    if !has_element_child {
        if let Some(text) = node.text().map(str::trim).filter(|text| !text.is_empty()) {
            values.push(SpaceUserObjectPropertyValue { path: path.join("."), value: text.to_string() });
        }
    }
    path.pop();
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
// them. Map-rendering downstream doesn't read skel_ext / animations, so those
// stay empty.
//
// Dyes ARE parsed: the inline dye representation is byte-identical to the
// assets.bin DyeEntry layout (0x20 header: u32 matter, u32 replaces,
// u32 tints_count, pad4, i64 tint_names relptr, i64 tint_materials relptr;
// array relptr based at the proto base, per-entry relptrs at the entry base).
// Verified against 01_solomon_islands / 20_NE_two_brothers / 54_Faroe:
// every space.bin instance dye pair {matter_id, tint_name_id} joins a
// prototype DyeEntry (matter match) with the pair's second u32 matching one
// of that dye's tint_name_ids.
fn parse_inline_model_proto_header(data: &[u8], base: usize) -> Result<ModelPrototype, Report<MergedModelsError>> {
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
    let _animations_count = le_u8.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp animations_count: {e}")))
    })?;
    let dyes_count = le_u8.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp dyes_count: {e}")))
    })?;
    // padding(4), skel_ext relptr, animations relptr — read & discard
    let _ = take(4usize + 16).parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp tail: {e}")))
    })?;
    let dye_relptr = le_i64.parse_next(input).map_err(|e: ErrMode<ContextError>| {
        Report::new(MergedModelsError::ParseError(format!("inline_mp dye_relptr: {e}")))
    })?;
    let dyes = if dyes_count > 0 {
        let abs = resolve_relptr(base, dye_relptr);
        crate::models::model::parse_dye_entries(data, abs, dyes_count as usize)
            .map_err(|e| Report::new(MergedModelsError::ParseError(format!("inline_mp dyes: {e}"))))?
    } else {
        Vec::new()
    };
    Ok(ModelPrototype {
        visual_resource_id,
        misc_type,
        skel_ext_res_ids: Vec::new(),
        animations: Vec::new(),
        dyes,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::material::{PropertyType, PropertyValue};

    #[test]
    fn parses_inline_model_proto_dyes() {
        // Synthetic models.bin: header 0x18 + one 0xA8 model record. The
        // inline proto's dye array uses the assets.bin DyeEntry layout with
        // the array relptr based at the proto base (byte-verified on
        // 01_solomon_islands / 20_NE_two_brothers / 54_Faroe).
        let models_off = 0x18usize;
        let rec = models_off;
        let mp = rec + 0x08;
        let dye_off = models_off + MODEL_RECORD_SIZE;
        let names_off = dye_off + 0x20;
        let mats_off = names_off + 0x08;

        let mut data = vec![0u8; mats_off + 0x10];
        put_u32(&mut data, 0x00, 1); // models_count
        put_i64(&mut data, 0x08, models_off as i64);
        put_i64(&mut data, 0x10, (mats_off + 0x10) as i64); // skeletons (count 0)

        put_u64(&mut data, rec, 0xdead_beef_cafe_f00d); // path_id
        data[mp + 0x0B] = 1; // dyes_count
        put_i64(&mut data, mp + 0x20, (dye_off - mp) as i64); // dye relptr, base = proto

        put_u32(&mut data, dye_off, 0x46f1_b231); // matter
        put_u32(&mut data, dye_off + 0x04, 0x46f1_b231); // replaces
        put_u32(&mut data, dye_off + 0x08, 2); // tints_count
        put_i64(&mut data, dye_off + 0x10, (names_off - dye_off) as i64);
        put_i64(&mut data, dye_off + 0x18, (mats_off - dye_off) as i64);
        put_u32(&mut data, names_off, 0x955d_be29);
        put_u32(&mut data, names_off + 4, 0x0e6c_e501);
        put_u64(&mut data, mats_off, 0xc042_cbb3_aa90_21e2);
        put_u64(&mut data, mats_off + 8, 0x1111_2222_3333_4444);

        let parsed = parse_merged_models(&data).expect("merged models");
        let dyes = &parsed.models[0].model_proto.dyes;
        assert_eq!(dyes.len(), 1);
        assert_eq!(dyes[0].matter_id, 0x46f1_b231);
        assert_eq!(dyes[0].replaces_id, 0x46f1_b231);
        assert_eq!(dyes[0].tint_name_ids, [0x955d_be29, 0x0e6c_e501]);
        assert_eq!(dyes[0].tint_material_ids, [0xc042_cbb3_aa90_21e2, 0x1111_2222_3333_4444]);
    }

    fn put_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(data: &mut [u8], offset: usize, value: i32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i64(data: &mut [u8], offset: usize, value: i64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_f32(data: &mut [u8], offset: usize, value: f32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn parses_model_dyes_relative_to_model_instance_base() {
        let model_offset = SPACE_HEADER_SIZE;
        let rec1 = model_offset + SPACE_INSTANCE_SIZE;
        let dye_offset = 0x160usize;
        let old_wrong_offset = dye_offset + 0x60;

        let mut data = vec![0u8; old_wrong_offset + 8];
        put_u32(&mut data, 0x00, 2);
        put_i64(&mut data, 0x20, model_offset as i64);

        put_u64(&mut data, rec1 + 0x50, 0x1234_5678_90ab_cdef);
        data[rec1 + 0x59] = 1;
        data[rec1 + 0x5b] = 4;
        put_i64(&mut data, rec1 + 0x60, dye_offset as i64 - rec1 as i64);

        put_u32(&mut data, dye_offset, 0x46f1_b231);
        put_u32(&mut data, dye_offset + 4, 0x955d_be29);
        put_u32(&mut data, old_wrong_offset, 0x1111_1111);
        put_u32(&mut data, old_wrong_offset + 4, 0x2222_2222);

        let parsed = parse_space_instances(&data).expect("space instances");
        let dye = parsed.instances[1].model_dyes[0];
        assert_eq!(dye.matter_id, 0x46f1_b231);
        assert_eq!(dye.replaces_id, 0x955d_be29);
    }

    #[test]
    fn parses_stable_guid_descriptor_relative_to_descriptor_base() {
        let model_offset = SPACE_HEADER_SIZE;
        let rec0 = model_offset;
        let guid_offset = 0x100usize;
        let guid = b"2E8F86C7.4B8A6236.75A3978F.4D162DE3\0";

        let mut data = vec![0u8; guid_offset + guid.len()];
        put_u32(&mut data, 0x00, 1);
        put_i64(&mut data, 0x20, model_offset as i64);

        put_u64(&mut data, rec0 + 0x40, guid.len() as u64);
        put_i64(&mut data, rec0 + 0x48, guid_offset as i64 - (rec0 + 0x40) as i64);
        put_u64(&mut data, rec0 + 0x50, 0x1234_5678_90ab_cdef);
        data[guid_offset..guid_offset + guid.len()].copy_from_slice(guid);

        let parsed = parse_space_instances(&data).expect("space instances");
        assert_eq!(
            parsed.instances[0].stable_guid.as_deref(),
            Some("2E8F86C7.4B8A6236.75A3978F.4D162DE3")
        );
    }

    #[test]
    fn parses_material_instances_relative_to_model_instance_base() {
        let model_offset = SPACE_HEADER_SIZE;
        let rec1 = model_offset + SPACE_INSTANCE_SIZE;
        let material_offset = 0x180usize;

        let mut data = vec![0u8; 0x210];
        put_u32(&mut data, 0x00, 2);
        put_i64(&mut data, 0x20, model_offset as i64);

        put_u64(&mut data, rec1 + 0x50, 0x1234_5678_90ab_cdef);
        data[rec1 + 0x5a] = 1;
        data[rec1 + 0x5b] = 4;
        put_i64(&mut data, rec1 + 0x68, material_offset as i64 - rec1 as i64);

        put_u16(&mut data, material_offset, 1);
        put_u32(&mut data, material_offset + 0x04, 0x700);
        put_u64(&mut data, material_offset + 0x08, 0x700);
        put_u64(&mut data, material_offset + 0x10, 0x70);
        put_u64(&mut data, material_offset + 0x18, 0x74);
        put_u64(&mut data, material_offset + 0x38, 0x78);
        put_u64(&mut data, material_offset + 0x68, 0x6a79c245);
        put_u32(&mut data, material_offset + 0x70, 0x5acc9c8b);
        put_u16(&mut data, material_offset + 0x74, 3);
        put_f32(&mut data, material_offset + 0x78, 0.5);

        let parsed = parse_space_instances(&data).expect("space instances");
        let material = &parsed.instances[1].material_instances[0];
        assert_eq!(material.shader_id, 0x700);
        assert_eq!(material.material_hash, 0x6a79c245);
        assert_eq!(material.properties[0].property_type, PropertyType::FloatB);
        assert!(
            matches!(material.properties[0].value, Some(PropertyValue::Float(v)) if (v - 0.5).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn parses_obstacle_packed_indices_and_grid_bounds() {
        let obstacle_offset = SPACE_HEADER_SIZE;
        let rec = obstacle_offset;
        let mut data = vec![0u8; obstacle_offset + SPACE_OBSTACLE_SIZE];

        put_u32(&mut data, 0x04, 1);
        put_i64(&mut data, 0x28, obstacle_offset as i64);

        put_f32(&mut data, rec, 1.0);
        put_f32(&mut data, rec + 0x14, 1.0);
        put_f32(&mut data, rec + 0x28, 1.0);
        put_f32(&mut data, rec + 0x3c, 1.0);
        put_f32(&mut data, rec + 0x30, -629.387);
        put_f32(&mut data, rec + 0x34, -8.695);
        put_f32(&mut data, rec + 0x38, -599.672);
        let packed = (5u32 << 16) | 4u32;
        put_u32(&mut data, rec + 0x40, packed);
        put_i32(&mut data, rec + 0x48, -8);
        put_i32(&mut data, rec + 0x4c, -7);
        put_i32(&mut data, rec + 0x50, -6);
        put_i32(&mut data, rec + 0x54, -6);

        let parsed = parse_space_instances(&data).expect("space instances");
        let obstacle = &parsed.obstacles[0];
        assert_eq!(obstacle.source_offset, rec);
        assert_eq!(obstacle.position, [-629.387, -8.695, -599.672]);
        assert_eq!(obstacle.packed_indices, packed);
        assert_eq!(obstacle.candidate_model_instance_index, 4);
        assert_eq!(obstacle.candidate_collision_model_index, 5);
        assert_eq!(obstacle.grid_min, [-8, -7]);
        assert_eq!(obstacle.grid_max, [-6, -6]);
    }

    #[test]
    fn parses_particle_intensities_relative_to_record_base() {
        let particle_offset = SPACE_HEADER_SIZE;
        let rec = particle_offset;
        let intensity_offset = particle_offset + SPACE_PARTICLE_SIZE;
        let old_wrong_offset = intensity_offset + 0x60;
        let mut data = vec![0u8; old_wrong_offset + 24];

        put_u32(&mut data, 0x08, 1);
        put_i64(&mut data, 0x30, particle_offset as i64);
        put_f32(&mut data, rec, 1.0);
        put_f32(&mut data, rec + 0x14, 1.0);
        put_f32(&mut data, rec + 0x28, 1.0);
        put_f32(&mut data, rec + 0x3c, 1.0);
        put_f32(&mut data, rec + 0x30, -741.305);
        put_f32(&mut data, rec + 0x34, 13.294);
        put_f32(&mut data, rec + 0x38, -172.677);
        data[rec + 0x40..rec + 0x50].copy_from_slice(&[0xAB; 16]);
        put_u64(&mut data, rec + 0x50, 0x1234_5678_90ab_cdef);
        put_u32(&mut data, rec + 0x58, 6);
        put_i64(&mut data, rec + 0x60, intensity_offset as i64 - rec as i64);

        let values = [0.25, 0.8, 0.9, 1.5, 1.0, 1.0];
        for (idx, value) in values.iter().enumerate() {
            put_f32(&mut data, intensity_offset + idx * 4, *value);
            put_f32(&mut data, old_wrong_offset + idx * 4, 99.0);
        }

        let parsed = parse_space_instances(&data).expect("space instances");
        let particle = &parsed.particles[0];
        assert_eq!(particle.position, [-741.305, 13.294, -172.677]);
        assert_eq!(particle.raw_guid_blob, [0xAB; 16]);
        assert_eq!(particle.resource_id, 0x1234_5678_90ab_cdef);
        assert_eq!(particle.intensity_values, values);
    }

    #[test]
    fn parses_particle_with_zero_intensity_count() {
        let particle_offset = SPACE_HEADER_SIZE;
        let rec = particle_offset;
        let mut data = vec![0u8; particle_offset + SPACE_PARTICLE_SIZE];

        put_u32(&mut data, 0x08, 1);
        put_i64(&mut data, 0x30, particle_offset as i64);
        put_u64(&mut data, rec + 0x50, 0x0fed_cba9_8765_4321);

        let parsed = parse_space_instances(&data).expect("space instances");
        let particle = &parsed.particles[0];
        assert_eq!(particle.resource_id, 0x0fed_cba9_8765_4321);
        assert_eq!(particle.intensity_count, 0);
        assert!(particle.intensity_values.is_empty());
    }

    #[test]
    fn parses_static_decal_texture_block_and_descriptors() {
        let decal_offset = SPACE_HEADER_SIZE;
        let rec = decal_offset;
        let texture_block_offset = 0x120usize;
        let tex0_offset = 0x180usize;
        let tex1_offset = 0x1d0usize;
        let tex2_offset = 0x220usize;
        let tex0 = b"maps/decals/01_Solomon/spot_alpha_a.tga\0";
        let tex1 = b"maps/decals/01_Solomon/spot_n.tga\0";
        let tex2 = b"maps/decals/01_Solomon/spot_mg.tga\0";

        let mut data = vec![0u8; tex2_offset + tex2.len()];
        put_u32(&mut data, 0x14, 1);
        put_i64(&mut data, 0x48, decal_offset as i64);
        put_u32(&mut data, rec, 3);
        put_u32(&mut data, rec + 0x04, 1);
        put_u32(&mut data, rec + 0x08, 4);
        put_u32(&mut data, rec + 0x10, 5);
        put_f32(&mut data, rec + 0x14, 0.31);
        put_u32(&mut data, rec + 0x18, 1);
        put_i64(&mut data, rec + 0x20, texture_block_offset as i64 - rec as i64);
        put_f32(&mut data, rec + 0x28, 1.0);
        put_f32(&mut data, rec + 0x3c, 1.0);
        put_f32(&mut data, rec + 0x50, 1.0);
        put_f32(&mut data, rec + 0x64, 1.0);
        put_f32(&mut data, rec + 0x58, -231.926);
        put_f32(&mut data, rec + 0x5c, 0.188);
        put_f32(&mut data, rec + 0x60, -66.611);

        put_u64(&mut data, texture_block_offset, tex0.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x08, tex0_offset as i64 - texture_block_offset as i64);
        put_u64(&mut data, texture_block_offset + 0x10, tex1.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x18, tex1_offset as i64 - (texture_block_offset + 0x10) as i64);
        put_u64(&mut data, texture_block_offset + 0x20, tex2.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x28, tex2_offset as i64 - (texture_block_offset + 0x20) as i64);
        data[tex0_offset..tex0_offset + tex0.len()].copy_from_slice(tex0);
        data[tex1_offset..tex1_offset + tex1.len()].copy_from_slice(tex1);
        data[tex2_offset..tex2_offset + tex2.len()].copy_from_slice(tex2);

        let parsed = parse_space_instances(&data).expect("space instances");
        let decal = &parsed.static_decals[0];
        assert_eq!(decal.header.technique, 3);
        assert_eq!(decal.header.variant, 5);
        assert!((decal.header.alpha - 0.31).abs() < f32::EPSILON);
        assert_eq!(decal.position, [-231.926, 0.188, -66.611]);
        assert_eq!(decal.texture_paths[0].as_deref(), Some("maps/decals/01_Solomon/spot_alpha_a.tga"));
        assert_eq!(decal.texture_paths[1].as_deref(), Some("maps/decals/01_Solomon/spot_n.tga"));
        assert_eq!(decal.texture_paths[2].as_deref(), Some("maps/decals/01_Solomon/spot_mg.tga"));
    }

    #[test]
    fn preserves_blank_static_decal_third_texture_slot() {
        let decal_offset = SPACE_HEADER_SIZE;
        let rec = decal_offset;
        let texture_block_offset = 0x100usize;
        let tex0_offset = 0x130usize;
        let tex1_offset = 0x150usize;
        let tex2_offset = 0x170usize;
        let tex0 = b"maps/decals/Dock/a.dds\0";
        let tex1 = b"maps/decals/Dock/n.dds\0";
        let tex2 = b"\0";

        let mut data = vec![0u8; tex2_offset + tex2.len()];
        put_u32(&mut data, 0x14, 1);
        put_i64(&mut data, 0x48, decal_offset as i64);
        put_i64(&mut data, rec + 0x20, texture_block_offset as i64 - rec as i64);

        put_u64(&mut data, texture_block_offset, tex0.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x08, tex0_offset as i64 - texture_block_offset as i64);
        put_u64(&mut data, texture_block_offset + 0x10, tex1.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x18, tex1_offset as i64 - (texture_block_offset + 0x10) as i64);
        put_u64(&mut data, texture_block_offset + 0x20, tex2.len() as u64);
        put_i64(&mut data, texture_block_offset + 0x28, tex2_offset as i64 - (texture_block_offset + 0x20) as i64);
        data[tex0_offset..tex0_offset + tex0.len()].copy_from_slice(tex0);
        data[tex1_offset..tex1_offset + tex1.len()].copy_from_slice(tex1);
        data[tex2_offset..tex2_offset + tex2.len()].copy_from_slice(tex2);

        let parsed = parse_space_instances(&data).expect("space instances");
        let decal = &parsed.static_decals[0];
        assert_eq!(decal.texture_paths[0].as_deref(), Some("maps/decals/Dock/a.dds"));
        assert_eq!(decal.texture_paths[1].as_deref(), Some("maps/decals/Dock/n.dds"));
        assert_eq!(decal.texture_paths[2].as_deref(), Some(""));
    }

    #[test]
    fn parses_space_probe_record_from_probe_subarray() {
        let probe_offset = SPACE_HEADER_SIZE;
        let rec = probe_offset;
        let guid_offset = 0x100usize;
        let name_offset = 0x128usize;
        let guid = b"FFE10835.4B0F3F54.96C54EAD.82303DAF\0";
        let name = b"main_probe\0";

        let mut data = vec![0u8; name_offset + name.len()];
        put_u32(&mut data, 0x10, 1);
        put_i64(&mut data, 0x40, probe_offset as i64);
        put_f32(&mut data, rec, 1.0);
        put_f32(&mut data, rec + 0x14, 1.0);
        put_f32(&mut data, rec + 0x28, 1.0);
        put_f32(&mut data, rec + 0x3c, 1.0);
        put_f32(&mut data, rec + 0x34, 0.5);
        put_u64(&mut data, rec + 0x40, guid.len() as u64);
        put_i64(&mut data, rec + 0x48, guid_offset as i64 - (rec + 0x40) as i64);
        put_u64(&mut data, rec + 0x50, name.len() as u64);
        put_i64(&mut data, rec + 0x58, name_offset as i64 - (rec + 0x50) as i64);
        put_u32(&mut data, rec + 0x60, 512);
        data[rec + 0x68] = 1;
        data[rec + 0x69] = 1;
        data[guid_offset..guid_offset + guid.len()].copy_from_slice(guid);
        data[name_offset..name_offset + name.len()].copy_from_slice(name);

        let parsed = parse_space_instances(&data).expect("space instances");
        let probe = &parsed.probes[0];
        assert_eq!(probe.guid.as_deref(), Some("FFE10835.4B0F3F54.96C54EAD.82303DAF"));
        assert_eq!(probe.name.as_deref(), Some("main_probe"));
        assert_eq!(probe.resolution, 512);
        assert!(probe.is_main_probe);
        assert!(probe.draw_full_scene);
        assert_eq!(probe.position, [0.0, 0.5, 0.0]);
    }

    #[test]
    fn parses_user_object_record_from_user_objects_subarray() {
        let user_object_offset = SPACE_HEADER_SIZE;
        let rec = user_object_offset;
        let guid_offset = 0x100usize;
        let type_offset = 0x128usize;
        let properties_offset = 0x140usize;
        let guid = b"D03F81A8.4506DA63.2999C39B.9EB68E8D\0";
        let object_type = b"WayPoint\0";
        let properties = b"<properties><next><item><guid>89E297B4.4A9DF3B8.880451BD.70EA35B6</guid><chunkId>fffaffffo</chunkId></item></next><speed>12.5</speed></properties>\0";

        let mut data = vec![0u8; properties_offset + properties.len()];
        put_u32(&mut data, 0x18, 1);
        put_i64(&mut data, 0x50, user_object_offset as i64);
        put_f32(&mut data, rec, 1.0);
        put_f32(&mut data, rec + 0x14, 1.0);
        put_f32(&mut data, rec + 0x28, 1.0);
        put_f32(&mut data, rec + 0x3c, 1.0);
        put_f32(&mut data, rec + 0x30, 10.0);
        put_f32(&mut data, rec + 0x34, 2.0);
        put_f32(&mut data, rec + 0x38, -5.0);
        put_u64(&mut data, rec + 0x40, guid.len() as u64);
        put_i64(&mut data, rec + 0x48, guid_offset as i64 - (rec + 0x40) as i64);
        put_u64(&mut data, rec + 0x50, object_type.len() as u64);
        put_i64(&mut data, rec + 0x58, type_offset as i64 - (rec + 0x50) as i64);
        put_u64(&mut data, rec + 0x60, properties.len() as u64);
        put_i64(&mut data, rec + 0x68, properties_offset as i64 - (rec + 0x60) as i64);
        data[guid_offset..guid_offset + guid.len()].copy_from_slice(guid);
        data[type_offset..type_offset + object_type.len()].copy_from_slice(object_type);
        data[properties_offset..properties_offset + properties.len()].copy_from_slice(properties);

        let parsed = parse_space_instances(&data).expect("space instances");
        let object = &parsed.user_objects[0];
        assert_eq!(object.guid.as_deref(), Some("D03F81A8.4506DA63.2999C39B.9EB68E8D"));
        assert_eq!(object.object_type.as_deref(), Some("WayPoint"));
        assert_eq!(object.position, [10.0, 2.0, -5.0]);
        assert!(object.properties_well_formed);
        assert_eq!(object.property_tags, ["next", "item", "guid", "chunkId", "speed"]);
        assert_eq!(object.property_values.len(), 3);
        assert_eq!(object.property_values[0].path, "next.item.guid");
        assert_eq!(object.property_values[1].path, "next.item.chunkId");
        assert_eq!(object.property_values[2].path, "speed");
    }

    #[test]
    fn preserves_malformed_user_object_properties_as_raw_text() {
        let user_object_offset = SPACE_HEADER_SIZE;
        let rec = user_object_offset;
        let type_offset = 0x100usize;
        let properties_offset = 0x110usize;
        let object_type = b"Barge\0";
        let properties = b"<properties><bad></properties>\0";

        let mut data = vec![0u8; properties_offset + properties.len()];
        put_u32(&mut data, 0x18, 1);
        put_i64(&mut data, 0x50, user_object_offset as i64);
        put_u64(&mut data, rec + 0x50, object_type.len() as u64);
        put_i64(&mut data, rec + 0x58, type_offset as i64 - (rec + 0x50) as i64);
        put_u64(&mut data, rec + 0x60, properties.len() as u64);
        put_i64(&mut data, rec + 0x68, properties_offset as i64 - (rec + 0x60) as i64);
        data[type_offset..type_offset + object_type.len()].copy_from_slice(object_type);
        data[properties_offset..properties_offset + properties.len()].copy_from_slice(properties);

        let parsed = parse_space_instances(&data).expect("space instances");
        let object = &parsed.user_objects[0];
        assert_eq!(object.object_type.as_deref(), Some("Barge"));
        assert_eq!(object.properties_xml.as_deref(), Some("<properties><bad></properties>"));
        assert!(!object.properties_well_formed);
        assert!(object.property_tags.is_empty());
        assert!(object.property_values.is_empty());
    }
}
