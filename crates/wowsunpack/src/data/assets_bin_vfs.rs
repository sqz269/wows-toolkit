//! VFS abstraction for reading files from an assets.bin PrototypeDatabase.
//!
//! Exposes prototype records as virtual files, keyed by their reconstructed
//! path from the pathsStorage tree. Each file's contents are the raw prototype
//! record data from the record start through the end of the containing blob,
//! preserving relative pointer resolution into out-of-line data.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Cursor;

use vfs::FileSystem;
use vfs::VfsMetadata;
use vfs::error::VfsErrorKind;

use crate::models::assets_bin;
use crate::models::assets_bin::AssetsBinError;
use crate::models::assets_bin::PrototypeDatabase;

/// The type of prototype stored in a given assets.bin blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrototypeType {
    Material,         // blob 0
    Visual,           // blob 1
    SkeletonExtender, // blob 2
    Model,            // blob 3
    PointLight,       // blob 4
    Effect,           // blob 5
    VelocityField,    // blob 6
    EffectPreset,     // blob 7
    EffectMetadata,   // blob 8
    AtlasContour,     // blob 9
}

impl PrototypeType {
    /// Convert a blob index (0-9) to a `PrototypeType`.
    pub fn from_blob_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Material),
            1 => Some(Self::Visual),
            2 => Some(Self::SkeletonExtender),
            3 => Some(Self::Model),
            4 => Some(Self::PointLight),
            5 => Some(Self::Effect),
            6 => Some(Self::VelocityField),
            7 => Some(Self::EffectPreset),
            8 => Some(Self::EffectMetadata),
            9 => Some(Self::AtlasContour),
            _ => None,
        }
    }

    /// Infer prototype type from a file extension (e.g. `".visual"`, `".mfm"`).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            ".mfm" => Some(Self::Material),
            ".visual" => Some(Self::Visual),
            ".skeleton_extender" => Some(Self::SkeletonExtender),
            ".model" => Some(Self::Model),
            ".point_light" => Some(Self::PointLight),
            ".effect" => Some(Self::Effect),
            ".velocity_field" => Some(Self::VelocityField),
            ".effect_preset" => Some(Self::EffectPreset),
            ".effect_metadata" => Some(Self::EffectMetadata),
            ".atlas_contour" => Some(Self::AtlasContour),
            _ => None,
        }
    }

    /// The fixed record stride in bytes for this prototype type.
    pub fn item_size(self) -> usize {
        match self {
            Self::Material => 0x78,
            Self::Visual => 0x70,
            Self::SkeletonExtender => 0x20,
            Self::Model => 0x28,
            Self::PointLight => 0x70,
            Self::Effect => 0x10,
            Self::VelocityField => 0x18,
            Self::EffectPreset => 0x10,
            Self::EffectMetadata => 0x10,
            Self::AtlasContour => 0x10,
        }
    }
}

/// Pre-computed file location within the owned assets.bin data.
#[derive(Debug, Clone)]
struct FileLocation {
    /// Byte offset from start of `data` to the record.
    byte_offset: usize,
    /// Byte offset from start of `data` to end of the blob.
    byte_end: usize,
    /// The prototype type this record belongs to.
    prototype_type: PrototypeType,
    /// Index of the containing blob in `PrototypeDatabase::databases`.
    blob_index: usize,
    /// Index of the record within its blob.
    record_index: usize,
}

/// A virtual filesystem backed by an assets.bin PrototypeDatabase.
///
/// Owns the raw file data and exposes prototype records as virtual files.
/// Paths match the game's resource paths (e.g. `content/gameplay/.../foo.visual`).
#[derive(Debug)]
pub struct AssetsBinVfs {
    data: Vec<u8>,
    files: HashMap<String, FileLocation>,
    dirs: HashMap<String, Vec<String>>,
}

/// Compute the byte offset of a subslice within a parent slice.
fn subslice_offset(parent: &[u8], child: &[u8]) -> usize {
    let parent_start = parent.as_ptr() as usize;
    let child_start = child.as_ptr() as usize;
    debug_assert!(
        child_start >= parent_start && child_start + child.len() <= parent_start + parent.len(),
        "child slice is not within parent"
    );
    child_start - parent_start
}

/// Register a file path's directory ancestors in the directory map.
///
/// Paths use `/`-prefixed format (e.g. `/content/foo.visual`), root = `"/"`.
fn register_path_in_dirs(path: &str, dirs: &mut HashMap<String, BTreeSet<String>>) {
    let mut current = path.to_string();
    while let Some(pos) = current.rfind('/') {
        let child_name = &current[pos + 1..];
        let mut parent = current[..pos].to_string();
        if parent.is_empty() {
            parent = "/".to_string();
        }

        dirs.entry(parent.clone()).or_default().insert(child_name.to_string());

        if parent == "/" {
            break;
        }
        current = parent;
    }
}

impl AssetsBinVfs {
    /// Build a VFS from owned assets.bin file data.
    ///
    /// Parses the PrototypeDatabase, builds a path index mapping every
    /// prototype record to its byte range within `data`, then discards
    /// the parsed database. Only the raw bytes and the index are retained.
    pub fn new(data: Vec<u8>) -> Result<Self, rootcause::Report<AssetsBinError>> {
        let db = assets_bin::parse_assets_bin(&data)?;
        let (files, dirs) = Self::build_index(&db, &data);
        Ok(Self { data, files, dirs })
    }

    fn build_index(
        db: &PrototypeDatabase<'_>,
        data: &[u8],
    ) -> (HashMap<String, FileLocation>, HashMap<String, Vec<String>>) {
        let self_id_index = db.build_self_id_index();
        let mut files = HashMap::new();
        let mut dir_children: HashMap<String, BTreeSet<String>> = HashMap::new();

        // Ensure root directory exists.
        dir_children.entry("/".to_string()).or_default();

        // Register all paths that have prototype data as files.
        for (i, entry) in db.paths_storage.iter().enumerate() {
            let Some(r2p_value) = db.lookup_r2p(entry.self_id) else {
                continue;
            };
            let Ok(location) = db.decode_r2p_value(r2p_value) else {
                continue;
            };
            let Some(proto_type) = PrototypeType::from_blob_index(location.blob_index) else {
                continue;
            };

            let raw_path = db.reconstruct_path(i, &self_id_index);
            if raw_path.is_empty() {
                continue;
            }

            let item_size = proto_type.item_size();
            let blob = &db.databases[location.blob_index];

            let blob_start = subslice_offset(data, blob.data);
            let header_size = 16usize;
            let record_offset = blob_start + header_size + location.record_index * item_size;
            let blob_end = blob_start + blob.data.len();

            if record_offset + item_size > blob_end {
                continue;
            }

            let full_path = format!("/{raw_path}");
            files.insert(
                full_path.clone(),
                FileLocation {
                    byte_offset: record_offset,
                    byte_end: blob_end,
                    prototype_type: proto_type,
                    blob_index: location.blob_index,
                    record_index: location.record_index,
                },
            );

            register_path_in_dirs(&full_path, &mut dir_children);
        }

        // Register parent directories for path entries that have no prototype
        // data (e.g. .geometry files that live in PKG archives but appear in
        // pathsStorage). Only register parent dirs — NOT the leaf itself, which
        // would shadow PKG files in the overlay VFS.
        for (i, entry) in db.paths_storage.iter().enumerate() {
            if db.lookup_r2p(entry.self_id).is_some() {
                continue;
            }
            let raw_path = db.reconstruct_path(i, &self_id_index);
            if !raw_path.is_empty() {
                let full_path = format!("/{raw_path}");
                register_path_in_dirs(&full_path, &mut dir_children);
            }
        }

        let dirs = dir_children.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();

        (files, dirs)
    }

    /// Number of file entries in this VFS.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Number of directory entries in this VFS.
    pub fn dir_count(&self) -> usize {
        self.dirs.len()
    }

    /// Iterate over all file paths and their sizes (in bytes).
    pub fn files(&self) -> impl Iterator<Item = (&str, usize)> {
        self.files.iter().map(|(path, loc)| (path.as_str(), loc.byte_end - loc.byte_offset))
    }

    /// Iterate over all directory paths.
    pub fn dirs(&self) -> impl Iterator<Item = &str> {
        self.dirs.keys().map(|k| k.as_str())
    }

    /// Get the prototype type for a file path, if it exists.
    pub fn prototype_type(&self, path: &str) -> Option<PrototypeType> {
        let key = lookup_key(path);
        self.files.get(key).map(|loc| loc.prototype_type)
    }

    /// Iterate over all file paths with their sizes and prototype types.
    pub fn files_with_type(&self) -> impl Iterator<Item = (&str, usize, PrototypeType)> {
        self.files.iter().map(|(path, loc)| (path.as_str(), loc.byte_end - loc.byte_offset, loc.prototype_type))
    }

    /// Map every `(blob_index, record_index)` back to the path(s) that name it.
    ///
    /// Used to turn prototype reads recorded by [`crate::data::deps`] into
    /// paths. Several paths can share one record.
    pub fn prototype_paths(&self) -> HashMap<(usize, usize), Vec<String>> {
        let mut out: HashMap<(usize, usize), Vec<String>> = HashMap::new();
        for (path, loc) in &self.files {
            out.entry((loc.blob_index, loc.record_index)).or_default().push(path.clone());
        }
        for paths in out.values_mut() {
            paths.sort();
        }
        out
    }

    /// Content hash of the record at `path`, stable across game builds.
    ///
    /// A virtual file's bytes run from the record to the end of its blob, so
    /// they change whenever anything later in the blob changes. This hashes
    /// what the record actually holds instead. See [`record_content_hash`].
    pub fn content_hash(&self, path: &str) -> Option<RecordHash> {
        let loc = self.files.get(lookup_key(path))?;
        Some(record_content_hash(loc.prototype_type, &self.data[loc.byte_offset..loc.byte_end]))
    }
}

impl PrototypeType {
    /// Lower-case name used in JSON output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Material => "material",
            Self::Visual => "visual",
            Self::SkeletonExtender => "skeleton_extender",
            Self::Model => "model",
            Self::PointLight => "point_light",
            Self::Effect => "effect",
            Self::VelocityField => "velocity_field",
            Self::EffectPreset => "effect_preset",
            Self::EffectMetadata => "effect_metadata",
            Self::AtlasContour => "atlas_contour",
        }
    }
}

/// How a [`RecordHash`] was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashKind {
    /// Hash of the decoded record, including everything it points to.
    /// Unchanged content gives the same hash in every build.
    Decoded,
    /// Hash of the fixed-size record bytes only. The record's relative
    /// pointers move when earlier records grow or shrink, so this can report
    /// a change that didn't happen, but never misses one in the fixed part.
    /// Used for types the toolkit has no decoder for.
    Shallow,
}

impl HashKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Decoded => "decoded",
            Self::Shallow => "shallow",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHash {
    pub hash: u64,
    pub kind: HashKind,
}

impl RecordHash {
    /// 16-digit lower-case hex form used in JSON output.
    pub fn hex(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

/// 64-bit FNV-1a. Defined here so hashes never depend on the Rust version.
fn fnv1a64(chunks: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for chunk in chunks {
        for &b in *chunk {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Hash a prototype record so the result only changes when its content does.
///
/// `record_data` runs from the record start to the end of its blob, the same
/// slice the per-type parsers take.
///
/// - Visual and Material records are decoded and the decoded form is hashed,
///   so pointer offsets drop out.
/// - Model records hash their header, skeleton-extender ids, animation
///   entries and decoded dyes.
/// - SkeletonExtender records hash their three placement arrays.
/// - Every other type, and any record that fails to decode, falls back to
///   [`HashKind::Shallow`].
pub fn record_content_hash(proto_type: PrototypeType, record_data: &[u8]) -> RecordHash {
    if let Some(hash) = decoded_hash(proto_type, record_data) {
        return RecordHash { hash, kind: HashKind::Decoded };
    }
    let fixed = &record_data[..proto_type.item_size().min(record_data.len())];
    RecordHash { hash: fnv1a64(&[fixed]), kind: HashKind::Shallow }
}

fn decoded_hash(proto_type: PrototypeType, record_data: &[u8]) -> Option<u64> {
    match proto_type {
        #[cfg(feature = "json")]
        PrototypeType::Visual => {
            let v = crate::models::visual::parse_visual(record_data).ok()?;
            Some(fnv1a64(&[&serde_json::to_vec(&v).ok()?]))
        }
        PrototypeType::Model => model_hash(record_data),
        #[cfg(feature = "json")]
        PrototypeType::Material => {
            let m = crate::models::material::parse_material(record_data).ok()?;
            Some(fnv1a64(&[&serde_json::to_vec(&m).ok()?]))
        }
        PrototypeType::SkeletonExtender => skeleton_extender_hash(record_data),
        PrototypeType::AtlasContour => atlas_contour_hash(record_data),
        _ => None,
    }
}

/// ModelPrototype record (0x28 bytes): `u64 visual_resource_id`, `u8
/// skel_ext_count, misc_type, animations_count, dyes_count`, 4 pad bytes, then
/// record-relative offsets to `u64[skel_ext_count]` ids, the animation entries
/// (0x28 bytes each) and the dye entries (0x20 bytes each).
///
/// The animation entries aren't decoded (`parse_model` reads them as nested
/// models, which fails), so their raw bytes are hashed.
fn model_hash(buf: &[u8]) -> Option<u64> {
    const MODEL_ITEM_SIZE: usize = 0x28;
    const ANIMATION_ENTRY_SIZE: usize = 0x28;
    let header = buf.get(..0x0c)?;
    let skel_count = buf[8] as usize;
    let anim_count = buf[10] as usize;
    let dye_count = buf[11] as usize;
    let i64_at = |o: usize| buf.get(o..o + 8).map(|b| i64::from_le_bytes(b.try_into().unwrap()));
    let target = |o: usize| -> Option<usize> { usize::try_from(i64_at(o)?).ok() };
    if buf.len() < MODEL_ITEM_SIZE {
        return None;
    }

    let skel_ids: &[u8] = if skel_count > 0 {
        let at = target(0x10)?;
        buf.get(at..at.checked_add(skel_count * 8)?)?
    } else {
        &[]
    };
    let animations: &[u8] = if anim_count > 0 {
        let at = target(0x18)?;
        buf.get(at..at.checked_add(anim_count * ANIMATION_ENTRY_SIZE)?)?
    } else {
        &[]
    };
    let dyes: Vec<u8> = if dye_count > 0 {
        let at = target(0x20)?;
        dye_bytes(buf, at, dye_count)?
    } else {
        Vec::new()
    };
    Some(fnv1a64(&[header, skel_ids, animations, &dyes]))
}

#[cfg(feature = "json")]
fn dye_bytes(buf: &[u8], at: usize, count: usize) -> Option<Vec<u8>> {
    let dyes = crate::models::model::parse_dye_entries(buf, at, count).ok()?;
    serde_json::to_vec(&dyes).ok()
}

#[cfg(not(feature = "json"))]
fn dye_bytes(_buf: &[u8], _at: usize, _count: usize) -> Option<Vec<u8>> {
    None
}

/// SkeletonExtender record (0x20 bytes): `u16 type, u16 count, u32 pad`, then
/// three record-relative offsets to `u32[count]` placement ids, `u32[count]`
/// parent-bone hashes and `count` 4x4 f32 matrices.
fn skeleton_extender_hash(buf: &[u8]) -> Option<u64> {
    let u16_at = |o: usize| buf.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u64_at = |o: usize| buf.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let kind = u16_at(0)?;
    let count = u16_at(2)? as usize;
    let p0 = u64_at(8)? as usize;
    let p1 = u64_at(16)? as usize;
    let p2 = u64_at(24)? as usize;
    let ids = buf.get(p0..p0.checked_add(count * 4)?)?;
    let parents = buf.get(p1..p1.checked_add(count * 4)?)?;
    let matrices_end = p2.checked_add(count * 64)?.min(buf.len());
    let matrices = buf.get(p2..matrices_end)?;
    Some(fnv1a64(&[&kind.to_le_bytes(), &(count as u16).to_le_bytes(), ids, parents, matrices]))
}

/// AtlasContour record (0x10 bytes): `u64 frame_count` and an offset (from
/// the record) to `frame_count` frames. Each frame (0x10 bytes) is
/// `u64 point_count` and an offset (from the frame) to that many `(f32, f32)`
/// contour points. Only the counts and points are hashed: the offsets move
/// whenever an earlier record grows, and are the same in every record with
/// the same frame layout.
fn atlas_contour_hash(buf: &[u8]) -> Option<u64> {
    const FRAME_SIZE: usize = 0x10;
    const POINT_SIZE: usize = 8;
    const MAX_FRAMES: usize = 1 << 16;
    const MAX_POINTS: usize = 1 << 16;
    let u64_at = |o: usize| buf.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let target = |base: usize| -> Option<usize> {
        let rel = i64::from_le_bytes(buf.get(base + 8..base + 16)?.try_into().ok()?);
        usize::try_from(i64::try_from(base).ok()?.checked_add(rel)?).ok()
    };
    let frames = usize::try_from(u64_at(0)?).ok()?;
    if frames > MAX_FRAMES {
        return None;
    }
    let mut bytes = Vec::with_capacity(8 + frames * (8 + 8 * POINT_SIZE));
    bytes.extend_from_slice(&(frames as u64).to_le_bytes());
    if frames > 0 {
        let frames_at = target(0)?;
        for i in 0..frames {
            let frame = frames_at.checked_add(i * FRAME_SIZE)?;
            let points = usize::try_from(u64_at(frame)?).ok()?;
            if points > MAX_POINTS {
                return None;
            }
            bytes.extend_from_slice(&(points as u64).to_le_bytes());
            if points > 0 {
                let at = target(frame)?;
                bytes.extend_from_slice(buf.get(at..at.checked_add(points * POINT_SIZE)?)?);
            }
        }
    }
    Some(fnv1a64(&[&bytes]))
}

fn lookup_key(path: &str) -> &str {
    if path.is_empty() { "/" } else { path }
}

impl FileSystem for AssetsBinVfs {
    fn read_dir(&self, path: &str) -> vfs::VfsResult<Box<dyn Iterator<Item = String> + Send>> {
        let key = lookup_key(path);
        let children = self.dirs.get(key).ok_or_else(|| vfs::VfsError::from(VfsErrorKind::FileNotFound))?;
        Ok(Box::new(children.clone().into_iter()))
    }

    fn create_dir(&self, _path: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn open_file(&self, path: &str) -> vfs::VfsResult<Box<dyn vfs::SeekAndRead + Send>> {
        let key = lookup_key(path);
        let loc = self.files.get(key).ok_or_else(|| vfs::VfsError::from(VfsErrorKind::FileNotFound))?;
        crate::data::deps::record_file(key);
        let data = self.data[loc.byte_offset..loc.byte_end].to_vec();
        Ok(Box::new(Cursor::new(data)))
    }

    fn create_file(&self, _path: &str) -> vfs::VfsResult<Box<dyn vfs::SeekAndWrite + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn append_file(&self, _path: &str) -> vfs::VfsResult<Box<dyn vfs::SeekAndWrite + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn metadata(&self, path: &str) -> vfs::VfsResult<VfsMetadata> {
        let key = lookup_key(path);
        if let Some(loc) = self.files.get(key) {
            Ok(VfsMetadata {
                file_type: vfs::VfsFileType::File,
                len: (loc.byte_end - loc.byte_offset) as u64,
                created: None,
                modified: None,
                accessed: None,
            })
        } else if self.dirs.contains_key(key) {
            Ok(VfsMetadata {
                file_type: vfs::VfsFileType::Directory,
                len: 0,
                created: None,
                modified: None,
                accessed: None,
            })
        } else {
            Err(VfsErrorKind::FileNotFound.into())
        }
    }

    fn exists(&self, path: &str) -> vfs::VfsResult<bool> {
        let key = lookup_key(path);
        Ok(self.files.contains_key(key) || self.dirs.contains_key(key))
    }

    fn remove_file(&self, _path: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn remove_dir(&self, _path: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn set_creation_time(&self, _path: &str, _time: std::time::SystemTime) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn set_modification_time(&self, _path: &str, _time: std::time::SystemTime) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn set_access_time(&self, _path: &str, _time: std::time::SystemTime) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn copy_file(&self, _src: &str, _dest: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn move_file(&self, _src: &str, _dest: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn move_dir(&self, _src: &str, _dest: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }
}
