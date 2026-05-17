//! MikkTSpace tangent generation for glTF emit.
//!
//! WG `.geometry` vertex data carries POSITION + NORMAL + UV0 but **no
//! TANGENT** attribute. Without it, downstream consumers (Three.js's
//! `MeshStandardMaterial`, Blender's PBR shader, Unity's surface shader)
//! must reconstruct tangent space from screen-space derivatives at
//! shading time. That fallback produces washed-out normal-map detail
//! that degrades with viewing angle and tessellation density — visible
//! as "the normal map isn't doing anything" in viewers.
//!
//! Emitting MikkTSpace tangents at export time fixes every consumer at
//! once with no shader-side change. MikkTSpace is the de-facto standard
//! the glTF spec recommends for tangent generation.
//!
//! Implementation uses [`bevy_mikktspace`] (a maintained safe-Rust port
//! of Mikkelsen's reference algorithm). The trait expects per-face-corner
//! lookups; we map those back to vertex indices via the index buffer
//! and accumulate per-vertex tangents. WG meshes already split vertices
//! at UV seams, so per-vertex averaging is lossless in practice.

use bevy_mikktspace::{Geometry, TangentSpace};

/// Compute per-vertex tangents for a triangle-indexed mesh using
/// MikkTSpace. Returns a `Vec<[f32; 4]>` (xyz = tangent direction,
/// w = bitangent handedness sign per glTF spec) parallel to the
/// input vertex arrays.
///
/// Requirements:
/// - `indices.len()` is a multiple of 3 (triangle list)
/// - All arrays parallel-indexed; vertex `i` has position[i], normal[i], uv[i]
/// - At least one face references each vertex (orphans get a zero tangent)
///
/// Returns an empty `Vec` (NOT zero-filled) when the inputs are
/// degenerate (no indices, no vertices, or mismatched vertex array
/// lengths). Callers should treat empty-output as "skip TANGENT
/// attribute emit" rather than emit a zero accessor.
pub fn compute_mikkt_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    if indices.is_empty() || indices.len() % 3 != 0 {
        return Vec::new();
    }
    let n = positions.len();
    if normals.len() != n || uvs.len() != n {
        // Vertex arrays disagree on length — refuse rather than risk
        // an out-of-bounds during tangent accumulation.
        return Vec::new();
    }

    let mut mesh = TangentMesh {
        positions,
        normals,
        uvs,
        indices,
        // Accumulate (sum.xyz, sum.w, count) per vertex. After
        // generate_tangents, divide xyz by count and renormalize w by
        // majority-sign.
        accum: vec![[0.0f32; 4]; n],
        // Track how many face-corners contributed per vertex — drives
        // the post-pass averaging. Also catches orphan vertices (count
        // == 0) so we can emit a unit-X fallback rather than NaN.
        counts: vec![0u32; n],
    };

    // bevy_mikktspace returns Err for degenerate input (zero-area
    // triangles, NaN positions, etc.). Treat that as "no tangents" —
    // the caller skips the TANGENT attribute, leaving consumers on
    // their existing derivative-based fallback.
    if bevy_mikktspace::generate_tangents(&mut mesh).is_err() {
        return Vec::new();
    }

    // Average + normalize per vertex. The bitangent sign (.w) is
    // expected to be ±1; averaging produces near-±1 magnitudes which
    // we snap back to a hard sign.
    let mut out = Vec::with_capacity(n);
    for (acc, &c) in mesh.accum.iter().zip(mesh.counts.iter()) {
        if c == 0 {
            // Orphan vertex (no face references it). Emit a placeholder
            // tangent along world X; the consumer's normal-map sample
            // is undefined here anyway.
            out.push([1.0, 0.0, 0.0, 1.0]);
            continue;
        }
        let inv_c = 1.0 / (c as f32);
        let tx = acc[0] * inv_c;
        let ty = acc[1] * inv_c;
        let tz = acc[2] * inv_c;
        let len_sq = tx * tx + ty * ty + tz * tz;
        let (nx, ny, nz) = if len_sq > 1.0e-12 {
            let inv = 1.0 / len_sq.sqrt();
            (tx * inv, ty * inv, tz * inv)
        } else {
            (1.0, 0.0, 0.0)
        };
        // Average of ±1 handedness → sign of the average; ties resolve to +1.
        let w = if acc[3] >= 0.0 { 1.0 } else { -1.0 };
        out.push([nx, ny, nz, w]);
    }
    out
}

/// Internal adapter implementing `bevy_mikktspace::Geometry` over our
/// flat `[positions, normals, uvs, indices]` mesh representation.
///
/// `accum` accumulates per-vertex tangents across face-corner writes;
/// `counts` records the number of contributions so the post-pass can
/// average without an extra pass over `indices`.
struct TangentMesh<'a> {
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    indices: &'a [u32],
    accum: Vec<[f32; 4]>,
    counts: Vec<u32>,
}

impl<'a> TangentMesh<'a> {
    #[inline]
    fn vert_index(&self, face: usize, vert: usize) -> usize {
        self.indices[face * 3 + vert] as usize
    }
}

impl<'a> Geometry for TangentMesh<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.vert_index(face, vert)]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.vert_index(face, vert)]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.vert_index(face, vert)]
    }

    fn set_tangent(&mut self, tangent: Option<TangentSpace>, face: usize, vert: usize) {
        let Some(ts) = tangent else { return };
        let encoded = ts.tangent_encoded();
        let v = self.vert_index(face, vert);
        let a = &mut self.accum[v];
        a[0] += encoded[0];
        a[1] += encoded[1];
        a[2] += encoded[2];
        a[3] += encoded[3];
        self.counts[v] = self.counts[v].saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inputs_yield_empty_output() {
        let out = compute_mikkt_tangents(&[], &[], &[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn mismatched_lengths_yield_empty_output() {
        let positions = vec![[0.0, 0.0, 0.0]; 3];
        let normals = vec![[0.0, 0.0, 1.0]; 2]; // wrong length
        let uvs = vec![[0.0, 0.0]; 3];
        let indices = vec![0, 1, 2];
        let out = compute_mikkt_tangents(&positions, &normals, &uvs, &indices);
        assert!(out.is_empty());
    }

    #[test]
    fn single_triangle_produces_per_vertex_tangents() {
        // A flat triangle in the XY plane with a U-axis-aligned UV layout
        // should produce tangents pointing in +X (the U-axis direction).
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2];

        let out = compute_mikkt_tangents(&positions, &normals, &uvs, &indices);
        assert_eq!(out.len(), 3);
        for t in &out {
            // Length ≈ 1
            let len_sq = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
            assert!((len_sq - 1.0).abs() < 1.0e-4, "tangent not unit length: {t:?}");
            // X component dominant
            assert!(t[0].abs() > 0.9, "tangent X not dominant: {t:?}");
            // Handedness is ±1
            assert!(t[3].abs() > 0.99 && t[3].abs() < 1.01, "handedness not ±1: {t:?}");
        }
    }
}
