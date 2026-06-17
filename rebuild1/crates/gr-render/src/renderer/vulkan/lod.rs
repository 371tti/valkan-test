use meshopt::SimplifyOptions;

use crate::renderer::{assets::MeshVertex, visibility::MeshLodLevel};

const MEDIUM_LOD_RATIO: f32 = 0.72;
const LOW_LOD_RATIO: f32 = 0.45;
const MEDIUM_LOD_ERROR: f32 = 0.015;
const LOW_LOD_ERROR: f32 = 0.04;

/// Builds full, medium, and low LOD index streams and removes duplicates.
///
/// The mesh module owns GPU upload, while this helper owns the CPU-side simplification policy.
pub(super) fn unique_lod_indices(
    vertices: &[MeshVertex],
    indices: &[u32],
) -> Vec<(MeshLodLevel, Vec<u32>)> {
    let mut lods = Vec::with_capacity(3);
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Full,
        optimized_index_order(indices, vertices.len()),
    );
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Medium,
        simplified_lod_indices(vertices, indices, MEDIUM_LOD_RATIO, MEDIUM_LOD_ERROR),
    );
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Low,
        simplified_lod_indices(vertices, indices, LOW_LOD_RATIO, LOW_LOD_ERROR),
    );
    lods
}

/// Keeps one LOD stream only when it is not identical to the previous uploaded stream.
fn push_unique_lod(
    lods: &mut Vec<(MeshLodLevel, Vec<u32>)>,
    level: MeshLodLevel,
    indices: Vec<u32>,
) {
    if lods
        .last()
        .is_some_and(|(_, previous)| previous == &indices)
    {
        return;
    }
    lods.push((level, indices));
}

/// Simplifies one triangle-list index stream while preserving borders and triangle validity.
fn simplified_lod_indices(
    vertices: &[MeshVertex],
    indices: &[u32],
    ratio: f32,
    target_error: f32,
) -> Vec<u32> {
    if vertices.len() < 3 || indices.len() < 6 {
        return optimized_index_order(indices, vertices.len());
    }

    let target_count = lod_target_index_count(indices.len(), ratio);
    if target_count >= indices.len() {
        return optimized_index_order(indices, vertices.len());
    }

    let simplified = meshopt::simplify_decoder(
        indices,
        vertices,
        target_count,
        target_error,
        SimplifyOptions::LockBorder,
        None,
    );
    let aligned = triangle_aligned_indices(simplified, indices);
    optimized_index_order(&aligned, vertices.len())
}

/// Returns a triangle-aligned target count for meshoptimizer simplification.
fn lod_target_index_count(index_count: usize, ratio: f32) -> usize {
    let target = (index_count as f32 * ratio.clamp(0.05, 1.0)).round() as usize;
    (target / 3).max(1) * 3
}

/// Drops any incomplete trailing triangle and falls back when simplification produced no draw.
fn triangle_aligned_indices(mut indices: Vec<u32>, fallback: &[u32]) -> Vec<u32> {
    indices.truncate(indices.len() / 3 * 3);
    if indices.len() < 3 {
        fallback.to_vec()
    } else {
        indices
    }
}

/// Reorders an index stream for the GPU vertex cache without changing mesh topology.
fn optimized_index_order(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    meshopt::optimize_vertex_cache(indices, vertex_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that generated LOD targets remain triangle-aligned and conservative.
    #[test]
    fn lod_target_counts_keep_triangle_groups() {
        let medium = lod_target_index_count(101, MEDIUM_LOD_RATIO);
        let low = lod_target_index_count(101, LOW_LOD_RATIO);

        assert_eq!(medium % 3, 0);
        assert_eq!(low % 3, 0);
        assert!(medium < 101);
        assert!(low < medium);
    }

    // Verifies that tiny meshes never disappear when simplification cannot safely reduce them.
    #[test]
    fn tiny_lod_keeps_drawable_indices() {
        let vertices = [
            MeshVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0], [1.0; 4]),
            MeshVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0], [1.0; 4]),
            MeshVertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0], [1.0; 4]),
        ];
        let indices = [0, 1, 2];
        let lod = simplified_lod_indices(&vertices, &indices, LOW_LOD_RATIO, LOW_LOD_ERROR);

        assert_eq!(lod.len(), 3);
    }

    // Verifies that duplicate LOD buffers are skipped for meshes that cannot simplify safely.
    #[test]
    fn unique_lods_skip_duplicate_tiny_mesh_buffers() {
        let vertices = [
            MeshVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0], [1.0; 4]),
            MeshVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0], [1.0; 4]),
            MeshVertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0], [1.0; 4]),
        ];
        let indices = [0, 1, 2];
        let lods = unique_lod_indices(&vertices, &indices);

        assert_eq!(lods.len(), 1);
        assert_eq!(lods[0].0, MeshLodLevel::Full);
    }
}
