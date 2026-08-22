use std::mem;

use meshopt::SimplifyOptions;

use crate::renderer::{assets::MeshVertex, visibility::MeshLodLevel};

const MEDIUM_LOD_RATIO: f32 = 0.60;
const LOW_LOD_RATIO: f32 = 0.30;
const VERY_LOW_LOD_RATIO: f32 = 0.10;
const MEDIUM_LOD_ERROR: f32 = 0.025;
const LOW_LOD_ERROR: f32 = 0.075;
const VERY_LOW_LOD_ERROR: f32 = 0.15;
const OVERDRAW_CACHE_THRESHOLD: f32 = 1.05;
const MEDIUM_LOD_OPTIONS: SimplifyOptions = SimplifyOptions::LockBorder;
const LOW_LOD_OPTIONS: SimplifyOptions = SimplifyOptions::None;
const VERY_LOW_LOD_OPTIONS: SimplifyOptions = SimplifyOptions::Regularize;
const LOD_ATTRIBUTE_COUNT: usize = 5;
const LOD_ATTRIBUTE_WEIGHTS: [f32; LOD_ATTRIBUTE_COUNT] = [
    1.0, 1.0, 1.0, // shading normal
    1.0, 1.0, // material UV
];

/// Attribute stream and hard locks shared by every generated LOD of one mesh.
struct LodSimplificationData {
    attributes: Vec<f32>,
    vertex_locks: Vec<bool>,
}

impl LodSimplificationData {
    fn new(vertices: &[MeshVertex]) -> Self {
        let mut attributes = Vec::with_capacity(vertices.len() * LOD_ATTRIBUTE_COUNT);
        for vertex in vertices {
            attributes.extend_from_slice(&vertex.normal);
            attributes.extend_from_slice(&vertex.uv);
        }

        Self {
            attributes,
            vertex_locks: vec![false; vertices.len()],
        }
    }
}

/// Builds full through very-low LOD index streams and removes duplicates.
///
/// The mesh module owns GPU upload, while this helper owns the CPU-side simplification policy.
pub(super) fn unique_lod_indices(
    vertices: &[MeshVertex],
    indices: &[u32],
    reduce_overdraw: bool,
) -> Vec<(MeshLodLevel, Vec<u32>)> {
    let simplification_data = LodSimplificationData::new(vertices);
    let mut lods = Vec::with_capacity(4);
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Full,
        optimized_index_order(indices, vertices, reduce_overdraw),
    );
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Medium,
        simplified_lod_indices(
            vertices,
            indices,
            MEDIUM_LOD_RATIO,
            MEDIUM_LOD_ERROR,
            MEDIUM_LOD_OPTIONS,
            reduce_overdraw,
            &simplification_data,
        ),
    );
    push_unique_lod(
        &mut lods,
        MeshLodLevel::Low,
        simplified_lod_indices(
            vertices,
            indices,
            LOW_LOD_RATIO,
            LOW_LOD_ERROR,
            LOW_LOD_OPTIONS,
            reduce_overdraw,
            &simplification_data,
        ),
    );
    push_unique_lod(
        &mut lods,
        MeshLodLevel::VeryLow,
        simplified_lod_indices(
            vertices,
            indices,
            VERY_LOW_LOD_RATIO,
            VERY_LOW_LOD_ERROR,
            VERY_LOW_LOD_OPTIONS,
            reduce_overdraw,
            &simplification_data,
        ),
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

/// Simplifies one triangle-list index stream according to its LOD policy while preserving validity.
fn simplified_lod_indices(
    vertices: &[MeshVertex],
    indices: &[u32],
    ratio: f32,
    target_error: f32,
    options: SimplifyOptions,
    reduce_overdraw: bool,
    simplification_data: &LodSimplificationData,
) -> Vec<u32> {
    if vertices.len() < 3 || indices.len() < 6 {
        return optimized_index_order(indices, vertices, reduce_overdraw);
    }

    let target_count = lod_target_index_count(indices.len(), ratio);
    if target_count >= indices.len() {
        return optimized_index_order(indices, vertices, reduce_overdraw);
    }

    let simplified = meshopt::simplify_with_attributes_and_locks_decoder(
        indices,
        vertices,
        &simplification_data.attributes,
        &LOD_ATTRIBUTE_WEIGHTS,
        LOD_ATTRIBUTE_COUNT * mem::size_of::<f32>(),
        &simplification_data.vertex_locks,
        target_count,
        target_error,
        options,
        None,
    );
    let aligned = triangle_aligned_indices(simplified, indices);
    optimized_index_order(&aligned, vertices, reduce_overdraw)
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

/// Reorders an index stream for the vertex cache and, when safe, for lower pixel overdraw.
///
/// Overdraw optimization may change triangle submission order. The caller therefore disables it
/// for alpha-blended materials, where order is part of the rendered result.
fn optimized_index_order(
    indices: &[u32],
    vertices: &[MeshVertex],
    reduce_overdraw: bool,
) -> Vec<u32> {
    let mut optimized = meshopt::optimize_vertex_cache(indices, vertices.len());
    if reduce_overdraw && optimized.len() >= 6 {
        meshopt::optimize_overdraw_in_place_decoder(
            &mut optimized,
            vertices,
            OVERDRAW_CACHE_THRESHOLD,
        );
    }
    optimized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "repository asset profiling helper"]
    fn profile_repository_model_lods() {
        use std::path::Path;

        use crate::{
            import::import_asset,
            protocol::{CameraSnapshot, RenderOptimizationSettings},
            renderer::{
                assets::MeshGeometry,
                visibility::{MeshVisibility, classify_mesh},
            },
        };

        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/model.glb");
        let scene = import_asset(&path).expect("repository model should import");
        let mut triangles = [0_usize; 4];
        let mut buffers = [0_usize; 4];
        let mut source_vertices_transformed = 0_usize;
        let mut optimized_vertices_transformed = 0_usize;
        let mut source_fetch_bytes = 0_usize;
        let mut optimized_fetch_bytes = 0_usize;
        let mut fetch_reordered_bytes = 0_usize;
        let mut selected_requests = [0_usize; 4];
        let mut selected_actual = [0_usize; 4];
        let mut selected_actual_triangles = [0_usize; 4];
        let mut selected_triangles = 0_usize;
        let mut sloppy_very_low_triangles = 0_usize;
        let mut sloppy_selected_triangles = 0_usize;
        let mut sloppy_max_error = 0.0_f32;
        let mut unlocked_very_low_triangles = 0_usize;
        let mut unlocked_selected_triangles = 0_usize;
        let mut unlocked_max_error = 0.0_f32;
        let mut regularized_very_low_triangles = 0_usize;
        let mut regularized_selected_triangles = 0_usize;
        let mut regularized_max_error = 0.0_f32;
        let mut culled = 0_usize;
        let scene_bounds = scene.bounds().expect("repository model has finite bounds");
        let center = scene_bounds.center();
        let radius = scene_bounds.radius().max(1.0);
        let eye = [
            center[0],
            center[1] + radius * 0.24,
            center[2] + radius * 2.45,
        ];
        let camera = CameraSnapshot::perspective(
            eye,
            center,
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.06,
            5000.0,
        )
        .expect("profile camera is finite");

        for mesh in scene.meshes() {
            let geometry = MeshGeometry::from_imported(mesh);
            if geometry.indices().len() < 3 {
                continue;
            }

            let source_cache = meshopt::analyze::analyze_vertex_cache(
                geometry.indices(),
                geometry.vertices().len(),
                16,
                32,
                256,
            );
            let source_fetch = meshopt::analyze::analyze_vertex_fetch(
                geometry.indices(),
                geometry.vertices().len(),
                32,
            );
            source_vertices_transformed += source_cache.vertices_transformed as usize;
            source_fetch_bytes += source_fetch.bytes_fetched as usize;

            let lods = unique_lod_indices(geometry.vertices(), geometry.indices(), false);
            let sloppy_target =
                lod_target_index_count(geometry.indices().len(), VERY_LOW_LOD_RATIO);
            let mut sloppy_error = 0.0_f32;
            let sloppy = triangle_aligned_indices(
                meshopt::simplify_sloppy_decoder(
                    geometry.indices(),
                    geometry.vertices(),
                    sloppy_target,
                    VERY_LOW_LOD_ERROR,
                    Some(&mut sloppy_error),
                ),
                geometry.indices(),
            );
            let mut unlocked_error = 0.0_f32;
            let unlocked = triangle_aligned_indices(
                meshopt::simplify_decoder(
                    geometry.indices(),
                    geometry.vertices(),
                    sloppy_target,
                    VERY_LOW_LOD_ERROR,
                    SimplifyOptions::None,
                    Some(&mut unlocked_error),
                ),
                geometry.indices(),
            );
            let mut regularized_error = 0.0_f32;
            let regularized = triangle_aligned_indices(
                meshopt::simplify_decoder(
                    geometry.indices(),
                    geometry.vertices(),
                    sloppy_target,
                    VERY_LOW_LOD_ERROR,
                    VERY_LOW_LOD_OPTIONS,
                    Some(&mut regularized_error),
                ),
                geometry.indices(),
            );
            sloppy_very_low_triangles += sloppy.len() / 3;
            sloppy_max_error = sloppy_max_error.max(sloppy_error);
            unlocked_very_low_triangles += unlocked.len() / 3;
            unlocked_max_error = unlocked_max_error.max(unlocked_error);
            regularized_very_low_triangles += regularized.len() / 3;
            regularized_max_error = regularized_max_error.max(regularized_error);
            for (level, indices) in &lods {
                let index = level.index();
                triangles[index] += indices.len() / 3;
                buffers[index] += 1;
                if *level == MeshLodLevel::Full {
                    let optimized_cache = meshopt::analyze::analyze_vertex_cache(
                        &indices,
                        geometry.vertices().len(),
                        16,
                        32,
                        256,
                    );
                    let optimized_fetch = meshopt::analyze::analyze_vertex_fetch(
                        &indices,
                        geometry.vertices().len(),
                        32,
                    );
                    let mut fetch_indices = indices.clone();
                    let dummy_vertices = vec![[0_u8; 32]; geometry.vertices().len()];
                    let reordered_vertices =
                        meshopt::optimize_vertex_fetch(&mut fetch_indices, &dummy_vertices);
                    let reordered_fetch = meshopt::analyze::analyze_vertex_fetch(
                        &fetch_indices,
                        reordered_vertices.len(),
                        32,
                    );
                    optimized_vertices_transformed += optimized_cache.vertices_transformed as usize;
                    optimized_fetch_bytes += optimized_fetch.bytes_fetched as usize;
                    fetch_reordered_bytes += reordered_fetch.bytes_fetched as usize;
                }
            }

            match classify_mesh(
                camera,
                1280.0 / 720.0,
                720,
                geometry.bounds(),
                geometry.indices().len() / 3,
                RenderOptimizationSettings::balanced(),
            ) {
                MeshVisibility::Culled { .. } => culled += 1,
                MeshVisibility::Visible { lod: requested } => {
                    selected_requests[requested.index()] += 1;
                    let (actual, indices) = lods
                        .iter()
                        .filter(|(level, _)| level.index() <= requested.index())
                        .max_by_key(|(level, _)| level.index())
                        .expect("full LOD is always present");
                    selected_actual[actual.index()] += 1;
                    let triangle_count = indices.len() / 3;
                    selected_actual_triangles[actual.index()] += triangle_count;
                    selected_triangles += triangle_count;
                    sloppy_selected_triangles += if requested == MeshLodLevel::VeryLow {
                        sloppy.len() / 3
                    } else {
                        triangle_count
                    };
                    unlocked_selected_triangles += if requested == MeshLodLevel::VeryLow {
                        unlocked.len() / 3
                    } else {
                        triangle_count
                    };
                    regularized_selected_triangles += if requested == MeshLodLevel::VeryLow {
                        regularized.len() / 3
                    } else {
                        triangle_count
                    };
                }
            }
        }

        let source_triangles = triangles[MeshLodLevel::Full.index()];
        eprintln!(
            "lod-profile meshes={} triangles={triangles:?} buffers={buffers:?} cache_transforms={source_vertices_transformed}->{optimized_vertices_transformed} acmr={:.4}->{:.4} fetch_bytes={source_fetch_bytes}->{optimized_fetch_bytes}->{fetch_reordered_bytes}",
            scene.mesh_count(),
            source_vertices_transformed as f64 / source_triangles as f64,
            optimized_vertices_transformed as f64 / source_triangles as f64,
        );
        eprintln!(
            "lod-selection scene_center={center:?} scene_radius={radius:.3} camera_eye={eye:?} requested={selected_requests:?} actual={selected_actual:?} actual_triangles={selected_actual_triangles:?} culled={culled} selected_triangles={selected_triangles} sloppy_very_low_triangles={sloppy_very_low_triangles} sloppy_selected_triangles={sloppy_selected_triangles} sloppy_max_error={sloppy_max_error:.5}"
        );
        eprintln!(
            "very-low-options unlocked_triangles={unlocked_very_low_triangles} unlocked_selected={unlocked_selected_triangles} unlocked_max_error={unlocked_max_error:.5} regularized_triangles={regularized_very_low_triangles} regularized_selected={regularized_selected_triangles} regularized_max_error={regularized_max_error:.5}"
        );
    }

    // Verifies that generated LOD targets remain triangle-aligned and conservative.
    #[test]
    fn lod_target_counts_keep_triangle_groups() {
        let medium = lod_target_index_count(101, MEDIUM_LOD_RATIO);
        let low = lod_target_index_count(101, LOW_LOD_RATIO);
        let very_low = lod_target_index_count(101, VERY_LOW_LOD_RATIO);

        assert_eq!(medium % 3, 0);
        assert_eq!(low % 3, 0);
        assert_eq!(very_low % 3, 0);
        assert!(medium < 101);
        assert!(low < medium);
        assert!(very_low < low);
        assert_eq!(lod_target_index_count(300, MEDIUM_LOD_RATIO), 180);
        assert_eq!(lod_target_index_count(300, LOW_LOD_RATIO), 90);
        assert_eq!(lod_target_index_count(300, VERY_LOW_LOD_RATIO), 30);
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
        let simplification_data = LodSimplificationData::new(&vertices);
        let lod = simplified_lod_indices(
            &vertices,
            &indices,
            LOW_LOD_RATIO,
            LOW_LOD_ERROR,
            LOW_LOD_OPTIONS,
            false,
            &simplification_data,
        );

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
        let lods = unique_lod_indices(&vertices, &indices, false);

        assert_eq!(lods.len(), 1);
        assert_eq!(lods[0].0, MeshLodLevel::Full);
    }

    // Open foliage and cloth often consist almost entirely of border vertices. Medium keeps the
    // silhouette locked, while Low deliberately unlocks it so distance and cascade LODs still
    // reduce vertex work.
    #[test]
    fn low_lod_reduces_open_strip_when_border_lock_is_disabled() {
        let column_count = 12_usize;
        let mut vertices = Vec::with_capacity(column_count * 2);
        for column in 0..column_count {
            let x = column as f32;
            vertices.push(MeshVertex::new(
                [x, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [x / (column_count - 1) as f32, 0.0],
                [1.0; 4],
            ));
            vertices.push(MeshVertex::new(
                [x, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [x / (column_count - 1) as f32, 1.0],
                [1.0; 4],
            ));
        }

        let mut indices = Vec::with_capacity((column_count - 1) * 6);
        for column in 0..column_count - 1 {
            let lower_left = (column * 2) as u32;
            let upper_left = lower_left + 1;
            let lower_right = lower_left + 2;
            let upper_right = lower_left + 3;
            indices.extend_from_slice(&[
                lower_left,
                lower_right,
                upper_left,
                upper_left,
                lower_right,
                upper_right,
            ]);
        }
        let simplification_data = LodSimplificationData::new(&vertices);

        let locked = simplified_lod_indices(
            &vertices,
            &indices,
            LOW_LOD_RATIO,
            LOW_LOD_ERROR,
            SimplifyOptions::LockBorder,
            false,
            &simplification_data,
        );
        let unlocked = simplified_lod_indices(
            &vertices,
            &indices,
            LOW_LOD_RATIO,
            LOW_LOD_ERROR,
            LOW_LOD_OPTIONS,
            false,
            &simplification_data,
        );
        let lods = unique_lod_indices(&vertices, &indices, false);
        let stored_low = lods
            .iter()
            .find(|(level, _)| *level == MeshLodLevel::Low)
            .expect("unlocked Low LOD should be stored");

        assert_eq!(MEDIUM_LOD_OPTIONS, SimplifyOptions::LockBorder);
        assert_eq!(LOW_LOD_OPTIONS, SimplifyOptions::None);
        assert_eq!(VERY_LOW_LOD_OPTIONS, SimplifyOptions::Regularize);
        assert_eq!(locked.len(), indices.len());
        assert!(unlocked.len() < locked.len());
        assert_eq!(unlocked.len() % 3, 0);
        assert_eq!(stored_low.1, unlocked);
    }

    #[test]
    fn overdraw_reordering_preserves_triangle_topology() {
        let vertices = [
            MeshVertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [1.0; 4]),
            MeshVertex::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [1.0; 4]),
            MeshVertex::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0; 4]),
            MeshVertex::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0], [1.0; 4]),
            MeshVertex::new([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0], [1.0; 4]),
            MeshVertex::new([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0; 4]),
        ];
        let indices = [0, 1, 2, 3, 4, 5, 0, 3, 5, 0, 5, 2];
        let optimized = optimized_index_order(&indices, &vertices, true);

        let mut expected = indices
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect::<Vec<_>>();
        let mut actual = optimized
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect::<Vec<_>>();
        expected.sort_unstable();
        actual.sort_unstable();

        assert_eq!(actual, expected);
    }
}
