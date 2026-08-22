use crate::{
    math::{cross3, dot3, normalize_or, sub3},
    protocol::{CameraSnapshot, RenderOptimizationSettings, SceneBounds},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshLodLevel {
    Full,
    Medium,
    Low,
    VeryLow,
}

impl MeshLodLevel {
    /// Returns the preferred LOD buffer index for backend-local mesh storage.
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::Medium => 1,
            Self::Low => 2,
            Self::VeryLow => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshCullReason {
    OutsideViewFrustum,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshVisibility {
    Visible { lod: MeshLodLevel },
    Culled { reason: MeshCullReason },
}

/// Classifies one mesh bounds against the active camera and optimization policy.
///
/// Missing bounds are intentionally treated as visible/full detail so incomplete import metadata
/// never hides an object. Culling and LOD use only owned snapshot/cached mesh data.
pub(crate) fn classify_mesh(
    camera: CameraSnapshot,
    aspect: f32,
    viewport_height: u32,
    bounds: Option<SceneBounds>,
    triangle_count: usize,
    settings: RenderOptimizationSettings,
) -> MeshVisibility {
    let Some(bounds) = bounds else {
        return MeshVisibility::Visible {
            lod: MeshLodLevel::Full,
        };
    };

    let basis = CameraBasis::from_snapshot(camera);
    if settings.frustum_culling() && !sphere_intersects_frustum(camera, basis, aspect, bounds) {
        return MeshVisibility::Culled {
            reason: MeshCullReason::OutsideViewFrustum,
        };
    }

    MeshVisibility::Visible {
        lod: choose_lod(
            camera,
            basis,
            aspect,
            viewport_height,
            bounds,
            triangle_count,
            settings,
        ),
    }
}

#[derive(Clone, Copy)]
struct CameraBasis {
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

impl CameraBasis {
    /// Builds an orthonormal basis from a renderer-facing camera snapshot.
    fn from_snapshot(camera: CameraSnapshot) -> Self {
        let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
        let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
        let up = cross3(right, forward);

        Self { forward, right, up }
    }
}

/// Returns whether a bounding sphere overlaps the perspective camera frustum.
fn sphere_intersects_frustum(
    camera: CameraSnapshot,
    basis: CameraBasis,
    aspect: f32,
    bounds: SceneBounds,
) -> bool {
    let to_center = sub3(bounds.center(), camera.eye);
    let depth = dot3(to_center, basis.forward);
    let radius = bounds.radius();

    if depth + radius < camera.near || depth - radius > camera.far {
        return false;
    }

    let side_depth = depth.max(camera.near);
    let tan_y = (camera.fov_y_radians * 0.5).tan();
    let tan_x = tan_y * aspect.max(0.001);
    let max_x = side_depth * tan_x + radius;
    let max_y = side_depth * tan_y + radius;
    let x = dot3(to_center, basis.right).abs();
    let y = dot3(to_center, basis.up).abs();

    x <= max_x && y <= max_y
}

/// Selects geometry detail from the projected screen-space size of a bounding sphere.
fn choose_lod(
    camera: CameraSnapshot,
    basis: CameraBasis,
    aspect: f32,
    viewport_height: u32,
    bounds: SceneBounds,
    triangle_count: usize,
    settings: RenderOptimizationSettings,
) -> MeshLodLevel {
    if !settings.distance_lod() {
        return MeshLodLevel::Full;
    }

    let viewport_area_px = viewport_pixel_area(aspect, viewport_height);
    let projected_radius = projected_radius_px(camera, basis, viewport_height, bounds);
    let (size_lod, projected_area_px) = if let Some(radius_px) = projected_radius {
        let size_lod = if radius_px >= settings.high_detail_screen_radius_px() {
            MeshLodLevel::Full
        } else if radius_px >= settings.medium_detail_screen_radius_px() {
            MeshLodLevel::Medium
        } else if radius_px >= settings.medium_detail_screen_radius_px() * 0.25 {
            MeshLodLevel::Low
        } else {
            MeshLodLevel::VeryLow
        };
        let circle_area = std::f32::consts::PI * radius_px * radius_px;
        (size_lod, circle_area.min(viewport_area_px))
    } else {
        // A sphere crossing the near plane (including a camera inside the bounds) has no finite
        // perspective radius. It can cover the screen, but it must not bypass triangle-density
        // LOD: high-poly camera shells were otherwise forced to full detail at the worst moment.
        (MeshLodLevel::Full, viewport_area_px)
    };
    let density_lod = triangle_density_lod(projected_area_px, triangle_count);
    if size_lod.index() >= density_lod.index() {
        size_lod
    } else {
        density_lod
    }
}

/// Prevents sub-pixel triangles from keeping an otherwise large high-poly mesh at full detail.
fn triangle_density_lod(projected_area_px: f32, triangle_count: usize) -> MeshLodLevel {
    // One source triangle per covered pixel is deliberately conservative: closed meshes normally
    // rasterize only their front-facing subset, while thin double-sided geometry needs the full
    // budget. Values above one retain geometry that is already sub-pixel before rasterization.
    const TRIANGLES_PER_PIXEL: f32 = 1.0;
    const MIN_TRIANGLE_BUDGET: f32 = 128.0;
    let budget = (projected_area_px * TRIANGLES_PER_PIXEL).max(MIN_TRIANGLE_BUDGET);
    let triangles = triangle_count as f32;

    if triangles <= budget {
        MeshLodLevel::Full
    } else if triangles * 0.60 <= budget {
        MeshLodLevel::Medium
    } else if triangles * 0.30 <= budget {
        MeshLodLevel::Low
    } else {
        MeshLodLevel::VeryLow
    }
}

/// Returns the actual viewport pixel budget reconstructed from its height and aspect ratio.
fn viewport_pixel_area(aspect: f32, viewport_height: u32) -> f32 {
    let height = viewport_height.max(1) as f32;
    let width = (height * aspect.max(0.001)).max(1.0);
    width * height
}

/// Projects a sphere radius into pixels using the active vertical field of view.
fn projected_radius_px(
    camera: CameraSnapshot,
    basis: CameraBasis,
    viewport_height: u32,
    bounds: SceneBounds,
) -> Option<f32> {
    let to_center = sub3(bounds.center(), camera.eye);
    let depth = dot3(to_center, basis.forward);
    if !depth.is_finite() || depth <= camera.near.max(f32::EPSILON) {
        return None;
    }

    let tan_y = (camera.fov_y_radians * 0.5).tan();
    if !tan_y.is_finite() || tan_y <= f32::EPSILON {
        return None;
    }

    let height = viewport_height.max(1) as f32;
    let radius_px = bounds.radius() / (depth * tan_y) * (height * 0.5);
    radius_px.is_finite().then_some(radius_px)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraSnapshot {
        CameraSnapshot::perspective(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid")
    }

    fn bounds(center: [f32; 3], radius: f32) -> SceneBounds {
        SceneBounds::new(center, radius).expect("test bounds are valid")
    }

    // Verifies that an object behind the camera is removed before Vulkan draw recording.
    #[test]
    fn culls_sphere_behind_camera() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, 2.0], 0.5)),
            300,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Culled {
                reason: MeshCullReason::OutsideViewFrustum
            }
        );
    }

    // Verifies that distant small objects select the coarsest generated index LOD.
    #[test]
    fn distant_small_sphere_uses_very_low_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -90.0], 0.5)),
            300,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::VeryLow
            }
        );
    }

    // Verifies that large on-screen meshes keep the full index stream.
    #[test]
    fn near_large_sphere_uses_full_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -3.0], 1.0)),
            300,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Full
            }
        );
    }

    // Verifies that mid-sized meshes drop to the medium generated index stream.
    #[test]
    fn mid_screen_sphere_uses_medium_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -8.0], 1.0)),
            300,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Medium
            }
        );
    }

    // Verifies that explicit full-detail policy keeps bounds visible at full geometry detail.
    #[test]
    fn disabled_policy_keeps_full_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -90.0], 0.5)),
            1_000_000,
            RenderOptimizationSettings::disabled(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Full
            }
        );
    }

    #[test]
    fn dense_near_mesh_avoids_subpixel_full_detail_triangles() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -3.0], 1.0)),
            500_000,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::VeryLow
            }
        );
    }

    // A sphere larger than the screen must not invent off-screen pixels to justify full geometry.
    #[test]
    fn ten_million_triangle_closeup_is_bounded_by_viewport_pixels() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -0.2], 1.0)),
            10_000_000,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::VeryLow
            }
        );
    }

    // Near-plane projection is singular when the camera sits inside a mesh. Treating the object
    // as full-screen preserves a finite density budget instead of falling back to full detail.
    #[test]
    fn camera_inside_dense_mesh_still_uses_density_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, 0.0], 2.0)),
            10_000_000,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::VeryLow
            }
        );
    }

    // Camera-inside handling remains quality-first for geometry below the viewport budget.
    #[test]
    fn camera_inside_modest_mesh_keeps_full_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, 0.0], 2.0)),
            100_000,
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Full
            }
        );
    }
}
