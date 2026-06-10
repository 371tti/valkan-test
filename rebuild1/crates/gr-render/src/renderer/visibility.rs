use crate::{
    math::{cross3, dot3, normalize_or, sub3},
    protocol::{CameraSnapshot, RenderOptimizationSettings, SceneBounds},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeshLodLevel {
    Full,
    Medium,
    Low,
}

impl MeshLodLevel {
    /// Returns the preferred LOD buffer index for backend-local mesh storage.
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::Medium => 1,
            Self::Low => 2,
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
        lod: choose_lod(camera, basis, viewport_height, bounds, settings),
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
    viewport_height: u32,
    bounds: SceneBounds,
    settings: RenderOptimizationSettings,
) -> MeshLodLevel {
    if !settings.distance_lod() {
        return MeshLodLevel::Full;
    }

    let Some(radius_px) = projected_radius_px(camera, basis, viewport_height, bounds) else {
        return MeshLodLevel::Full;
    };

    if radius_px >= settings.high_detail_screen_radius_px() {
        MeshLodLevel::Full
    } else if radius_px >= settings.medium_detail_screen_radius_px() {
        MeshLodLevel::Medium
    } else {
        MeshLodLevel::Low
    }
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
    fn distant_small_sphere_uses_low_lod() {
        let decision = classify_mesh(
            camera(),
            16.0 / 9.0,
            720,
            Some(bounds([0.0, 0.0, -90.0], 0.5)),
            RenderOptimizationSettings::balanced(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Low
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
            RenderOptimizationSettings::disabled(),
        );

        assert_eq!(
            decision,
            MeshVisibility::Visible {
                lod: MeshLodLevel::Full
            }
        );
    }
}
