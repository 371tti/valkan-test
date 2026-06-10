use thiserror::Error;

use crate::math::{cross3, dot3, length3, normalize_or, sub3};

use super::{
    Exposure, ExternalObjectId, FrameId, MaterialHandle, MeshHandle, NonZeroExtent, SceneHandle,
    SurfaceGeneration, SurfaceId, ViewId,
};

#[derive(Clone, Debug)]
pub struct ViewPacket {
    pub view_id: ViewId,
    pub extent: NonZeroExtent,
    pub camera: CameraSnapshot,
}

impl ViewPacket {
    /// Creates one extracted view for a frame snapshot.
    pub fn new(view_id: ViewId, extent: NonZeroExtent) -> Self {
        Self {
            view_id,
            extent,
            camera: CameraSnapshot::default(),
        }
    }

    /// Replaces the default camera with an extracted user/ECS camera snapshot.
    pub fn with_camera(mut self, camera: CameraSnapshot) -> Self {
        self.camera = camera;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraSnapshot {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraEffects {
    exposure: Exposure,
    white_balance: [f32; 3],
    contrast: f32,
    saturation: f32,
    enabled: bool,
}

impl CameraEffects {
    const MIN_WHITE_BALANCE: f32 = 0.25;
    const MAX_WHITE_BALANCE: f32 = 4.0;
    const MIN_CONTRAST: f32 = 0.25;
    const MAX_CONTRAST: f32 = 4.0;
    const MIN_SATURATION: f32 = 0.0;
    const MAX_SATURATION: f32 = 4.0;

    /// Creates the camera response used by post processing for one extracted frame.
    ///
    /// The renderer receives only these owned scalar values. Auto exposure, white-balance
    /// metering, and ECS/user state remain on the app side of the protocol boundary.
    pub fn new(exposure: Exposure, white_balance: [f32; 3]) -> Option<Self> {
        Self::with_look(exposure, white_balance, 1.0, 1.0, true)
    }

    /// Creates the full camera look after validating values that cross the renderer boundary.
    pub fn with_look(
        exposure: Exposure,
        white_balance: [f32; 3],
        contrast: f32,
        saturation: f32,
        enabled: bool,
    ) -> Option<Self> {
        let valid_white_balance = white_balance.iter().all(|value| {
            value.is_finite() && (Self::MIN_WHITE_BALANCE..=Self::MAX_WHITE_BALANCE).contains(value)
        });
        let valid_contrast =
            contrast.is_finite() && (Self::MIN_CONTRAST..=Self::MAX_CONTRAST).contains(&contrast);
        let valid_saturation = saturation.is_finite()
            && (Self::MIN_SATURATION..=Self::MAX_SATURATION).contains(&saturation);

        (valid_white_balance && valid_contrast && valid_saturation).then_some(Self {
            exposure,
            white_balance,
            contrast,
            saturation,
            enabled,
        })
    }

    /// Returns a copy with a different validated exposure and the same look values.
    pub fn with_exposure(self, exposure: Exposure) -> Self {
        Self { exposure, ..self }
    }

    /// Returns whether post processing should apply camera response before tone mapping.
    pub fn enabled(self) -> bool {
        self.enabled
    }

    /// Returns the finite non-negative exposure multiplier used before tone mapping.
    pub fn exposure(self) -> Exposure {
        self.exposure
    }

    /// Returns per-channel white-balance multipliers used before tone mapping.
    pub fn white_balance(self) -> [f32; 3] {
        self.white_balance
    }

    /// Returns the contrast multiplier applied around middle gray before tone mapping.
    pub fn contrast(self) -> f32 {
        self.contrast
    }

    /// Returns the saturation multiplier applied in linear scene color.
    pub fn saturation(self) -> f32 {
        self.saturation
    }
}

impl Default for CameraEffects {
    /// Creates a neutral camera response that keeps existing snapshots valid.
    fn default() -> Self {
        Self {
            exposure: Exposure::default(),
            white_balance: [1.0; 3],
            contrast: 1.0,
            saturation: 1.0,
            enabled: true,
        }
    }
}

impl CameraSnapshot {
    /// Creates a finite perspective camera snapshot before it crosses the renderer boundary.
    pub fn perspective(
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fov_y_radians: f32,
        near: f32,
        far: f32,
    ) -> Option<Self> {
        let camera = Self {
            eye,
            target,
            up,
            fov_y_radians,
            near,
            far,
        };
        camera.is_valid().then_some(camera)
    }

    /// Returns a Vulkan clip-space projection multiplied by the right-handed view matrix.
    pub fn view_projection(self, aspect: f32) -> [f32; 16] {
        mat4_mul(
            perspective_vulkan(self.fov_y_radians, aspect, self.near, self.far),
            look_at_rh(self.eye, self.target, self.up),
        )
    }

    /// Returns whether all camera values are finite and the clipping planes are ordered.
    fn is_valid(self) -> bool {
        self.eye.iter().all(|value| value.is_finite())
            && self.target.iter().all(|value| value.is_finite())
            && self.up.iter().all(|value| value.is_finite())
            && self.fov_y_radians.is_finite()
            && (0.1..std::f32::consts::PI - 0.1).contains(&self.fov_y_radians)
            && self.near.is_finite()
            && self.far.is_finite()
            && self.near > 0.0
            && self.far > self.near
            && length3(sub3(self.target, self.eye)) > f32::EPSILON
            && length3(self.up) > f32::EPSILON
    }
}

impl Default for CameraSnapshot {
    /// Creates a small default viewer camera for early app snapshots without ECS state.
    fn default() -> Self {
        Self {
            eye: [0.0, 1.8, 5.0],
            target: [0.0, 1.5, 4.0],
            up: [0.0, 1.0, 0.0],
            fov_y_radians: 60.0_f32.to_radians(),
            near: 0.03,
            far: 5000.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderItemFlags {
    pub visible: bool,
    pub casts_shadow: bool,
}

#[derive(Clone, Debug)]
pub struct RenderItemPacket {
    pub object_id: Option<ExternalObjectId>,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub flags: RenderItemFlags,
    pub layer: u16,
}

impl RenderItemPacket {
    /// Creates one renderer-facing item extracted from user or ECS state.
    pub fn new(mesh: MeshHandle, material: MaterialHandle) -> Self {
        Self {
            object_id: None,
            mesh,
            material,
            flags: RenderItemFlags {
                visible: true,
                casts_shadow: true,
            },
            layer: 0,
        }
    }

    /// Attaches a stable debug id that is not an ECS entity.
    pub fn with_object_id(mut self, object_id: ExternalObjectId) -> Self {
        self.object_id = Some(object_id);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LightPacket {
    pub intensity: f32,
}

impl LightPacket {
    /// Creates a light packet and clamps invalid intensity to zero.
    pub fn new(intensity: f32) -> Self {
        let intensity = if intensity.is_finite() {
            intensity.max(0.0)
        } else {
            0.0
        };
        Self { intensity }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DebugDraw {
    pub show_bounds: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptimizationSettings {
    frustum_culling: bool,
    distance_lod: bool,
    high_detail_screen_radius_px: f32,
    medium_detail_screen_radius_px: f32,
}

impl RenderOptimizationSettings {
    const MIN_SCREEN_RADIUS_PX: f32 = 1.0;
    const MAX_SCREEN_RADIUS_PX: f32 = 4096.0;

    /// Creates per-frame renderer optimization policy after constraining LOD thresholds.
    ///
    /// Screen radius is measured from the projected mesh bounding sphere. Objects larger than the
    /// high threshold keep full geometry; objects below the medium threshold use the coarsest LOD.
    pub fn new(
        frustum_culling: bool,
        distance_lod: bool,
        high_detail_screen_radius_px: f32,
        medium_detail_screen_radius_px: f32,
    ) -> Option<Self> {
        let high_valid = (Self::MIN_SCREEN_RADIUS_PX..=Self::MAX_SCREEN_RADIUS_PX)
            .contains(&high_detail_screen_radius_px);
        let medium_valid = (Self::MIN_SCREEN_RADIUS_PX..=Self::MAX_SCREEN_RADIUS_PX)
            .contains(&medium_detail_screen_radius_px);

        (high_valid
            && medium_valid
            && high_detail_screen_radius_px > medium_detail_screen_radius_px)
            .then_some(Self {
                frustum_culling,
                distance_lod,
                high_detail_screen_radius_px,
                medium_detail_screen_radius_px,
            })
    }

    /// Creates the default policy used by window snapshots.
    pub fn balanced() -> Self {
        Self {
            frustum_culling: true,
            distance_lod: true,
            high_detail_screen_radius_px: 72.0,
            medium_detail_screen_radius_px: 16.0,
        }
    }

    /// Creates a policy that keeps all extracted mesh draws at full detail.
    pub fn disabled() -> Self {
        Self {
            frustum_culling: false,
            distance_lod: false,
            ..Self::balanced()
        }
    }

    /// Returns whether meshes outside the active camera frustum may be skipped.
    pub fn frustum_culling(self) -> bool {
        self.frustum_culling
    }

    /// Returns whether projected screen size may select a coarser index buffer.
    pub fn distance_lod(self) -> bool {
        self.distance_lod
    }

    /// Returns the projected radius that keeps a mesh at full geometry detail.
    pub fn high_detail_screen_radius_px(self) -> f32 {
        self.high_detail_screen_radius_px
    }

    /// Returns the projected radius below which the coarsest geometry LOD is used.
    pub fn medium_detail_screen_radius_px(self) -> f32 {
        self.medium_detail_screen_radius_px
    }
}

impl Default for RenderOptimizationSettings {
    /// Enables conservative mesh culling and screen-size LOD by default.
    fn default() -> Self {
        Self::balanced()
    }
}

#[derive(Clone, Debug)]
pub struct FrameSnapshot {
    pub frame_id: FrameId,
    pub surface_id: SurfaceId,
    pub surface_generation: SurfaceGeneration,
    pub scene: SceneHandle,
    pub views: Vec<ViewPacket>,
    pub lights: Vec<LightPacket>,
    pub render_items: Vec<RenderItemPacket>,
    pub camera_effects: CameraEffects,
    pub debug_draw: DebugDraw,
    pub optimization: RenderOptimizationSettings,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("frame snapshot needs at least one view")]
    MissingView,
}

pub struct FrameSnapshotBuilder {
    frame_id: FrameId,
    surface_id: SurfaceId,
    surface_generation: SurfaceGeneration,
    scene: SceneHandle,
    views: Vec<ViewPacket>,
    lights: Vec<LightPacket>,
    render_items: Vec<RenderItemPacket>,
    camera_effects: CameraEffects,
    debug_draw: DebugDraw,
    optimization: RenderOptimizationSettings,
}

impl FrameSnapshotBuilder {
    /// Starts a snapshot for one extracted frame and scene.
    pub fn new(
        frame_id: FrameId,
        scene: SceneHandle,
        surface_id: SurfaceId,
        surface_generation: SurfaceGeneration,
    ) -> Self {
        Self {
            frame_id,
            surface_id,
            surface_generation,
            scene,
            views: Vec::new(),
            lights: Vec::new(),
            render_items: Vec::new(),
            camera_effects: CameraEffects::default(),
            debug_draw: DebugDraw::default(),
            optimization: RenderOptimizationSettings::default(),
        }
    }

    /// Adds one validated view to the snapshot.
    pub fn add_view(&mut self, view: ViewPacket) -> &mut Self {
        self.views.push(view);
        self
    }

    /// Adds one light packet to the snapshot.
    pub fn add_light(&mut self, light: LightPacket) -> &mut Self {
        self.lights.push(light);
        self
    }

    /// Adds one renderer-facing item to the snapshot.
    pub fn add_render_item(&mut self, item: RenderItemPacket) -> &mut Self {
        self.render_items.push(item);
        self
    }

    /// Sets the validated exposure while preserving the rest of the camera look.
    pub fn set_exposure(&mut self, exposure: Exposure) -> &mut Self {
        self.camera_effects = self.camera_effects.with_exposure(exposure);
        self
    }

    /// Sets the validated camera response used by post processing.
    pub fn set_camera_effects(&mut self, camera_effects: CameraEffects) -> &mut Self {
        self.camera_effects = camera_effects;
        self
    }

    /// Sets debug draw flags copied into the frame snapshot.
    pub fn set_debug_draw(&mut self, debug_draw: DebugDraw) -> &mut Self {
        self.debug_draw = debug_draw;
        self
    }

    /// Sets per-frame renderer optimization policy copied into the snapshot.
    pub fn set_optimization(&mut self, optimization: RenderOptimizationSettings) -> &mut Self {
        self.optimization = optimization;
        self
    }

    /// Builds an owned snapshot and rejects missing view data.
    pub fn build(self) -> Result<FrameSnapshot, SnapshotError> {
        if self.views.is_empty() {
            return Err(SnapshotError::MissingView);
        }

        Ok(FrameSnapshot {
            frame_id: self.frame_id,
            surface_id: self.surface_id,
            surface_generation: self.surface_generation,
            scene: self.scene,
            views: self.views,
            lights: self.lights,
            render_items: self.render_items,
            camera_effects: self.camera_effects,
            debug_draw: self.debug_draw,
            optimization: self.optimization,
        })
    }
}

/// Multiplies two column-major 4x4 matrices.
fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = a[row] * b[column * 4]
                + a[4 + row] * b[column * 4 + 1]
                + a[8 + row] * b[column * 4 + 2]
                + a[12 + row] * b[column * 4 + 3];
        }
    }
    out
}

/// Builds a Vulkan right-handed perspective matrix with NDC depth in 0..1.
fn perspective_vulkan(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let safe_aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    };
    let f = 1.0 / (fov_y * 0.5).tan();
    let z = far / (near - far);

    [
        f / safe_aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        -f,
        0.0,
        0.0,
        0.0,
        0.0,
        z,
        -1.0,
        0.0,
        0.0,
        z * near,
        0.0,
    ]
}

/// Builds a right-handed view matrix from explicit camera basis vectors.
fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let forward = normalize_or(sub3(target, eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);

    [
        right[0],
        up[0],
        -forward[0],
        0.0,
        right[1],
        up[1],
        -forward[1],
        0.0,
        right[2],
        up[2],
        -forward[2],
        0.0,
        -dot3(right, eye),
        -dot3(up, eye),
        dot3(forward, eye),
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that a frame snapshot cannot be built without a render view.
    #[test]
    fn snapshot_requires_view() {
        let frame = FrameId::from_raw(1).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene handle is non-zero");
        let surface = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");

        let result = FrameSnapshotBuilder::new(frame, scene, surface, generation).build();

        assert!(matches!(result, Err(SnapshotError::MissingView)));
    }

    // Verifies that camera effect values crossing the renderer boundary are constrained.
    #[test]
    fn camera_effects_reject_invalid_white_balance() {
        let exposure = Exposure::default();

        assert!(CameraEffects::new(exposure, [1.0, 1.2, 0.8]).is_some());
        assert!(CameraEffects::new(exposure, [1.0, f32::NAN, 0.8]).is_none());
        assert!(CameraEffects::new(exposure, [1.0, 8.0, 0.8]).is_none());
    }

    // Verifies that LOD thresholds cannot be inverted across the protocol boundary.
    #[test]
    fn optimization_rejects_inverted_lod_thresholds() {
        assert!(RenderOptimizationSettings::new(true, true, 96.0, 28.0).is_some());
        assert!(RenderOptimizationSettings::new(true, true, 20.0, 40.0).is_none());
        assert!(RenderOptimizationSettings::new(true, true, f32::NAN, 20.0).is_none());
    }
}
