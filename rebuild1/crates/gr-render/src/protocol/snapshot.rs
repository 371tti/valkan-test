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
    pub direction: [f32; 3],
    pub color: [f32; 3],
}

impl LightPacket {
    /// Creates a light packet and clamps invalid intensity to zero.
    pub fn new(intensity: f32) -> Self {
        let intensity = if intensity.is_finite() {
            intensity.max(0.0)
        } else {
            0.0
        };
        Self {
            intensity,
            direction: [0.45, -1.0, 0.25],
            color: [3.00, 2.65, 2.15],
        }
    }

    /// Replaces the app-owned direction and linear RGB color.
    pub fn with_direction_and_color(mut self, direction: [f32; 3], color: [f32; 3]) -> Self {
        if direction.iter().all(|component| component.is_finite())
            && direction
                .iter()
                .any(|component| component.abs() > f32::EPSILON)
        {
            self.direction = direction;
        }
        if color.iter().all(|component| component.is_finite()) {
            self.color = color.map(|component| component.max(0.0));
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalLightKind {
    Point,
    Sphere,
    Spot,
    Rectangle,
}

impl LocalLightKind {
    /// Returns the compact shader code copied into local-light uniforms.
    pub(crate) fn shader_code(self) -> f32 {
        match self {
            Self::Point => 0.0,
            Self::Sphere => 1.0,
            Self::Spot => 2.0,
            Self::Rectangle => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalLightPacket {
    pub kind: LocalLightKind,
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub direction: [f32; 3],
    pub source_radius: f32,
    pub half_size: [f32; 2],
    pub casts_shadow: bool,
}

impl LocalLightPacket {
    const MIN_RANGE: f32 = 0.01;
    const MAX_RANGE: f32 = 4096.0;
    const MAX_INTENSITY: f32 = 2048.0;

    /// Creates a bounded point light with optional analytic local shadows.
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Option<Self> {
        Self::new(
            LocalLightKind::Point,
            position,
            color,
            intensity,
            range,
            [0.0, -1.0, 0.0],
            0.0,
            [0.0, 0.0],
            true,
        )
    }

    /// Creates a bounded sphere light whose radius softens highlights and shadows.
    pub fn sphere(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        source_radius: f32,
    ) -> Option<Self> {
        Self::new(
            LocalLightKind::Sphere,
            position,
            color,
            intensity,
            range,
            [0.0, -1.0, 0.0],
            source_radius,
            [0.0, 0.0],
            true,
        )
    }

    /// Creates a bounded spotlight facing along `direction`.
    pub fn spot(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        direction: [f32; 3],
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Option<Self> {
        const MIN_OUTER_CONE: f32 = 0.01;
        const MAX_OUTER_CONE: f32 = std::f32::consts::FRAC_PI_2;
        const MIN_CONE_GAP: f32 = 0.001;

        let outer_cone_angle = finite_clamp(
            outer_cone_angle,
            MIN_OUTER_CONE,
            MAX_OUTER_CONE,
            std::f32::consts::FRAC_PI_4,
        );
        let inner_cone_angle = finite_clamp(
            inner_cone_angle,
            0.0,
            (outer_cone_angle - MIN_CONE_GAP).max(0.0),
            0.0,
        );

        Self::new(
            LocalLightKind::Spot,
            position,
            color,
            intensity,
            range,
            direction,
            0.0,
            [inner_cone_angle.cos(), outer_cone_angle.cos()],
            true,
        )
    }

    /// Creates a bounded rectangular area light facing along `direction`.
    pub fn rectangle(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        direction: [f32; 3],
        half_size: [f32; 2],
    ) -> Option<Self> {
        let source_radius = (half_size[0] * half_size[0] + half_size[1] * half_size[1]).sqrt();
        Self::new(
            LocalLightKind::Rectangle,
            position,
            color,
            intensity,
            range,
            direction,
            source_radius,
            half_size,
            true,
        )
    }

    /// Returns a copy with analytic local shadows toggled.
    pub fn with_shadow(mut self, casts_shadow: bool) -> Self {
        self.casts_shadow = casts_shadow;
        self
    }

    /// Builds one validated local-light packet before it crosses the renderer boundary.
    fn new(
        kind: LocalLightKind,
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        direction: [f32; 3],
        source_radius: f32,
        half_size: [f32; 2],
        casts_shadow: bool,
    ) -> Option<Self> {
        let finite_position = position.iter().all(|value| value.is_finite());
        let finite_color = color.iter().all(|value| value.is_finite() && *value >= 0.0);
        let finite_direction = direction.iter().all(|value| value.is_finite());
        let finite_half_size = half_size
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0);
        let direction = normalize_or(direction, [0.0, -1.0, 0.0]);
        let range = finite_clamp(range, Self::MIN_RANGE, Self::MAX_RANGE, 12.0);
        let intensity = finite_clamp(intensity, 0.0, Self::MAX_INTENSITY, 0.0);
        let source_radius = finite_clamp(source_radius, 0.0, range, 0.0);

        (finite_position && finite_color && finite_direction && finite_half_size).then_some(Self {
            kind,
            position,
            color,
            intensity,
            range,
            direction,
            source_radius,
            half_size,
            casts_shadow,
        })
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DebugViewMode {
    #[default]
    Disabled = 0,
    Normals = 1,
    Depth = 2,
    PcssFilterRadius = 3,
    /// Shows the SMAA edge target produced by pass 1 (R=west, G=north).
    SmaaEdges = 5,
    /// Shows the SMAA blend target produced by pass 2 (R=horizontal, G=vertical, B=sum).
    SmaaWeights = 6,
    /// Shows the raw GTAO visibility term before the post material multiplier.
    Ssao = 7,
    /// Shows the raw directional PCSS visibility before the scene's temporal history.
    PcssVisibilityRaw = 14,
    /// Shows the previous PCSS visibility history buffer sampled at the current raster pixel.
    PcssHistory = 15,
}

impl DebugViewMode {
    pub const COUNT: u32 = 16;

    pub fn next(self) -> Self {
        match self {
            Self::Disabled => Self::Normals,
            Self::Normals => Self::Depth,
            Self::Depth => Self::PcssFilterRadius,
            Self::PcssFilterRadius => Self::SmaaEdges,
            Self::SmaaEdges => Self::SmaaWeights,
            Self::SmaaWeights => Self::Ssao,
            Self::Ssao => Self::PcssVisibilityRaw,
            Self::PcssVisibilityRaw => Self::PcssHistory,
            Self::PcssHistory => Self::Disabled,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Disabled => "scene",
            Self::Normals => "normals",
            Self::Depth => "linear_depth",
            Self::PcssFilterRadius => "pcss_filter_radius",
            Self::SmaaEdges => "smaa_edges",
            Self::SmaaWeights => "smaa_weights",
            Self::Ssao => "ssao_gtao",
            Self::PcssVisibilityRaw => "pcss_visibility_raw",
            Self::PcssHistory => "pcss_history",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DebugDraw {
    pub show_bounds: bool,
    pub view_mode: DebugViewMode,
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
            high_detail_screen_radius_px: 128.0,
            medium_detail_screen_radius_px: 36.0,
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
    pub local_lights: Vec<LocalLightPacket>,
    pub render_items: Vec<RenderItemPacket>,
    pub camera_effects: CameraEffects,
    pub debug_draw: DebugDraw,
    pub optimization: RenderOptimizationSettings,
    /// Target presentation rate used to convert time-based temporal filters into per-frame rates.
    /// The app-side window pacing populates this from its configured frame interval.
    pub frame_rate_hz: f32,
}

const DEFAULT_FRAME_RATE_HZ: f32 = 120.0;
const MIN_FRAME_RATE_HZ: f32 = 15.0;
const MAX_FRAME_RATE_HZ: f32 = 240.0;

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
    local_lights: Vec<LocalLightPacket>,
    render_items: Vec<RenderItemPacket>,
    camera_effects: CameraEffects,
    debug_draw: DebugDraw,
    optimization: RenderOptimizationSettings,
    frame_rate_hz: f32,
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
            local_lights: Vec::new(),
            render_items: Vec::new(),
            camera_effects: CameraEffects::default(),
            debug_draw: DebugDraw::default(),
            optimization: RenderOptimizationSettings::default(),
            frame_rate_hz: DEFAULT_FRAME_RATE_HZ,
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

    /// Adds one local point, sphere, or area light to the snapshot.
    pub fn add_local_light(&mut self, light: LocalLightPacket) -> &mut Self {
        self.local_lights.push(light);
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

    /// Sets the configured presentation rate used by frame-rate-aware temporal filters.
    ///
    /// Invalid values are ignored and values outside the supported window pacing range are
    /// clamped, so a malformed app setting cannot produce a zero or non-finite decay interval.
    pub fn set_frame_rate_hz(&mut self, frame_rate_hz: f32) -> &mut Self {
        self.frame_rate_hz = finite_clamp(
            frame_rate_hz,
            MIN_FRAME_RATE_HZ,
            MAX_FRAME_RATE_HZ,
            DEFAULT_FRAME_RATE_HZ,
        );
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
            local_lights: self.local_lights,
            render_items: self.render_items,
            camera_effects: self.camera_effects,
            debug_draw: self.debug_draw,
            optimization: self.optimization,
            frame_rate_hz: self.frame_rate_hz,
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

/// Clamps finite values and substitutes a fallback for invalid inputs.
fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
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

    // Verifies that local area lights survive snapshot building as renderer-owned data.
    #[test]
    fn snapshot_carries_local_area_lights() {
        let frame = FrameId::from_raw(1).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene handle is non-zero");
        let surface = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let extent = NonZeroExtent::new(640, 360).expect("test extent is non-zero");
        let light = LocalLightPacket::rectangle(
            [1.0, 2.0, 3.0],
            [1.0, 0.7, 0.4],
            3.0,
            18.0,
            [0.0, -1.0, 0.0],
            [2.0, 0.5],
        )
        .expect("area light should be valid");
        let mut builder = FrameSnapshotBuilder::new(frame, scene, surface, generation);

        builder
            .add_view(ViewPacket::new(view, extent))
            .add_local_light(light);
        let snapshot = builder.build().expect("snapshot has one view");

        assert_eq!(snapshot.local_lights, vec![light]);
        assert_eq!(snapshot.local_lights[0].kind, LocalLightKind::Rectangle);
        assert_eq!(snapshot.local_lights[0].source_radius, (4.25_f32).sqrt());
    }

    #[test]
    fn snapshot_carries_configured_frame_rate_for_temporal_filters() {
        let frame = FrameId::from_raw(1).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene id is non-zero");
        let surface = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let extent = NonZeroExtent::new(320, 180).expect("test extent is non-zero");

        let mut builder = FrameSnapshotBuilder::new(frame, scene, surface, generation);
        builder
            .add_view(ViewPacket::new(view, extent))
            .set_frame_rate_hz(60.0);
        let snapshot = builder.build().expect("snapshot has one view");

        assert_eq!(snapshot.frame_rate_hz, 60.0);
    }

    // Verifies that spotlights keep their direction and cone falloff values for shaders.
    #[test]
    fn local_spot_light_packs_cone_cosines() {
        let light = LocalLightPacket::spot(
            [0.0, 1.0, 2.0],
            [1.0, 0.9, 0.7],
            2.0,
            24.0,
            [0.0, -2.0, 0.0],
            0.2,
            0.6,
        )
        .expect("spot light should be valid");

        assert_eq!(light.kind, LocalLightKind::Spot);
        assert_eq!(light.direction, [0.0, -1.0, 0.0]);
        assert!(light.half_size[0] > light.half_size[1]);
        assert!((light.half_size[0] - 0.2_f32.cos()).abs() < 0.0001);
        assert!((light.half_size[1] - 0.6_f32.cos()).abs() < 0.0001);
    }

    // Verifies that invalid local-light values are rejected at the protocol edge.
    #[test]
    fn local_light_rejects_invalid_values() {
        assert!(LocalLightPacket::point([f32::NAN, 0.0, 0.0], [1.0; 3], 1.0, 4.0).is_none());
        assert!(LocalLightPacket::point([0.0; 3], [-1.0, 0.0, 0.0], 1.0, 4.0).is_none());
        assert!(
            LocalLightPacket::rectangle(
                [0.0; 3],
                [1.0; 3],
                1.0,
                4.0,
                [0.0, -1.0, 0.0],
                [1.0, f32::NAN],
            )
            .is_none()
        );
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

    // Verifies that the default snapshot policy keeps LOD active with meaningful cutoffs.
    #[test]
    fn balanced_optimization_enables_aggressive_distance_lod() {
        let optimization = RenderOptimizationSettings::balanced();

        assert!(optimization.frustum_culling());
        assert!(optimization.distance_lod());
        assert_eq!(optimization.high_detail_screen_radius_px(), 128.0);
        assert_eq!(optimization.medium_detail_screen_radius_px(), 36.0);
    }
}
