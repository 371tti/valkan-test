use ash::vk;

use crate::{
    math::{add3, cross3, dot3, identity_mat4, mul3, normalize_or, sub3},
    protocol::{CameraSnapshot, LightPacket, RenderItemPacket, StableCsmPcssQualitySettings},
    renderer::{
        DEFAULT_AMBIENT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_DIR, DEFAULT_SHADOW_CASCADE_METRICS,
        DEFAULT_SHADOW_CASCADE_SPLITS, graph::SHADOW_CASCADE_COUNT, shadow_map_size,
    },
};

#[cfg(test)]
use crate::protocol::SceneBounds;

use super::mesh::{
    EmissiveLightUniforms, LOCAL_SHADOW_FACE_COUNT, LOCAL_SHADOW_MATRIX_COUNT, MAX_LOCAL_LIGHTS,
    MeshFrameUniform, ShadowCascadeCull,
};

const SHADOW_SPLIT_LAMBDA: f32 = 0.78;
const SHADOW_SPLIT_NEAR_FLOOR: f32 = 1.0;
const SHADOW_RADIUS_PADDING: f32 = 1.08;
const SHADOW_MIN_RADIUS: f32 = 4.0;
const SHADOW_DEPTH_PADDING: f32 = 24.0;
const SHADOW_CASTER_DEPTH_RESERVE_MAX: f32 = 640.0;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LocalShadowFrameSignature {
    caster_hash: u64,
    light_hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ShadowFrameData {
    pub(super) view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
    pub(super) coverage_near: f32,
    pub(super) splits: [f32; 4],
    pub(super) texel_world: [f32; 4],
    pub(super) depth_span: [f32; 4],
    pub(super) cascade_resolution: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LocalShadowFrameData {
    pub(super) view_proj: [[f32; 16]; LOCAL_SHADOW_MATRIX_COUNT],
    pub(super) params: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) lights: [Option<LocalShadowLightData>; MAX_LOCAL_LIGHTS],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LocalShadowLightData {
    pub(super) light_index: usize,
    pub(super) light_position_radius: [f32; 4],
    pub(super) source_radius: f32,
}

/// Precomputed light-space basis for one local cubemap face.
///
/// Face culling is performed for every shadow caster. Keeping the invariant light/range values and
/// the six face bases here avoids rebuilding the same cross products and near-plane clamp inside
/// the inner item loop.
#[derive(Clone, Copy)]
pub(super) struct LocalShadowFaceCull {
    light_position: [f32; 3],
    range: f32,
    near: f32,
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

#[derive(Clone, Copy)]
struct ShadowCascadeProjection {
    view_projection: [f32; 16],
    texel_world: f32,
    depth_span: f32,
}

/// Direction-independent receiver sphere reused by every light sample in one cascade.
#[derive(Clone, Copy)]
struct ShadowCascadeBounds {
    receiver_points: [[f32; 3]; 8],
    center: [f32; 3],
    radius: f32,
    texel_world: f32,
    snap_alignment_texels: f32,
    shadow_resolution: f32,
    caster_depth_reserve: f32,
    cascade_near: f32,
    cascade_far: f32,
    cascade_index: usize,
}

/// Camera basis and frustum scale shared by all four cascade fits in one frame.
///
/// The previous path rebuilt the normalized basis and tangent-of-FOV values once per cascade even
/// though they are camera invariants. Keeping them in a small Copy value removes twelve cross/
/// normalize operations and repeated trigonometry without changing any fitted bounds.
#[derive(Clone, Copy)]
struct CameraFrustumBasis {
    eye: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    tan_x: f32,
    tan_y: f32,
}

#[derive(Clone, Copy)]
struct StableLightBasis {
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

/// Builds a stable signature for persistent local-light cubemap shadows.
///
/// Local shadows depend only on the opaque caster set and the light geometry. Camera movement and
/// light color changes do not alter the depth cubemaps, so neither should force six-face redraws.
pub(super) fn local_shadow_frame_signature(
    items: &[RenderItemPacket],
    emissive_lights: &EmissiveLightUniforms,
) -> LocalShadowFrameSignature {
    let light_count =
        (emissive_lights.count[0] as usize).min(emissive_lights.position_radius.len());
    let mut light_hash = 0xcbf2_9ce4_8422_2325_u64;
    light_hash = fnv1a(light_hash, light_count as u64);
    for light_index in 0..light_count {
        if emissive_lights.size_kind[light_index][3] <= 0.0 {
            continue;
        }
        light_hash = fnv1a(light_hash, light_index as u64);
        light_hash = hash_float4(light_hash, emissive_lights.position_radius[light_index]);
        light_hash = hash_float4(light_hash, emissive_lights.direction_radius[light_index]);
        light_hash = hash_float4(light_hash, emissive_lights.size_kind[light_index]);
    }

    LocalShadowFrameSignature {
        caster_hash: shadow_caster_hash(items),
        light_hash,
    }
}

/// Builds the stable CSM using the actual shared resolution of the allocated depth array.
///
/// Each cascade is fitted to a camera-frustum bounding sphere whose radius remains stable while
/// the camera moves inside the sphere. The light-space center is then snapped to that cascade's
/// texel grid. No per-frame direction jitter or shadow-history state participates in this path.
pub(super) fn stable_csm_frame_data_for_resolution(
    camera: CameraSnapshot,
    extent: vk::Extent2D,
    light: LightPacket,
    shadow_resolution: u32,
) -> ShadowFrameData {
    let shadow_resolution = shadow_resolution.max(1);
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let coverage_near = camera.near.max(0.03);
    let coverage_far = shadow_coverage_distance(camera);
    let splits = shadow_cascade_splits(camera);
    let cascade_bounds = stable_csm_cascade_bounds_for_splits(
        camera,
        aspect,
        coverage_near,
        splits,
        shadow_resolution as f32,
    );
    let light_basis = stable_light_basis_value(light.direction);
    let mut view_proj = [identity_mat4(); SHADOW_CASCADE_COUNT];
    let mut texel_world = [0.0; SHADOW_CASCADE_COUNT];
    let mut depth_span = [0.0; SHADOW_CASCADE_COUNT];

    for cascade_index in 0..SHADOW_CASCADE_COUNT {
        let projection =
            shadow_view_projection_from_bounds(cascade_bounds[cascade_index], light_basis);
        view_proj[cascade_index] = projection.view_projection;
        texel_world[cascade_index] = projection.texel_world;
        depth_span[cascade_index] = projection.depth_span;
    }

    tracing::trace!(
        coverage_near,
        coverage_far,
        "built stable CSM receiver coverage"
    );

    ShadowFrameData {
        view_proj,
        coverage_near,
        splits,
        texel_world,
        depth_span,
        cascade_resolution: [shadow_resolution as f32; SHADOW_CASCADE_COUNT],
    }
}

/// Returns the conservative camera-depth culling window for one shadow cascade.
pub(super) fn shadow_cascade_cull(
    camera: CameraSnapshot,
    cascade_index: usize,
    shadow_data: &ShadowFrameData,
) -> ShadowCascadeCull {
    let (min_depth, max_depth) = shadow_cascade_depth_range(shadow_data, cascade_index);

    ShadowCascadeCull::new(camera, min_depth, max_depth).with_light_space_projection(
        shadow_data.view_proj[cascade_index.min(SHADOW_CASCADE_COUNT - 1)],
    )
}

/// Builds the frame uniform consumed by mesh vertex and fragment shaders.
#[cfg(test)]
pub(super) fn mesh_frame_uniform_for_frame(
    camera: CameraSnapshot,
    light_intensity: f32,
    extent: vk::Extent2D,
    has_shadow_casters: bool,
    translucent_shadow_cascades: [bool; SHADOW_CASCADE_COUNT],
    shadow_data: Option<ShadowFrameData>,
    stable_quality: StableCsmPcssQualitySettings,
    local_shadow_data: Option<LocalShadowFrameData>,
    emissive_lights: EmissiveLightUniforms,
    debug_view_mode: crate::protocol::DebugViewMode,
) -> MeshFrameUniform {
    mesh_frame_uniform_for_light(
        camera,
        LightPacket::new(light_intensity),
        extent,
        has_shadow_casters,
        translucent_shadow_cascades,
        shadow_data,
        stable_quality,
        local_shadow_data,
        emissive_lights,
        debug_view_mode,
    )
}

pub(super) fn mesh_frame_uniform_for_light(
    camera: CameraSnapshot,
    light: LightPacket,
    extent: vk::Extent2D,
    has_shadow_casters: bool,
    translucent_shadow_cascades: [bool; SHADOW_CASCADE_COUNT],
    shadow_data: Option<ShadowFrameData>,
    stable_quality: StableCsmPcssQualitySettings,
    local_shadow_data: Option<LocalShadowFrameData>,
    emissive_lights: EmissiveLightUniforms,
    debug_view_mode: crate::protocol::DebugViewMode,
) -> MeshFrameUniform {
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let light_dir = normalize_or(light.direction, DEFAULT_DIRECTIONAL_LIGHT_DIR);
    let shadow_data = shadow_data.unwrap_or_else(disabled_shadow_frame_data);
    let local_shadow_data = local_shadow_data.unwrap_or_else(disabled_local_shadow_frame_data);
    let shadow_view_proj = std::array::from_fn(|matrix_index| {
        if matrix_index < SHADOW_CASCADE_COUNT {
            shadow_data.view_proj[matrix_index]
        } else {
            identity_mat4()
        }
    });

    MeshFrameUniform {
        view_proj: camera.view_projection(aspect),
        view: look_at_rh(camera.eye, camera.target, camera.up),
        shadow_view_proj,
        stable_csm_pcss_params: [
            stable_quality.blocker_search_samples() as f32,
            stable_quality.filter_samples() as f32,
            stable_quality.light_angular_radius_radians(),
            f32::from(stable_quality.contact_shadows()),
        ],
        stable_csm_receiver_params: [
            stable_quality.receiver_bias_scale(),
            stable_quality.slope_bias_scale(),
            stable_quality.normal_offset_scale(),
            stable_quality.receiver_plane_bias_scale(),
        ],
        shadow_cascade_splits: shadow_data.splits,
        shadow_cascade_texel_world: shadow_data.texel_world,
        shadow_cascade_depth_span: shadow_data.depth_span,
        shadow_cascade_resolution: shadow_data.cascade_resolution,
        camera_pos: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
        light_dir: [
            light_dir[0],
            light_dir[1],
            light_dir[2],
            if has_shadow_casters { 1.0 } else { 0.0 },
        ],
        light_color: [
            light.color[0] * light.intensity,
            light.color[1] * light.intensity,
            light.color[2] * light.intensity,
            translucent_shadow_cascade_bits(translucent_shadow_cascades),
        ],
        ambient_color: ambient_color_for_light(light),
        local_shadow_view_proj: local_shadow_data.view_proj,
        local_shadow_params: local_shadow_data.params,
        emissive_light_position_radius: emissive_lights.position_radius,
        emissive_light_color: emissive_lights.color,
        emissive_light_direction_radius: emissive_lights.direction_radius,
        emissive_light_size_kind: emissive_lights.size_kind,
        emissive_light_count: emissive_lights.count,
        debug_view: [debug_view_mode as u32 as f32, 0.0, 0.0, 0.0],
    }
}

/// Dims the fixed sky/hemisphere contribution with the app-owned global light.
///
/// A small floor preserves silhouettes when the key light is essentially off, while the smooth
/// ramp prevents moonlight and dawn from inheriting the full daytime environment brightness.
fn ambient_color_for_light(light: LightPacket) -> [f32; 4] {
    let intensity = if light.intensity.is_finite() {
        light.intensity.max(0.0)
    } else {
        0.0
    };
    let t = ((intensity - 0.08) / (0.85 - 0.08)).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    let scale = 0.008 + 0.542 * smooth;

    DEFAULT_AMBIENT_COLOR.map(|component| component * scale)
}

/// Packs cascade-local translucent-map availability into the exactly representable low 4 bits.
fn translucent_shadow_cascade_bits(cascades: [bool; SHADOW_CASCADE_COUNT]) -> f32 {
    cascades
        .into_iter()
        .enumerate()
        .fold(0_u32, |bits, (index, enabled)| {
            bits | (u32::from(enabled) << index)
        }) as f32
}

/// Returns whether any local light should receive a cubemap shadow this frame.
pub(super) fn has_local_shadow_light(emissive_lights: &EmissiveLightUniforms) -> bool {
    let light_count =
        (emissive_lights.count[0] as usize).min(emissive_lights.position_radius.len());
    (0..light_count).any(|index| emissive_lights.size_kind[index][3] > 0.0)
}

/// Builds point-light projections for every local light that has cubemap shadows enabled.
pub(super) fn local_shadow_frame_data(
    emissive_lights: &EmissiveLightUniforms,
    extent: vk::Extent2D,
) -> Option<LocalShadowFrameData> {
    let light_count =
        (emissive_lights.count[0] as usize).min(emissive_lights.position_radius.len());
    let face_resolution = extent.width.min(extent.height).max(1) as f32;
    let filter_texel_angle = 1.5 / face_resolution;
    let mut view_proj = [identity_mat4(); LOCAL_SHADOW_MATRIX_COUNT];
    let mut params = [[0.0, 0.0, 1.0, 1.0]; MAX_LOCAL_LIGHTS];
    let mut lights = [None; MAX_LOCAL_LIGHTS];
    let mut enabled_count = 0usize;

    for light_index in 0..light_count {
        if emissive_lights.size_kind[light_index][3] <= 0.0 {
            continue;
        }

        let position_radius = emissive_lights.position_radius[light_index];
        let direction_radius = emissive_lights.direction_radius[light_index];
        let size_kind = emissive_lights.size_kind[light_index];
        let position = [position_radius[0], position_radius[1], position_radius[2]];
        let range = position_radius[3].max(0.25);
        let source_radius = direction_radius[3]
            .max(size_kind[0].abs().max(size_kind[1].abs()))
            .max(0.03);
        let near = (range * 0.005)
            .max(source_radius * 0.05)
            .clamp(0.03, range * 0.05);
        let far = range.max(near + 1.0);
        let matrix_offset = light_index * LOCAL_SHADOW_FACE_COUNT;
        let matrices = local_shadow_cube_view_projections(position, near, far);
        view_proj[matrix_offset..matrix_offset + LOCAL_SHADOW_FACE_COUNT]
            .copy_from_slice(&matrices);
        params[light_index] = [1.0, filter_texel_angle, near, far];
        lights[light_index] = Some(LocalShadowLightData {
            light_index,
            light_position_radius: position_radius,
            source_radius,
        });
        enabled_count += 1;
    }

    if enabled_count == 0 {
        return None;
    }

    Some(LocalShadowFrameData {
        view_proj,
        params,
        lights,
    })
}

/// Returns whether one mesh sphere overlaps one 90-degree local-shadow cubemap face.
///
/// Testing the six face frusta independently prevents every in-range caster from being submitted
/// six times. The sphere/plane test is conservative, so meshes crossing an edge are retained in
/// both neighboring faces and cannot create missing-shadow seams.
#[cfg(test)]
pub(super) fn local_shadow_face_contains_bounds(
    light: LocalShadowLightData,
    face_index: usize,
    bounds: SceneBounds,
) -> bool {
    let Some(face) = local_shadow_face_culls(light).get(face_index).copied() else {
        return false;
    };
    local_shadow_face_contains_bounds_cached(face, bounds.center(), bounds.radius())
}

/// Builds the six invariant face tests for one local shadow light.
pub(super) fn local_shadow_face_culls(
    light: LocalShadowLightData,
) -> [LocalShadowFaceCull; LOCAL_SHADOW_FACE_COUNT] {
    const FACES: [([f32; 3], [f32; 3]); LOCAL_SHADOW_FACE_COUNT] = [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ];
    let light_position = [
        light.light_position_radius[0],
        light.light_position_radius[1],
        light.light_position_radius[2],
    ];
    let range = light.light_position_radius[3].max(0.0);
    let source_radius = light.source_radius.max(0.0);
    let near = (range * 0.005)
        .max(source_radius * 0.05)
        .clamp(0.03, range * 0.05);

    std::array::from_fn(|face_index| {
        let (forward, up_hint) = FACES[face_index];
        let right = normalize_or(cross3(forward, up_hint), [1.0, 0.0, 0.0]);
        let up = cross3(right, forward);
        LocalShadowFaceCull {
            light_position,
            range,
            near,
            forward,
            right,
            up,
        }
    })
}

/// Tests one resolved mesh sphere against a precomputed local cubemap face.
pub(super) fn local_shadow_face_contains_bounds_cached(
    face: LocalShadowFaceCull,
    center: [f32; 3],
    radius: f32,
) -> bool {
    let delta = sub3(center, face.light_position);
    let depth = dot3(delta, face.forward);
    if depth + radius < face.near || depth - radius > face.range {
        return false;
    }

    let plane_radius = radius * std::f32::consts::SQRT_2;
    dot3(delta, face.right).abs() <= depth + plane_radius
        && dot3(delta, face.up).abs() <= depth + plane_radius
}

fn local_shadow_cube_view_projections(
    position: [f32; 3],
    near: f32,
    far: f32,
) -> [[f32; 16]; LOCAL_SHADOW_FACE_COUNT] {
    const FACES: [([f32; 3], [f32; 3]); LOCAL_SHADOW_FACE_COUNT] = [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ];
    let projection = perspective_cubemap(std::f32::consts::FRAC_PI_2, 1.0, near, far);

    std::array::from_fn(|index| {
        let (direction, up) = FACES[index];
        let view = look_at_rh(position, add3(position, direction), up);
        mat4_mul(projection, view)
    })
}

/// Returns the full camera-visible shadow distance.
///
/// Near-cascade density is bounded separately in `shadow_cascade_splits`; truncating the overall
/// range here made large framed models lose directional shadows beyond an arbitrary world scale.
pub(super) fn shadow_coverage_distance(camera: CameraSnapshot) -> f32 {
    let near = camera.near.max(0.03);

    camera.far.max(near + SHADOW_CASCADE_COUNT as f32)
}

/// Returns the split distances used by the fixed cascade count.
pub(super) fn shadow_cascade_splits(camera: CameraSnapshot) -> [f32; 4] {
    const NEAR_CASCADE_DENSITY_DISTANCE: f32 = 320.0;
    let near = camera.near.max(0.03);
    let coverage_far = shadow_coverage_distance(camera);
    let dense_far = coverage_far
        .min(NEAR_CASCADE_DENSITY_DISTANCE)
        .max(near + SHADOW_CASCADE_COUNT as f32);

    shadow_cascade_splits_with_density_range(near, dense_far, coverage_far)
}

/// Builds fixed-count cascade endpoints with independently configurable dense and total ranges.
fn shadow_cascade_splits_with_density_range(
    coverage_near: f32,
    dense_far: f32,
    coverage_far: f32,
) -> [f32; 4] {
    let near = coverage_near.max(0.03);
    let far = coverage_far.max(near + SHADOW_CASCADE_COUNT as f32);
    let dense_far = dense_far.clamp(near + SHADOW_CASCADE_COUNT as f32, far);
    let range = (dense_far - near).max(1.0);
    let split_near = near.max(SHADOW_SPLIT_NEAR_FLOOR);
    let ratio = (dense_far / split_near).max(1.0);
    let mut previous = near;

    std::array::from_fn(|index| {
        let t = (index + 1) as f32 / SHADOW_CASCADE_COUNT as f32;
        let uniform = near + range * t;
        let logarithmic = split_near * ratio.powf(t);
        let split = logarithmic * SHADOW_SPLIT_LAMBDA + uniform * (1.0 - SHADOW_SPLIT_LAMBDA);
        let split = if index + 1 == SHADOW_CASCADE_COUNT {
            far
        } else {
            split.clamp(
                previous + 1.0,
                dense_far - (SHADOW_CASCADE_COUNT - index - 1) as f32,
            )
        };
        previous = split;
        split
    })
}

fn shadow_cascade_bounds_with_sampling(
    frustum_basis: CameraFrustumBasis,
    cascade_near: f32,
    cascade_far: f32,
    cascade_index: usize,
    shadow_resolution: f32,
    snap_alignment_texels: f32,
) -> ShadowCascadeBounds {
    let receiver_points = camera_frustum_corners(frustum_basis, cascade_near, cascade_far);
    let (center, radius) = bounding_sphere(&receiver_points);
    let shadow_resolution = shadow_resolution.max(1.0);
    let radius = (radius * SHADOW_RADIUS_PADDING).max(SHADOW_MIN_RADIUS);
    let texel_world = radius * 2.0 / shadow_resolution;
    let caster_depth_reserve =
        shadow_caster_depth_reserve(radius, cascade_near, cascade_far, cascade_index);

    ShadowCascadeBounds {
        receiver_points,
        center,
        radius,
        texel_world,
        snap_alignment_texels,
        shadow_resolution,
        caster_depth_reserve,
        cascade_near,
        cascade_far,
        cascade_index,
    }
}

/// Orients and texel-snaps a fixed cascade sphere for one sampled light direction.
fn shadow_view_projection_from_bounds(
    bounds: ShadowCascadeBounds,
    light_basis: StableLightBasis,
) -> ShadowCascadeProjection {
    let ShadowCascadeBounds {
        receiver_points,
        center,
        radius,
        texel_world,
        snap_alignment_texels,
        shadow_resolution,
        caster_depth_reserve,
        cascade_near,
        cascade_far,
        cascade_index,
    } = bounds;
    let center =
        snap_shadow_center_with_basis(light_basis, center, texel_world * snap_alignment_texels);
    let view = stable_light_view_with_basis(light_basis, center, radius, caster_depth_reserve);

    let (near, far) = shadow_depth_range(
        &view,
        radius,
        &receiver_points,
        center,
        caster_depth_reserve,
    );
    let depth_span = (far - near).max(1.0);

    tracing::trace!(
        cascade_index,
        cascade_near,
        cascade_far,
        radius,
        texel_world,
        snap_alignment_texels,
        shadow_resolution,
        caster_depth_reserve,
        near,
        far,
        "built camera-cascade shadow projection"
    );

    ShadowCascadeProjection {
        view_projection: mat4_mul(
            orthographic_vulkan(radius * 2.0, radius * 2.0, near, far),
            view,
        ),
        texel_world,
        depth_span,
    }
}

/// Returns the camera-space near/far depths that feed one already-built cascade projection.
fn shadow_cascade_depth_range(shadow_data: &ShadowFrameData, cascade_index: usize) -> (f32, f32) {
    let index = cascade_index.min(SHADOW_CASCADE_COUNT - 1);
    let min_depth = match cascade_index {
        0 => shadow_data.coverage_near,
        _ => shadow_data.splits[index.saturating_sub(1)].max(shadow_data.coverage_near),
    };
    let max_depth = shadow_data.splits[index].max(min_depth + 1.0);

    (min_depth, max_depth)
}

/// Returns inert shadow matrices for frames that have no live shadow casters.
fn disabled_shadow_frame_data() -> ShadowFrameData {
    ShadowFrameData {
        view_proj: [identity_mat4(); SHADOW_CASCADE_COUNT],
        coverage_near: 0.03,
        splits: DEFAULT_SHADOW_CASCADE_SPLITS,
        texel_world: DEFAULT_SHADOW_CASCADE_METRICS,
        depth_span: DEFAULT_SHADOW_CASCADE_METRICS,
        cascade_resolution: [shadow_map_size() as f32; SHADOW_CASCADE_COUNT],
    }
}

fn disabled_local_shadow_frame_data() -> LocalShadowFrameData {
    LocalShadowFrameData {
        view_proj: [identity_mat4(); LOCAL_SHADOW_MATRIX_COUNT],
        params: [[0.0, 0.0, 1.0, 1.0]; MAX_LOCAL_LIGHTS],
        lights: [None; MAX_LOCAL_LIGHTS],
    }
}

/// Hashes the renderer-facing shadow caster set without depending on ECS state.
fn shadow_caster_hash(items: &[RenderItemPacket]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for item in items
        .iter()
        .filter(|item| item.flags.visible && item.flags.casts_shadow)
    {
        hash = fnv1a(hash, item.mesh.raw());
        hash = fnv1a(hash, item.material.raw());
        hash = fnv1a(hash, item.layer as u64);
        hash = fnv1a(hash, item.object_id.map_or(0, |id| id.raw()));
    }

    hash
}

/// Mixes one integer into a small deterministic FNV-1a hash.
fn fnv1a(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
}

fn hash_float4(mut hash: u64, value: [f32; 4]) -> u64 {
    for component in value {
        hash = fnv1a(hash, component.to_bits() as u64);
    }
    hash
}

/// Uses a stable per-cascade resolution and exact one-texel snapping.
///
/// The receiver sphere keeps the projection scale fixed while the camera moves, and the snapped
/// center prevents sub-texel shadow-map swimming.
fn stable_csm_cascade_bounds_for_splits(
    camera: CameraSnapshot,
    aspect: f32,
    mut cascade_near: f32,
    splits: [f32; SHADOW_CASCADE_COUNT],
    shadow_resolution: f32,
) -> [ShadowCascadeBounds; SHADOW_CASCADE_COUNT] {
    let frustum_basis = camera_frustum_basis(camera, aspect);
    let coverage_far = shadow_coverage_distance(camera);
    std::array::from_fn(|cascade_index| {
        let base_near = cascade_near;
        let base_far = splits[cascade_index].max(base_near + 1.0);
        let expanded_near = if cascade_index == 0 {
            base_near
        } else {
            (base_near - shadow_cascade_overlap_width(base_near)).max(0.03)
        };
        let expanded_far = if cascade_index + 1 == SHADOW_CASCADE_COUNT {
            base_far
        } else {
            (base_far + shadow_cascade_overlap_width(base_far)).min(coverage_far)
        };
        let bounds = shadow_cascade_bounds_with_sampling(
            frustum_basis,
            expanded_near,
            expanded_far,
            cascade_index,
            shadow_resolution,
            1.0,
        );
        cascade_near = base_far;
        bounds
    })
}

/// Returns the shared camera-depth overlap width used by CPU projection fitting and the shader.
fn shadow_cascade_overlap_width(split: f32) -> f32 {
    (split.max(0.0) * 0.045).max(1.0)
}

fn shadow_caster_depth_reserve(
    shadow_radius: f32,
    cascade_near: f32,
    cascade_far: f32,
    cascade_index: usize,
) -> f32 {
    let cascade_depth = (cascade_far - cascade_near).max(1.0);
    let reserve = shadow_radius * 2.5 + cascade_depth * 0.75;
    let minimum = match cascade_index {
        0 => 96.0,
        1 => 128.0,
        2 => 192.0,
        _ => 256.0,
    };

    reserve.max(minimum).min(SHADOW_CASTER_DEPTH_RESERVE_MAX)
}

fn camera_frustum_basis(camera: CameraSnapshot, aspect: f32) -> CameraFrustumBasis {
    let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);
    let tan_y = (camera.fov_y_radians * 0.5).tan().max(0.001);
    let tan_x = tan_y * aspect.max(0.001);

    CameraFrustumBasis {
        eye: camera.eye,
        forward,
        right,
        up,
        tan_x,
        tan_y,
    }
}

fn camera_frustum_corners(basis: CameraFrustumBasis, near: f32, far: f32) -> [[f32; 3]; 8] {
    let mut corners = [[0.0; 3]; 8];

    for (plane, depth) in [near, far].into_iter().enumerate() {
        let center = add3(basis.eye, mul3(basis.forward, depth));
        let x = mul3(basis.right, basis.tan_x * depth);
        let y = mul3(basis.up, basis.tan_y * depth);
        let base = plane * 4;
        corners[base] = add3(sub3(center, x), y);
        corners[base + 1] = add3(add3(center, x), y);
        corners[base + 2] = sub3(add3(center, x), y);
        corners[base + 3] = sub3(sub3(center, x), y);
    }

    corners
}

fn bounding_sphere(points: &[[f32; 3]]) -> ([f32; 3], f32) {
    let mut center = [0.0; 3];
    for point in points {
        center = add3(center, *point);
    }
    center = mul3(center, 1.0 / points.len().max(1) as f32);

    let radius = points
        .iter()
        .map(|point| distance_squared(center, *point))
        .fold(0.0_f32, f32::max)
        .sqrt();

    (center, radius)
}

/// Quantizes the orthographic projection center in light space.
///
/// Camera motion smaller than one shadow texel then leaves the projected texel grid unchanged,
/// eliminating the shadow-map swimming that is most visible as steps moving across sloped faces.
#[cfg(test)]
fn snap_shadow_center(light_dir: [f32; 3], center: [f32; 3], texel_world: f32) -> [f32; 3] {
    snap_shadow_center_with_basis(stable_light_basis_value(light_dir), center, texel_world)
}

fn snap_shadow_center_with_basis(
    light_basis: StableLightBasis,
    center: [f32; 3],
    texel_world: f32,
) -> [f32; 3] {
    if !texel_world.is_finite() || texel_world <= f32::EPSILON {
        return center;
    }

    let right_position = dot3(center, light_basis.right);
    let up_position = dot3(center, light_basis.up);
    let snapped_right = (right_position / texel_world).round() * texel_world;
    let snapped_up = (up_position / texel_world).round() * texel_world;

    add3(
        add3(
            center,
            mul3(light_basis.right, snapped_right - right_position),
        ),
        mul3(light_basis.up, snapped_up - up_position),
    )
}

fn stable_light_basis(light_dir: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let forward = normalize_or(light_dir, [0.0, -1.0, 0.0]);
    let reference_up = if dot3(forward, [0.0, 1.0, 0.0]).abs() > 0.92 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize_or(cross3(forward, reference_up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);

    (forward, right, up)
}

fn stable_light_basis_value(light_dir: [f32; 3]) -> StableLightBasis {
    let (forward, right, up) = stable_light_basis(light_dir);
    StableLightBasis { forward, right, up }
}

fn stable_light_view_with_basis(
    light_basis: StableLightBasis,
    center: [f32; 3],
    radius: f32,
    caster_depth_reserve: f32,
) -> [f32; 16] {
    let eye = sub3(
        center,
        mul3(
            light_basis.forward,
            radius * 3.0 + 16.0 + caster_depth_reserve,
        ),
    );

    look_at_rh(eye, center, light_basis.up)
}

fn shadow_depth_range(
    view: &[f32; 16],
    shadow_radius: f32,
    receiver_points: &[[f32; 3]],
    focus_center: [f32; 3],
    caster_depth_reserve: f32,
) -> (f32, f32) {
    let mut min_depth = f32::INFINITY;
    let mut max_depth = f32::NEG_INFINITY;
    let mut include_depth = |center: [f32; 3], radius: f32| {
        let light_center = transform_point(*view, center);
        let depth = -light_center[2];
        min_depth = min_depth.min(depth - radius);
        max_depth = max_depth.max(depth + radius);
    };

    include_depth(focus_center, shadow_radius);
    for point in receiver_points {
        include_depth(*point, 0.0);
    }

    if !min_depth.is_finite() || !max_depth.is_finite() {
        return (0.1, (shadow_radius * 4.0).max(16.0));
    }

    let receiver_margin = (shadow_radius * 0.08).clamp(1.0, SHADOW_DEPTH_PADDING);
    let near_margin = receiver_margin + caster_depth_reserve;
    let far_margin = receiver_margin + caster_depth_reserve * 0.25;
    let near = (min_depth - near_margin).max(0.05);
    let far = (max_depth + far_margin).max(near + 8.0);
    (near, far)
}

fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    let x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
    let y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
    let z = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
    let w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];

    if w.abs() > f32::EPSILON {
        [x / w, y / w, z / w]
    } else {
        [x, y, z]
    }
}

fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];

    dx * dx + dy * dy + dz * dz
}

/// Builds a Vulkan clip-space orthographic projection with NDC depth in 0..1.
fn orthographic_vulkan(width: f32, height: f32, near: f32, far: f32) -> [f32; 16] {
    let z = 1.0 / (near - far);
    [
        2.0 / width,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / height,
        0.0,
        0.0,
        0.0,
        0.0,
        z,
        0.0,
        0.0,
        0.0,
        near * z,
        1.0,
    ]
}

fn perspective_cubemap(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y * 0.5).tan().max(0.001);
    let z = far / (near - far).min(-0.001);
    [
        f / aspect.max(0.001),
        0.0,
        0.0,
        0.0,
        0.0,
        f,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MaterialHandle, MeshHandle};

    fn camera() -> CameraSnapshot {
        CameraSnapshot::perspective(
            [0.0, 1.5, 6.0],
            [0.0, 1.5, 5.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            5000.0,
        )
        .expect("test camera is finite")
    }

    #[test]
    fn translucent_shadow_cascade_mask_uses_one_bit_per_map() {
        assert_eq!(
            translucent_shadow_cascade_bits([true, false, true, false]),
            5.0
        );
        assert_eq!(
            translucent_shadow_cascade_bits([false; SHADOW_CASCADE_COUNT]),
            0.0
        );
    }

    #[test]
    fn caster_free_frame_keeps_directional_light_energy() {
        let uniform = mesh_frame_uniform_for_frame(
            camera(),
            1.0,
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            false,
            [false; SHADOW_CASCADE_COUNT],
            None,
            StableCsmPcssQualitySettings::balanced(),
            None,
            EmissiveLightUniforms::disabled(),
            crate::protocol::DebugViewMode::Disabled,
        );

        assert_eq!(uniform.light_dir[3], 0.0);
        assert!(
            uniform.light_color[..3]
                .iter()
                .any(|component| *component > 0.0)
        );
    }

    #[test]
    fn ambient_light_falls_near_darkness_with_the_global_light() {
        let dark = ambient_color_for_light(LightPacket::new(0.0));
        let moon = ambient_color_for_light(LightPacket::new(0.14));
        let day = ambient_color_for_light(LightPacket::new(1.0));

        assert!(dark[3] < DEFAULT_AMBIENT_COLOR[3] * 0.01);
        assert!(moon[3] < DEFAULT_AMBIENT_COLOR[3] * 0.08);
        assert!(dark[3] < moon[3]);
        assert!(moon[3] < day[3]);
        for (actual, default) in day.into_iter().zip(DEFAULT_AMBIENT_COLOR) {
            assert!((actual - default * 0.55).abs() < 1.0e-6);
        }
    }

    #[test]
    fn shadow_cascade_splits_keep_near_density_and_cover_camera_range() {
        let splits = shadow_cascade_splits(camera());

        assert!(splits[0] < 32.0);
        assert!(splits[0] < splits[1]);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert_eq!(splits[3], camera().far);
        assert_eq!(splits[3], shadow_coverage_distance(camera()));
    }

    #[test]
    fn shadow_center_snaps_to_the_light_space_texel_grid() {
        let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
        let texel_world = 0.25;
        let center = [3.137, -2.419, 8.731];
        let snapped = snap_shadow_center(light_dir, center, texel_world);
        let (_, right, up) = stable_light_basis(light_dir);

        let right_texels = dot3(snapped, right) / texel_world;
        let up_texels = dot3(snapped, up) / texel_world;
        assert!((right_texels - right_texels.round()).abs() < 0.0001);
        assert!((up_texels - up_texels.round()).abs() < 0.0001);

        let sub_texel_motion = add3(
            add3(snapped, mul3(right, texel_world * 0.40)),
            mul3(up, texel_world * 0.40),
        );
        let resnapped = snap_shadow_center(light_dir, sub_texel_motion, texel_world);
        assert!(distance_squared(snapped, resnapped) < 0.000001);
    }

    #[test]
    fn stable_csm_cull_uses_one_projection_per_cascade() {
        let data = stable_csm_frame_data_for_resolution(
            camera(),
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            LightPacket::new(1.0),
            shadow_map_size(),
        );
        let cull = shadow_cascade_cull(camera(), 0, &data);

        assert_eq!(cull.light_space_projection_count(), 1);
    }

    #[test]
    fn stable_csm_matrices_cover_all_cascades_with_metrics() {
        let camera = camera();
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let splits = shadow_cascade_splits(camera);
        let light = LightPacket::new(1.0);
        let stable = stable_csm_frame_data_for_resolution(camera, extent, light, shadow_map_size());

        assert_eq!(stable.splits, splits);
        for cascade_index in 0..SHADOW_CASCADE_COUNT {
            assert_ne!(stable.view_proj[cascade_index], identity_mat4());
            assert!(stable.texel_world[cascade_index] > 0.0);
            assert!(stable.depth_span[cascade_index] > 0.0);
        }
    }

    #[test]
    fn stable_csm_projection_fits_overlap_on_both_sides_of_each_split() {
        let camera = camera();
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let splits = shadow_cascade_splits(camera);
        let bounds = stable_csm_cascade_bounds_for_splits(
            camera,
            extent.width as f32 / extent.height as f32,
            camera.near.max(0.03),
            splits,
            4096.0,
        );

        for boundary in 0..SHADOW_CASCADE_COUNT - 1 {
            assert!(bounds[boundary].cascade_far > splits[boundary]);
            assert!(bounds[boundary + 1].cascade_near < splits[boundary]);
        }
    }

    #[test]
    fn shadow_projection_reserves_depth_for_large_casters() {
        let near_reserve = shadow_caster_depth_reserve(12.0, 0.1, 20.0, 0);
        let far_reserve = shadow_caster_depth_reserve(80.0, 111.0, 320.0, 3);

        assert!(near_reserve >= 96.0);
        assert!(far_reserve >= 256.0);
        assert!(far_reserve > near_reserve);
    }

    #[test]
    fn local_shadow_frame_data_builds_cube_faces_for_shadowed_lights() {
        let mut emissive_lights = EmissiveLightUniforms::disabled();
        emissive_lights.position_radius[0] = [0.0, 0.0, 0.0, 40.0];
        emissive_lights.direction_radius[0] = [0.0, -1.0, 0.0, 1.0];
        emissive_lights.size_kind[0] = [0.0, 0.0, 1.0, 1.0];
        emissive_lights.position_radius[1] = [2.0, 3.0, 4.0, 24.0];
        emissive_lights.direction_radius[1] = [0.0, -1.0, 0.0, 0.5];
        emissive_lights.size_kind[1] = [0.0, 0.0, 1.0, 0.0];
        emissive_lights.position_radius[2] = [5.0, 6.0, 7.0, 32.0];
        emissive_lights.direction_radius[2] = [0.0, -1.0, 0.0, 0.75];
        emissive_lights.size_kind[2] = [0.0, 0.0, 1.0, 1.0];
        emissive_lights.count[0] = 3.0;
        let extent = vk::Extent2D {
            width: 1024,
            height: 1024,
        };

        let data = local_shadow_frame_data(&emissive_lights, extent)
            .expect("enabled local light should build cubemap shadow data");

        assert_eq!(data.view_proj.len(), LOCAL_SHADOW_MATRIX_COUNT);
        assert_eq!(data.params[0][0], 1.0);
        assert!((data.params[0][1] - (1.5 / 1024.0)).abs() < 0.000001);
        assert!(data.params[0][2] > 0.0);
        assert!(data.params[0][3] > data.params[0][2]);
        assert_eq!(data.params[1][0], 0.0);
        assert_eq!(data.params[2][0], 1.0);
        assert_eq!(data.params[2][1], data.params[0][1]);
        assert!(data.lights[0].is_some());
        assert!(data.lights[1].is_none());
        assert!(data.lights[2].is_some());
    }

    #[test]
    fn local_shadow_cube_faces_match_sampler_axes() {
        let faces = local_shadow_cube_view_projections([0.0, 0.0, 0.0], 0.1, 100.0);
        let centers = [
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, -10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -10.0],
        ];

        for (face, center) in faces.iter().zip(centers) {
            let projected = transform_point(*face, center);
            assert!(projected[0].abs() < 0.0001);
            assert!(projected[1].abs() < 0.0001);
            assert!(projected[2].is_finite());
        }

        assert!(transform_point(faces[0], [10.0, 0.0, -1.0])[0] > 0.0);
        assert!(transform_point(faces[0], [10.0, 1.0, 0.0])[1] < 0.0);
        assert!(transform_point(faces[1], [-10.0, 0.0, 1.0])[0] > 0.0);
        assert!(transform_point(faces[1], [-10.0, 1.0, 0.0])[1] < 0.0);
        assert!(transform_point(faces[2], [1.0, 10.0, 0.0])[0] > 0.0);
        assert!(transform_point(faces[2], [0.0, 10.0, 1.0])[1] > 0.0);
        assert!(transform_point(faces[3], [1.0, -10.0, 0.0])[0] > 0.0);
        assert!(transform_point(faces[3], [0.0, -10.0, 1.0])[1] < 0.0);
        assert!(transform_point(faces[4], [1.0, 0.0, 10.0])[0] > 0.0);
        assert!(transform_point(faces[4], [0.0, 1.0, 10.0])[1] < 0.0);
        assert!(transform_point(faces[5], [-1.0, 0.0, -10.0])[0] > 0.0);
        assert!(transform_point(faces[5], [0.0, 1.0, -10.0])[1] < 0.0);
    }

    #[test]
    fn local_shadow_face_culling_keeps_edges_without_drawing_opposite_faces() {
        let light = LocalShadowLightData {
            light_index: 0,
            light_position_radius: [0.0, 0.0, 0.0, 100.0],
            source_radius: 0.5,
        };
        let positive_x =
            SceneBounds::new([10.0, 0.0, 0.0], 0.5).expect("test local-shadow bounds are finite");
        let cube_edge =
            SceneBounds::new([10.0, 0.0, 10.0], 0.5).expect("test cubemap-edge bounds are finite");

        assert!(local_shadow_face_contains_bounds(light, 0, positive_x));
        assert!(!local_shadow_face_contains_bounds(light, 1, positive_x));
        assert!(local_shadow_face_contains_bounds(light, 0, cube_edge));
        assert!(local_shadow_face_contains_bounds(light, 4, cube_edge));
    }

    #[test]
    fn local_shadow_signature_tracks_geometry_but_ignores_light_color() {
        let mut lights = EmissiveLightUniforms::disabled();
        lights.position_radius[0] = [1.0, 2.0, 3.0, 40.0];
        lights.direction_radius[0] = [0.0, -1.0, 0.0, 0.5];
        lights.size_kind[0] = [0.5, 0.5, 1.0, 1.0];
        lights.color[0] = [1.0, 0.5, 0.25, 1.0];
        lights.count[0] = 1.0;

        let empty = local_shadow_frame_signature(&[], &lights);
        lights.color[0] = [0.1, 0.2, 0.3, 4.0];
        assert_eq!(empty, local_shadow_frame_signature(&[], &lights));

        lights.position_radius[0][0] += 1.0;
        assert_ne!(empty, local_shadow_frame_signature(&[], &lights));

        let caster = RenderItemPacket::new(
            MeshHandle::from_raw(1).expect("test mesh handle is non-zero"),
            MaterialHandle::from_raw(1).expect("test material handle is non-zero"),
        );
        assert_ne!(
            local_shadow_frame_signature(&[], &lights),
            local_shadow_frame_signature(&[caster], &lights)
        );
    }
}
