use ash::vk;

use crate::{
    math::{add3, cross3, dot3, identity_mat4, mul3, normalize_or, sub3},
    protocol::{
        CameraSnapshot, ContactShadowQualitySettings, RenderItemPacket, RenderQualitySettings,
    },
    renderer::{
        DEFAULT_AMBIENT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_DIR,
        DEFAULT_SHADOW_CASCADE_METRICS, DEFAULT_SHADOW_CASCADE_SPLITS, graph::SHADOW_CASCADE_COUNT,
        shadow_cascade_size,
    },
};

use super::mesh::{EmissiveLightUniforms, MeshFrameUniform, ShadowCascadeCull};

const SHADOW_SPLIT_LAMBDA: f32 = 0.78;
const SHADOW_SPLIT_NEAR_FLOOR: f32 = 1.0;
const SHADOW_RADIUS_PADDING: f32 = 1.08;
const SHADOW_MIN_RADIUS: f32 = 4.0;
const SHADOW_DEPTH_PADDING: f32 = 24.0;
const SHADOW_CASTER_DEPTH_RESERVE_MAX: f32 = 640.0;
const SHADOW_SIGNATURE_POSITION_STEP: f32 = 0.0001;
const SHADOW_SIGNATURE_DIRECTION_STEP: f32 = 0.00001;
const SHADOW_SIGNATURE_FOV_STEP: f32 = 0.00001;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ShadowFrameSignature {
    camera_eye_bucket: [i32; 3],
    camera_forward_bucket: [i32; 3],
    fov_bucket: i32,
    caster_hash: u64,
    translucent_casters: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ShadowFrameData {
    pub(super) view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
    pub(super) splits: [f32; 4],
    pub(super) texel_world: [f32; 4],
    pub(super) depth_span: [f32; 4],
}

#[derive(Clone, Copy)]
struct ShadowCascadeProjection {
    view_projection: [f32; 16],
    texel_world: f32,
    depth_span: f32,
}

/// Builds the stable frame signature used to decide whether cached shadow maps can be reused.
pub(super) fn shadow_frame_signature(
    camera: CameraSnapshot,
    items: &[RenderItemPacket],
    has_translucent_shadow_casters: bool,
) -> ShadowFrameSignature {
    let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);

    ShadowFrameSignature {
        camera_eye_bucket: quantize3(camera.eye, SHADOW_SIGNATURE_POSITION_STEP),
        camera_forward_bucket: quantize3(forward, SHADOW_SIGNATURE_DIRECTION_STEP),
        fov_bucket: quantize_scalar(camera.fov_y_radians, SHADOW_SIGNATURE_FOV_STEP),
        caster_hash: shadow_caster_hash(items),
        translucent_casters: has_translucent_shadow_casters,
    }
}

/// Builds the shadow matrices that correspond to freshly rendered shadow maps.
pub(super) fn shadow_frame_data(camera: CameraSnapshot, extent: vk::Extent2D) -> ShadowFrameData {
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
    let splits = shadow_cascade_splits(camera);
    let projections = shadow_view_projections(camera, aspect, light_dir);

    ShadowFrameData {
        view_proj: std::array::from_fn(|index| projections[index].view_projection),
        splits,
        texel_world: shadow_cascade_metric_vec4(&projections, |projection| projection.texel_world),
        depth_span: shadow_cascade_metric_vec4(&projections, |projection| projection.depth_span),
    }
}

/// Returns the conservative camera-depth culling window for one shadow cascade.
pub(super) fn shadow_cascade_cull(
    camera: CameraSnapshot,
    cascade_index: usize,
) -> ShadowCascadeCull {
    let (min_depth, max_depth) = shadow_cascade_depth_range(camera, cascade_index);

    ShadowCascadeCull::new(camera, min_depth, max_depth)
}

/// Builds the frame uniform consumed by mesh vertex and fragment shaders.
pub(super) fn mesh_frame_uniform_for_frame(
    camera: CameraSnapshot,
    light_intensity: f32,
    quality: RenderQualitySettings,
    extent: vk::Extent2D,
    has_shadow_casters: bool,
    has_translucent_shadow_casters: bool,
    shadow_data: Option<ShadowFrameData>,
    emissive_lights: EmissiveLightUniforms,
) -> MeshFrameUniform {
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
    let shadow_data = shadow_data.unwrap_or_else(disabled_shadow_frame_data);

    MeshFrameUniform {
        view_proj: camera.view_projection(aspect),
        view: look_at_rh(camera.eye, camera.target, camera.up),
        shadow_view_proj: shadow_data.view_proj,
        shadow_cascade_splits: shadow_data.splits,
        shadow_cascade_texel_world: shadow_data.texel_world,
        shadow_cascade_depth_span: shadow_data.depth_span,
        camera_pos: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
        light_dir: [
            light_dir[0],
            light_dir[1],
            light_dir[2],
            if has_shadow_casters { 1.0 } else { 0.0 },
        ],
        light_color: [
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[0] * light_intensity,
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[1] * light_intensity,
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[2] * light_intensity,
            if has_translucent_shadow_casters {
                1.0
            } else {
                0.0
            },
        ],
        ambient_color: DEFAULT_AMBIENT_COLOR,
        contact_shadow: contact_shadow_params(
            quality.contact_shadow(),
            has_shadow_casters,
            light_intensity,
        ),
        emissive_light_position_radius: emissive_lights.position_radius,
        emissive_light_color: emissive_lights.color,
        emissive_light_count: emissive_lights.count,
        local_shadow_caster_center_radius: emissive_lights.shadow_caster_center_radius,
        local_shadow_caster_count: emissive_lights.shadow_caster_count,
    }
}

fn contact_shadow_params(
    settings: ContactShadowQualitySettings,
    has_shadow_casters: bool,
    light_intensity: f32,
) -> [f32; 4] {
    let intensity = if has_shadow_casters {
        settings.intensity() * light_intensity.clamp(0.0, 1.0)
    } else {
        0.0
    };

    [
        intensity,
        settings.max_distance(),
        settings.thickness(),
        settings.sample_count() as f32,
    ]
}

/// Returns the camera-local shadow distance so scene scale does not dilute cascade texels.
pub(super) fn shadow_coverage_distance(camera: CameraSnapshot) -> f32 {
    const MAX_SHADOW_DISTANCE: f32 = 320.0;
    let near = camera.near.max(0.03);

    camera
        .far
        .max(near + SHADOW_CASCADE_COUNT as f32)
        .min(MAX_SHADOW_DISTANCE)
}

/// Returns the split distances used by the fixed cascade count.
pub(super) fn shadow_cascade_splits(camera: CameraSnapshot) -> [f32; 4] {
    let near = camera.near.max(0.03);
    let far = shadow_coverage_distance(camera);
    let range = (far - near).max(1.0);
    let split_near = near.max(SHADOW_SPLIT_NEAR_FLOOR);
    let ratio = (far / split_near).max(1.0);
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
                far - (SHADOW_CASCADE_COUNT - index - 1) as f32,
            )
        };
        previous = split;
        split
    })
}

/// Builds a stable directional-light projection for one camera cascade range.
fn shadow_view_projection(
    camera: CameraSnapshot,
    aspect: f32,
    light_dir: [f32; 3],
    cascade_near: f32,
    cascade_far: f32,
    cascade_index: usize,
) -> ShadowCascadeProjection {
    let frustum = camera_frustum_corners(camera, aspect, cascade_near, cascade_far);
    let (center, radius) = bounding_sphere(&frustum);
    let shadow_resolution = shadow_cascade_resolution(cascade_index);
    let radius = (radius * SHADOW_RADIUS_PADDING).max(SHADOW_MIN_RADIUS);
    let caster_depth_reserve =
        shadow_caster_depth_reserve(radius, cascade_near, cascade_far, cascade_index);
    let view = stable_light_view(light_dir, center, radius, caster_depth_reserve);

    let (near, far) = shadow_depth_range(&view, radius, &frustum, center, caster_depth_reserve);
    let texel_world = radius * 2.0 / shadow_resolution;
    let depth_span = (far - near).max(1.0);

    tracing::trace!(
        cascade_index,
        cascade_near,
        cascade_far,
        radius,
        texel_world,
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

/// Packs cascade metrics into the vec4 layout consumed by mesh shaders.
fn shadow_cascade_metric_vec4(
    projections: &[ShadowCascadeProjection; SHADOW_CASCADE_COUNT],
    value: impl Fn(ShadowCascadeProjection) -> f32,
) -> [f32; 4] {
    let mut output = [0.0; 4];
    for (index, projection) in projections.iter().copied().enumerate() {
        output[index] = value(projection);
    }
    output
}

/// Returns the camera-space near/far depths that feed one cascade projection.
fn shadow_cascade_depth_range(camera: CameraSnapshot, cascade_index: usize) -> (f32, f32) {
    let splits = shadow_cascade_splits(camera);
    let min_depth = match cascade_index {
        0 => camera.near.max(0.03),
        index => splits[index.saturating_sub(1)].max(camera.near.max(0.03)),
    };
    let max_depth = splits[cascade_index.min(SHADOW_CASCADE_COUNT - 1)].max(min_depth + 1.0);

    (min_depth, max_depth)
}

/// Returns inert shadow matrices for frames that have no live shadow casters.
fn disabled_shadow_frame_data() -> ShadowFrameData {
    ShadowFrameData {
        view_proj: [identity_mat4(); SHADOW_CASCADE_COUNT],
        splits: DEFAULT_SHADOW_CASCADE_SPLITS,
        texel_world: DEFAULT_SHADOW_CASCADE_METRICS,
        depth_span: DEFAULT_SHADOW_CASCADE_METRICS,
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

/// Quantizes one vector so tiny camera changes do not force shadow-map redraws.
fn quantize3(value: [f32; 3], step: f32) -> [i32; 3] {
    [
        quantize_scalar(value[0], step),
        quantize_scalar(value[1], step),
        quantize_scalar(value[2], step),
    ]
}

/// Quantizes one finite scalar into a stable cache signature bucket.
fn quantize_scalar(value: f32, step: f32) -> i32 {
    if !value.is_finite() || step <= f32::EPSILON {
        return 0;
    }

    (value / step)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Builds directional-light projections for camera-distance cascades.
fn shadow_view_projections(
    camera: CameraSnapshot,
    aspect: f32,
    light_dir: [f32; 3],
) -> [ShadowCascadeProjection; SHADOW_CASCADE_COUNT] {
    let splits = shadow_cascade_splits(camera);
    let mut cascade_near = camera.near.max(0.03);

    std::array::from_fn(|cascade_index| {
        let cascade_far = splits[cascade_index].max(cascade_near + 1.0);
        let projection = shadow_view_projection(
            camera,
            aspect,
            light_dir,
            cascade_near,
            cascade_far,
            cascade_index,
        );
        cascade_near = cascade_far;
        projection
    })
}

fn shadow_cascade_resolution(cascade_index: usize) -> f32 {
    shadow_cascade_size(cascade_index) as f32
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

fn camera_frustum_corners(
    camera: CameraSnapshot,
    aspect: f32,
    near: f32,
    far: f32,
) -> [[f32; 3]; 8] {
    let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);
    let tan_y = (camera.fov_y_radians * 0.5).tan().max(0.001);
    let tan_x = tan_y * aspect.max(0.001);
    let mut corners = [[0.0; 3]; 8];

    for (plane, depth) in [near, far].into_iter().enumerate() {
        let center = add3(camera.eye, mul3(forward, depth));
        let x = mul3(right, tan_x * depth);
        let y = mul3(up, tan_y * depth);
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

fn stable_light_view(
    light_dir: [f32; 3],
    center: [f32; 3],
    radius: f32,
    caster_depth_reserve: f32,
) -> [f32; 16] {
    let light_dir = normalize_or(light_dir, [0.0, -1.0, 0.0]);
    let eye = sub3(
        center,
        mul3(light_dir, radius * 3.0 + 16.0 + caster_depth_reserve),
    );
    let up = if dot3(light_dir, [0.0, 1.0, 0.0]).abs() > 0.92 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    look_at_rh(eye, center, up)
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
    fn contact_shadow_uniform_requires_light_and_shadow_casters() {
        let extent = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        let quality = RenderQualitySettings::balanced();

        let without_casters = mesh_frame_uniform_for_frame(
            camera(),
            1.0,
            quality,
            extent,
            false,
            false,
            None,
            EmissiveLightUniforms::disabled(),
        );
        let without_light = mesh_frame_uniform_for_frame(
            camera(),
            0.0,
            quality,
            extent,
            true,
            false,
            None,
            EmissiveLightUniforms::disabled(),
        );
        let enabled = mesh_frame_uniform_for_frame(
            camera(),
            1.0,
            quality,
            extent,
            true,
            false,
            None,
            EmissiveLightUniforms::disabled(),
        );

        assert_eq!(without_casters.contact_shadow[0], 0.0);
        assert_eq!(without_light.contact_shadow[0], 0.0);
        assert_eq!(
            enabled.contact_shadow[0],
            quality.contact_shadow().intensity()
        );
        assert_eq!(
            enabled.contact_shadow[1],
            quality.contact_shadow().max_distance()
        );
    }

    #[test]
    fn shadow_cascade_splits_keep_near_density_without_following_scene_scale() {
        let splits = shadow_cascade_splits(camera());

        assert!(splits[0] < 32.0);
        assert!(splits[0] < splits[1]);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert!(splits[3] < camera().far);
        assert_eq!(splits[3], shadow_coverage_distance(camera()));
    }

    #[test]
    fn near_shadow_cascade_has_smaller_world_texels() {
        let camera = camera();
        let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
        let splits = shadow_cascade_splits(camera);
        let near = shadow_view_projection(camera, 16.0 / 9.0, light_dir, camera.near, splits[0], 0);
        let far = shadow_view_projection(camera, 16.0 / 9.0, light_dir, splits[2], splits[3], 3);

        assert!(near.view_projection[0].abs() > far.view_projection[0].abs());
        assert!(near.texel_world < far.texel_world);
        assert!(near.depth_span > 0.0);
        assert!(far.depth_span > 0.0);
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
    fn shadow_cascade_metrics_pack_four_values_into_vec4() {
        let projections = std::array::from_fn(|index| ShadowCascadeProjection {
            view_projection: identity_mat4(),
            texel_world: (index + 1) as f32,
            depth_span: 10.0 + index as f32,
        });

        assert_eq!(
            shadow_cascade_metric_vec4(&projections, |projection| projection.texel_world),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            shadow_cascade_metric_vec4(&projections, |projection| projection.depth_span),
            [10.0, 11.0, 12.0, 13.0]
        );
    }
}
