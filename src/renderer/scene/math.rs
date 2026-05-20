use super::{
    BoxReflectionSettings, Camera, PipelineId, RenderObject, RenderScene, SHADOW_CASCADE_COUNT,
    SHADOW_CASCADE_GRID, Transform,
    assets::{GpuAssets, GpuPrimitive},
    mat4_mul,
};

const FRUSTUM_CULL_RADIUS_SCALE: f32 = 1.35;
const FRUSTUM_CULL_MIN_MARGIN: f32 = 0.25;

pub(super) fn for_each_render_object(
    scene: &RenderScene,
    assets: &GpuAssets,
    mut visit: impl FnMut(RenderObject),
) {
    for object in &scene.objects {
        visit(*object);
    }

    for model in &scene.models {
        let Some(gpu_model) = assets.model(model.model) else {
            continue;
        };

        for primitive in &gpu_model.primitives {
            visit(model_object(model.transform, model.pipeline, primitive));
        }
    }
}

pub(super) fn for_each_visible_render_object(
    scene: &RenderScene,
    assets: &GpuAssets,
    camera: Camera,
    aspect: f32,
    mut visit: impl FnMut(RenderObject),
) {
    for_each_render_object(scene, assets, |object| {
        if render_object_visible(object, assets, camera, aspect) {
            visit(object);
        }
    });
}

fn model_object(
    transform: Transform,
    pipeline: PipelineId,
    primitive: &GpuPrimitive,
) -> RenderObject {
    RenderObject {
        mesh: primitive.mesh,
        pipeline,
        transform,
        material: primitive.material,
    }
}

fn render_object_visible(
    object: RenderObject,
    assets: &GpuAssets,
    camera: Camera,
    aspect: f32,
) -> bool {
    let Some(mesh) = assets.mesh(object.mesh) else {
        return false;
    };
    let matrix = object.transform.matrix();
    let center = transform_point(matrix, mesh.center);
    let radius = transformed_radius(object.transform, mesh.radius).max(0.01);

    sphere_visible(camera, aspect, center, radius)
}

fn sphere_visible(camera: Camera, aspect: f32, center: [f32; 3], radius: f32) -> bool {
    let radius = (radius * FRUSTUM_CULL_RADIUS_SCALE).max(radius + FRUSTUM_CULL_MIN_MARGIN);
    let forward = normalize_or(sub(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross(right, forward);
    let to_center = sub(center, camera.eye);
    let depth = dot3(to_center, forward);

    if depth + radius < camera.near || depth - radius > camera.far {
        return false;
    }

    let projected_depth = depth.max(camera.near);
    let tan_y = (camera.fov_y * 0.5).tan().max(0.001);
    let y_radius = radius * (1.0 + tan_y * tan_y).sqrt();
    let y_limit = projected_depth * tan_y + y_radius;
    if dot3(to_center, up).abs() > y_limit {
        return false;
    }

    let tan_x = tan_y * aspect.max(0.001);
    let x_radius = radius * (1.0 + tan_x * tan_x).sqrt();
    let x_limit = projected_depth * tan_x + x_radius;
    dot3(to_center, right).abs() <= x_limit
}

pub(super) fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

pub(super) fn transformed_radius(transform: Transform, radius: f32) -> f32 {
    let scale = transform.scale[0]
        .abs()
        .max(transform.scale[1].abs())
        .max(transform.scale[2].abs());

    radius * scale.max(0.001)
}

pub(super) fn desired_reflection_probe_size(settings: BoxReflectionSettings) -> u32 {
    settings.resolution.clamp(32, 2048)
}

pub(super) fn reflection_probe_face_axes(face: usize) -> ([f32; 3], [f32; 3]) {
    match face {
        0 => ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        1 => ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        2 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        3 => ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        4 => ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        _ => ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    }
}

pub(super) fn probe_binding_index(frame_index: usize, face: usize) -> usize {
    frame_index * super::assets::ReflectionProbe::FACE_COUNT + face
}

#[derive(Clone, Copy)]
pub(super) struct ShadowProjection {
    pub view_proj: [f32; 16],
    pub radius: f32,
    pub depth_range: f32,
    pub split_depth: f32,
    pub center: [f32; 3],
}

impl ShadowProjection {
    fn disabled() -> Self {
        Self {
            view_proj: [0.0; 16],
            radius: 0.0,
            depth_range: 1.0,
            split_depth: 0.0,
            center: [0.0; 3],
        }
    }
}

pub(super) fn cascaded_shadow_projection_for_aspect(
    scene: &RenderScene,
    assets: &GpuAssets,
    aspect: f32,
    atlas_resolution: f32,
) -> [ShadowProjection; SHADOW_CASCADE_COUNT] {
    let camera = scene.camera;
    let near = camera.near.max(0.01);
    let far = shadow_distance(scene, assets).max(near + 1.0);
    let light_dir = normalize_or(scene.light.direction, [-0.35, -0.75, -0.55]);
    let up = if dot3(light_dir, [0.0, 1.0, 0.0]).abs() > 0.92 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let cascade_resolution = (atlas_resolution / SHADOW_CASCADE_GRID.max(1) as f32).max(1.0);
    let mut cascades = [ShadowProjection::disabled(); SHADOW_CASCADE_COUNT];
    let mut previous_split = near;

    for (index, cascade) in cascades.iter_mut().enumerate() {
        let split = cascade_split(near, far, index + 1);
        let corners = frustum_corners(camera, aspect, previous_split, split);
        let center = average_points(&corners);
        let radius = cascade_radius(center, &corners);
        let eye = sub(center, scale(light_dir, radius * 2.0));
        let far_plane = (radius * 4.0).max(16.0);
        let mut view = look_at(eye, center, up);

        snap_shadow_view_to_texels(&mut view, center, radius, cascade_resolution);
        *cascade = ShadowProjection {
            view_proj: mat4_mul(orthographic_symmetric(radius, 0.1, far_plane), view),
            radius,
            depth_range: far_plane - 0.1,
            split_depth: split,
            center,
        };
        previous_split = split;
    }

    cascades
}

fn shadow_distance(scene: &RenderScene, assets: &GpuAssets) -> f32 {
    let scene_radius = scene_shadow_sphere(scene, assets)
        .map(|(_, radius)| radius * 3.0)
        .unwrap_or(512.0);

    scene_radius
        .max(192.0)
        .min(scene.camera.far)
        .clamp(scene.camera.near + 1.0, 3000.0)
}

fn cascade_split(near: f32, far: f32, cascade: usize) -> f32 {
    let p = cascade as f32 / SHADOW_CASCADE_COUNT as f32;
    let linear = near + (far - near) * p;
    let logarithmic = near * (far / near).powf(p);
    let practical = logarithmic * 0.68 + linear * 0.32;

    practical.clamp(near + 0.001, far)
}

fn frustum_corners(camera: Camera, aspect: f32, near: f32, far: f32) -> [[f32; 3]; 8] {
    let forward = normalize_or(sub(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross(right, forward);
    let tan_y = (camera.fov_y * 0.5).tan();
    let near_center = add(camera.eye, scale(forward, near));
    let far_center = add(camera.eye, scale(forward, far));
    let near_half_y = tan_y * near;
    let near_half_x = near_half_y * aspect.max(0.001);
    let far_half_y = tan_y * far;
    let far_half_x = far_half_y * aspect.max(0.001);

    let near = frustum_plane_corners(near_center, right, up, near_half_x, near_half_y);
    let far = frustum_plane_corners(far_center, right, up, far_half_x, far_half_y);

    [
        near[0], near[1], near[2], near[3], far[0], far[1], far[2], far[3],
    ]
}

fn frustum_plane_corners(
    center: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    half_x: f32,
    half_y: f32,
) -> [[f32; 3]; 4] {
    [
        add(add(center, scale(right, -half_x)), scale(up, -half_y)),
        add(add(center, scale(right, half_x)), scale(up, -half_y)),
        add(add(center, scale(right, -half_x)), scale(up, half_y)),
        add(add(center, scale(right, half_x)), scale(up, half_y)),
    ]
}

fn average_points(points: &[[f32; 3]; 8]) -> [f32; 3] {
    let sum = points.iter().fold([0.0; 3], |sum, point| add(sum, *point));

    scale(sum, 1.0 / points.len() as f32)
}

fn cascade_radius(center: [f32; 3], corners: &[[f32; 3]; 8]) -> f32 {
    corners
        .iter()
        .map(|corner| distance_squared(center, *corner).sqrt())
        .fold(0.0, f32::max)
        .mul_add(1.25, 0.0)
        .max(2.0)
}

fn snap_shadow_view_to_texels(
    view: &mut [f32; 16],
    center: [f32; 3],
    radius: f32,
    resolution: f32,
) {
    let texel_world = radius * 2.0 / resolution.max(1.0);
    if texel_world <= f32::EPSILON {
        return;
    }

    let center_in_light_space = transform_point(*view, center);
    let snapped_x = (center_in_light_space[0] / texel_world).round() * texel_world;
    let snapped_y = (center_in_light_space[1] / texel_world).round() * texel_world;

    view[12] += snapped_x - center_in_light_space[0];
    view[13] += snapped_y - center_in_light_space[1];
}

fn scene_shadow_sphere(scene: &RenderScene, assets: &GpuAssets) -> Option<([f32; 3], f32)> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut has_value = false;

    for_each_render_object(scene, assets, |object| {
        let Some(mesh) = assets.mesh(object.mesh) else {
            return;
        };
        let matrix = object.transform.matrix();
        let center = transform_point(matrix, mesh.center);
        let radius = transformed_radius(object.transform, mesh.radius).max(0.01);

        for axis in 0..3 {
            min[axis] = min[axis].min(center[axis] - radius);
            max[axis] = max[axis].max(center[axis] + radius);
        }
        has_value = true;
    });

    has_value.then(|| {
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let radius = distance_squared(min, max).sqrt() * 0.5;
        (center, radius)
    })
}

fn orthographic_symmetric(radius: f32, near: f32, far: f32) -> [f32; 16] {
    [
        1.0 / radius,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0 / radius,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / (near - far),
        0.0,
        0.0,
        0.0,
        near / (near - far),
        1.0,
    ]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let forward = normalize_or(sub(target, eye), [0.0, 0.0, -1.0]);
    let side = normalize_or(cross(forward, up), [1.0, 0.0, 0.0]);
    let up = cross(side, forward);

    [
        side[0],
        up[0],
        -forward[0],
        0.0,
        side[1],
        up[1],
        -forward[1],
        0.0,
        side[2],
        up[2],
        -forward[2],
        0.0,
        -dot3(side, eye),
        -dot3(up, eye),
        dot3(forward, eye),
        1.0,
    ]
}

pub(super) fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];

    dx * dx + dy * dy + dz * dz
}

pub(super) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let length = dot3(v, v).sqrt();
    if length <= f32::EPSILON {
        [0.0, 0.0, -1.0]
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}

pub(super) fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = dot3(v, v).sqrt();
    if length <= f32::EPSILON {
        fallback
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}

pub(super) fn reflect_camera(camera: Camera, plane_normal: [f32; 3], plane_d: f32) -> Camera {
    let eye = reflect_point(camera.eye, plane_normal, plane_d);
    let target = reflect_point(camera.target, plane_normal, plane_d);
    let up = normalize_or(reflect_vector(camera.up, plane_normal), camera.up);

    Camera {
        eye,
        target,
        up,
        ..camera
    }
}

fn reflect_point(point: [f32; 3], plane_normal: [f32; 3], plane_d: f32) -> [f32; 3] {
    let distance = dot3(point, plane_normal) + plane_d;
    [
        point[0] - 2.0 * distance * plane_normal[0],
        point[1] - 2.0 * distance * plane_normal[1],
        point[2] - 2.0 * distance * plane_normal[2],
    ]
}

fn reflect_vector(vector: [f32; 3], plane_normal: [f32; 3]) -> [f32; 3] {
    let distance = dot3(vector, plane_normal);
    [
        vector[0] - 2.0 * distance * plane_normal[0],
        vector[1] - 2.0 * distance * plane_normal[1],
        vector[2] - 2.0 * distance * plane_normal[2],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::{Camera, orthographic_symmetric, sphere_visible};

    fn project_z(matrix: [f32; 16], z: f32) -> f32 {
        matrix[10] * z + matrix[14]
    }

    #[test]
    fn shadow_ortho_uses_vulkan_depth_range() {
        let matrix = orthographic_symmetric(4.0, 0.1, 10.0);

        assert!((project_z(matrix, -0.1) - 0.0).abs() < 0.0001);
        assert!((project_z(matrix, -10.0) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn camera_frustum_rejects_far_offscreen_spheres() {
        let camera = Camera::default();

        assert!(sphere_visible(camera, 16.0 / 9.0, [0.0, 0.0, 0.0], 0.5));
        assert!(sphere_visible(camera, 16.0 / 9.0, [4.75, 0.0, 0.0], 0.5));
        assert!(!sphere_visible(camera, 16.0 / 9.0, [100.0, 0.0, 0.0], 0.5));
        assert!(!sphere_visible(camera, 16.0 / 9.0, [0.0, 0.0, 10.0], 0.5));
    }
}
