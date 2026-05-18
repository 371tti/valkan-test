use super::{
    BoxReflectionSettings, Camera, PipelineId, RenderObject, RenderScene, Transform,
    assets::{GpuAssets, GpuPrimitive},
    mat4_mul,
};

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
}

pub(super) fn shadow_projection(
    scene: &RenderScene,
    assets: &GpuAssets,
    resolution: f32,
) -> ShadowProjection {
    let (center, radius) =
        scene_shadow_sphere(scene, assets).unwrap_or((scene.camera.target, 12.0));
    let radius = (radius * 1.2).max(2.0);
    let light_dir = normalize_or(scene.light.direction, [-0.35, -0.75, -0.55]);
    let eye = sub(center, scale(light_dir, radius * 2.0));
    let far = (radius * 4.0).max(16.0);
    let up = if dot3(light_dir, [0.0, 1.0, 0.0]).abs() > 0.92 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut view = look_at(eye, center, up);

    snap_shadow_view_to_texels(&mut view, center, radius, resolution);

    ShadowProjection {
        view_proj: mat4_mul(orthographic_symmetric(radius, 0.1, far), view),
        radius,
        depth_range: far - 0.1,
    }
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
    use super::orthographic_symmetric;

    fn project_z(matrix: [f32; 16], z: f32) -> f32 {
        matrix[10] * z + matrix[14]
    }

    #[test]
    fn shadow_ortho_uses_vulkan_depth_range() {
        let matrix = orthographic_symmetric(4.0, 0.1, 10.0);

        assert!((project_z(matrix, -0.1) - 0.0).abs() < 0.0001);
        assert!((project_z(matrix, -10.0) - 1.0).abs() < 0.0001);
    }
}
