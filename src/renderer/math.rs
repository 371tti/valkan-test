use super::{
    BoxReflectionSettings, Camera, PipelineId, RenderObject, Transform, assets::GpuPrimitive,
};

pub(super) fn model_object(
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
