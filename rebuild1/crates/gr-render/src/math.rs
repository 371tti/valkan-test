/// Returns `a + b` for compact 3D vector accumulation.
pub(crate) fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Returns `a - b` for compact 3D geometry and camera math.
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Returns `a * scalar` for compact 3D geometry and camera math.
pub(crate) fn mul3(a: [f32; 3], scalar: f32) -> [f32; 3] {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
}

/// Returns the 3D vector cross product used to build basis vectors and normals.
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns the 3D vector dot product used by normalization and matrix translation.
pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns the Euclidean length of one 3D vector.
pub(crate) fn length3(value: [f32; 3]) -> f32 {
    dot3(value, value).sqrt()
}

/// Normalizes a 3D vector and returns `fallback` when the input cannot define a direction.
pub(crate) fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = length3(value);
    if !length.is_finite() || length <= f32::EPSILON {
        return fallback;
    }

    [value[0] / length, value[1] / length, value[2] / length]
}

/// Returns a 4x4 identity matrix in the flat layout consumed by renderer uniforms.
pub(crate) fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
