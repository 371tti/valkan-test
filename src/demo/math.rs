pub fn add_scaled(target: &mut [f32; 3], value: [f32; 3], scale: f32) {
    target[0] += value[0] * scale;
    target[1] += value[1] * scale;
    target[2] += value[2] * scale;
}

pub fn scale(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

pub fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    normalize_or_zero(v)
}

pub fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return [0.0; 3];
    }

    [v[0] / len, v[1] / len, v[2] / len]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
