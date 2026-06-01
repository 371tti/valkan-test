pub type Mat4 = [f32; 16];

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        mat4_mul(
            translation(self.translation),
            mat4_mul(
                rotation_z(self.rotation[2]),
                mat4_mul(
                    rotation_y(self.rotation[1]),
                    mat4_mul(rotation_x(self.rotation[0]), scale(self.scale)),
                ),
            ),
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
        }
    }
}

pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [0.0; 16];

    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|i| a[i * 4 + row] * b[col * 4 + i]).sum();
        }
    }

    out
}

pub(super) fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot(s, eye),
        -dot(u, eye),
        dot(f, eye),
        1.0,
    ]
}

pub(super) fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    perspective_with_y_scale(fov_y, aspect, near, far, -1.0)
}

pub(super) fn perspective_for_cubemap(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    perspective_with_y_scale(fov_y, aspect, near, far, 1.0)
}

fn perspective_with_y_scale(fov_y: f32, aspect: f32, near: f32, far: f32, y_scale: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();

    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f * y_scale,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        near * far / (near - far),
        0.0,
    ]
}

fn translation(v: [f32; 3]) -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, v[0], v[1], v[2], 1.0,
    ]
}

fn scale(v: [f32; 3]) -> Mat4 {
    [
        v[0], 0.0, 0.0, 0.0, 0.0, v[1], 0.0, 0.0, 0.0, 0.0, v[2], 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rotation_x(angle: f32) -> Mat4 {
    let (sin, cos) = angle.sin_cos();

    [
        1.0, 0.0, 0.0, 0.0, 0.0, cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rotation_y(angle: f32) -> Mat4 {
    let (sin, cos) = angle.sin_cos();

    [
        cos, 0.0, -sin, 0.0, 0.0, 1.0, 0.0, 0.0, sin, 0.0, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rotation_z(angle: f32) -> Mat4 {
    let (sin, cos) = angle.sin_cos();

    [
        cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt().max(f32::EPSILON);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
