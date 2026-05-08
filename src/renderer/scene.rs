#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);

impl MeshId {
    pub const CUBE: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

impl MaterialId {
    pub const DEFAULT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub usize);

impl ModelId {
    pub const CUBE: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub usize);

impl PipelineId {
    pub const LIT_MESH: Self = Self(0);
}

pub trait SceneController {
    fn on_renderer_ready(&mut self, _renderer: &mut super::Renderer) {}
    fn on_message(&mut self, _message: SceneMessage) {}
    fn scene(&mut self, context: SceneContext) -> RenderScene;
}

#[derive(Debug, Clone, Copy)]
pub struct SceneContext {
    pub elapsed: f32,
    pub delta_time: f32,
    pub frame: u64,
    pub window_size: [u32; 2],
}

#[derive(Debug, Clone)]
pub enum SceneMessage {
    Started { window_size: [u32; 2] },
    CloseRequested,
    Resized { width: u32, height: u32 },
    RedrawRequested,
    Keyboard { key: SceneKey, pressed: bool },
    User(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneKey {
    Space,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Other,
}

#[derive(Debug, Clone)]
pub struct RenderScene {
    pub camera: Camera,
    pub light: DirectionalLight,
    pub objects: Vec<RenderObject>,
    pub models: Vec<RenderModel>,
}

impl Default for RenderScene {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            light: DirectionalLight::default(),
            objects: Vec::new(),
            models: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderObject {
    pub mesh: MeshId,
    pub pipeline: PipelineId,
    pub transform: Transform,
    pub material: MaterialId,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderModel {
    pub model: ModelId,
    pub pipeline: PipelineId,
    pub transform: Transform,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub base_color: [f32; 4],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub ambient: [f32; 3],
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.35, -0.75, -0.55],
            color: [1.0, 0.94, 0.86],
            intensity: 1.2,
            ambient: [0.08, 0.1, 0.14],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        mat4_mul(
            perspective(self.fov_y, aspect, self.near, self.far),
            look_at(self.eye, self.target, self.up),
        )
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 4.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: 60.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }
}

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

pub type Mat4 = [f32; 16];

pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [0.0; 16];

    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|i| a[i * 4 + row] * b[col * 4 + i]).sum();
        }
    }

    out
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
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

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();

    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        -f,
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
