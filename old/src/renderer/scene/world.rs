use super::{Camera, CameraResponse, RenderDebugMode, Transform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

impl MaterialId {
    pub const DEFAULT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub usize);

impl TextureId {
    pub const DEFAULT: Self = Self(0);
    pub const NORMAL: Self = Self(1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub usize);

impl PipelineId {
    pub const LIT_MESH: Self = Self(0);
    pub const LIT_MESH_TRANSPARENT: Self = Self(1);
    pub const LIT_MESH_WIREFRAME: Self = Self(2);
    pub const LIT_MESH_DOUBLE_SIDED: Self = Self(3);
    pub const LIT_MESH_TRANSPARENT_DOUBLE_SIDED: Self = Self(4);
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
            ambient: [0.035, 0.04, 0.055],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectionSettings {
    pub box_projection: BoxReflectionSettings,
    pub planar: PlanarReflectionSettings,
}

#[derive(Debug, Clone, Copy)]
pub struct BoxReflectionSettings {
    pub enabled: bool,
    pub parallax_correction: bool,
    pub resolution: u32,
    pub intensity: f32,
    pub roughness_fallback: f32,
    pub bounds_padding: f32,
}

impl Default for BoxReflectionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            parallax_correction: true,
            resolution: 128,
            intensity: 1.0,
            roughness_fallback: 0.35,
            bounds_padding: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarReflectionSettings {
    pub enabled: bool,
    pub plane_origin: [f32; 3],
    pub plane_normal: [f32; 3],
    pub resolution_scale: f32,
    pub intensity: f32,
    pub max_roughness: f32,
    pub normal_alignment: f32,
    pub distance_fade: f32,
    pub clip_bias: f32,
    pub uv_flip_y: bool,
}

impl Default for PlanarReflectionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            plane_origin: [0.0, 0.0, 0.0],
            plane_normal: [0.0, 1.0, 0.0],
            resolution_scale: 0.5,
            intensity: 1.0,
            max_roughness: 0.75,
            normal_alignment: 0.35,
            distance_fade: 3.5,
            clip_bias: 0.03,
            uv_flip_y: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub camera: Camera,
    pub camera_response: CameraResponse,
    pub light: DirectionalLight,
    pub reflections: ReflectionSettings,
    pub debug_mode: RenderDebugMode,
    pub objects: Vec<RenderObject>,
    pub models: Vec<RenderModel>,
}
