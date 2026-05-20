use super::{
    Camera, CameraResponse, DirectionalLight, ReflectionSettings, RenderDebugMode, RenderModel,
    RenderObject,
};

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
