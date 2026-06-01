use super::RenderScene;

pub trait SceneController {
    fn on_renderer_ready(&mut self, _renderer: &mut crate::renderer::Renderer) {}
    fn on_message(&mut self, _message: SceneMessage) {}
    fn scene(&mut self, context: SceneContext) -> RenderScene;
}

#[derive(Debug, Clone, Copy)]
pub struct CameraMetering {
    pub valid: bool,
    pub average_luminance: f32,
    pub center_luminance: f32,
    pub highlight_fraction: f32,
    pub average_color: [f32; 3],
    pub white_balance_confidence: f32,
}

impl Default for CameraMetering {
    fn default() -> Self {
        Self {
            valid: false,
            average_luminance: 0.18,
            center_luminance: 0.18,
            highlight_fraction: 0.0,
            average_color: [0.18; 3],
            white_balance_confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SceneContext {
    pub elapsed: f32,
    pub delta_time: f32,
    pub frame: u64,
    pub window_size: [u32; 2],
    pub metering: CameraMetering,
}

#[derive(Debug, Clone)]
pub enum SceneMessage {
    Started { window_size: [u32; 2] },
    CloseRequested,
    Resized { width: u32, height: u32 },
    RedrawRequested,
    Keyboard { key: SceneKey, pressed: bool },
    MouseMotion { delta: [f32; 2] },
    MouseWheel { delta: f32 },
    User(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneKey {
    Escape,
    Space,
    ShiftLeft,
    ControlLeft,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    KeyQ,
    KeyE,
    F12,
    Other,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum RenderDebugMode {
    #[default]
    Default,
    Wireframe,
    Depth,
    Normals,
    ShadowMask,
    NoTexture,
}

impl RenderDebugMode {
    pub fn next(self) -> Self {
        match self {
            Self::Default => Self::Wireframe,
            Self::Wireframe => Self::Depth,
            Self::Depth => Self::Normals,
            Self::Normals => Self::ShadowMask,
            Self::ShadowMask => Self::NoTexture,
            Self::NoTexture => Self::Default,
        }
    }

    pub fn shader_value(self) -> f32 {
        match self {
            Self::Default => 0.0,
            Self::Wireframe => 1.0,
            Self::Depth => 2.0,
            Self::Normals => 3.0,
            Self::ShadowMask => 4.0,
            Self::NoTexture => 5.0,
        }
    }
}
