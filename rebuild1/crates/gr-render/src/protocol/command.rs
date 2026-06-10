use std::path::PathBuf;

use super::{
    FrameId, FrameSnapshot, FramebufferReadback, FramebufferReadbackOptions, MaterialHandle,
    MeshHandle, MessageEnvelope, NativeSurfacePlatform, NonZeroExtent, RenderQualitySettings,
    RequestId, SceneHandle, SurfaceDescriptor, SurfaceGeneration, SurfaceId, TextureHandle,
};

#[derive(Clone, Debug, Default)]
pub struct DebugOptions {
    pub validation_events: bool,
}

#[derive(Clone, Debug)]
pub enum RendererCommand {
    ConfigureSurface {
        surface: SurfaceDescriptor,
    },
    ResizeSurface {
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
    },
    LoadAsset {
        path: PathBuf,
    },
    UnloadAsset {
        asset: AssetHandle,
    },
    CreateScene,
    DestroyScene {
        scene: SceneHandle,
    },
    SetDebugOptions {
        options: DebugOptions,
    },
    SetFramebufferReadback {
        options: FramebufferReadbackOptions,
    },
    SetQualitySettings {
        settings: RenderQualitySettings,
    },
    SubmitFrame {
        snapshot: FrameSnapshot,
    },
    CaptureScreenshot {
        path: PathBuf,
    },
    Shutdown,
}

impl RendererCommand {
    /// Returns the command name used by trace logs and protocol diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ConfigureSurface { .. } => "ConfigureSurface",
            Self::ResizeSurface { .. } => "ResizeSurface",
            Self::LoadAsset { .. } => "LoadAsset",
            Self::UnloadAsset { .. } => "UnloadAsset",
            Self::CreateScene => "CreateScene",
            Self::DestroyScene { .. } => "DestroyScene",
            Self::SetDebugOptions { .. } => "SetDebugOptions",
            Self::SetFramebufferReadback { .. } => "SetFramebufferReadback",
            Self::SetQualitySettings { .. } => "SetQualitySettings",
            Self::SubmitFrame { .. } => "SubmitFrame",
            Self::CaptureScreenshot { .. } => "CaptureScreenshot",
            Self::Shutdown => "Shutdown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetHandle {
    Scene(SceneHandle),
    Mesh(MeshHandle),
    Material(MaterialHandle),
    Texture(TextureHandle),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedAsset {
    pub scene: Option<SceneHandle>,
    pub meshes: Vec<MeshHandle>,
    pub materials: Vec<MaterialHandle>,
    pub textures: Vec<TextureHandle>,
    pub bounds: Option<SceneBounds>,
}

impl LoadedAsset {
    /// Creates an asset load receipt with the handles produced by GPU upload.
    pub fn new(
        scene: Option<SceneHandle>,
        meshes: Vec<MeshHandle>,
        materials: Vec<MaterialHandle>,
        textures: Vec<TextureHandle>,
        bounds: Option<SceneBounds>,
    ) -> Self {
        Self {
            scene,
            meshes,
            materials,
            textures,
            bounds,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneBounds {
    center: [f32; 3],
    radius: f32,
}

impl SceneBounds {
    /// Creates a finite bounding sphere copied from imported CPU scene bounds.
    pub fn new(center: [f32; 3], radius: f32) -> Option<Self> {
        let finite_center = center.iter().all(|value| value.is_finite());
        (finite_center && radius.is_finite() && radius > 0.0).then_some(Self { center, radius })
    }

    /// Returns the world-space center used by app-side camera framing.
    pub fn center(self) -> [f32; 3] {
        self.center
    }

    /// Returns the positive world-space radius used by app-side camera framing.
    pub fn radius(self) -> f32 {
        self.radius
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DropReason {
    NoSurface {
        surface_id: SurfaceId,
    },
    StaleSurfaceGeneration {
        surface_id: SurfaceId,
        submitted: SurfaceGeneration,
        current: SurfaceGeneration,
    },
    SwapchainOutOfDate {
        surface_id: SurfaceId,
    },
}

impl DropReason {
    /// Returns the compact drop reason name used by trace logs and diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NoSurface { .. } => "NoSurface",
            Self::StaleSurfaceGeneration { .. } => "StaleSurfaceGeneration",
            Self::SwapchainOutOfDate { .. } => "SwapchainOutOfDate",
        }
    }
}

#[derive(Clone, Debug)]
pub enum RendererEvent {
    RendererReady,
    AssetLoaded {
        request_id: Option<RequestId>,
        asset: LoadedAsset,
    },
    AssetLoadFailed {
        request_id: Option<RequestId>,
        reason: String,
    },
    FramePresented {
        frame_id: FrameId,
    },
    FrameDropped {
        frame_id: FrameId,
        reason: DropReason,
    },
    FramebufferReadback {
        readback: FramebufferReadback,
    },
    SurfaceConfigured {
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
        platform: NativeSurfacePlatform,
    },
    SurfaceResized {
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
    },
    ScreenshotReady {
        request_id: Option<RequestId>,
        path: PathBuf,
    },
    ShaderReloaded,
    ShaderReloadFailed {
        reason: String,
    },
    ValidationWarning {
        message: String,
    },
    DeviceLost {
        reason: String,
    },
    RendererStopped,
}

impl RendererEvent {
    /// Returns the event name used by trace logs and protocol diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RendererReady => "RendererReady",
            Self::AssetLoaded { .. } => "AssetLoaded",
            Self::AssetLoadFailed { .. } => "AssetLoadFailed",
            Self::FramePresented { .. } => "FramePresented",
            Self::FrameDropped { .. } => "FrameDropped",
            Self::FramebufferReadback { .. } => "FramebufferReadback",
            Self::SurfaceConfigured { .. } => "SurfaceConfigured",
            Self::SurfaceResized { .. } => "SurfaceResized",
            Self::ScreenshotReady { .. } => "ScreenshotReady",
            Self::ShaderReloaded => "ShaderReloaded",
            Self::ShaderReloadFailed { .. } => "ShaderReloadFailed",
            Self::ValidationWarning { .. } => "ValidationWarning",
            Self::DeviceLost { .. } => "DeviceLost",
            Self::RendererStopped => "RendererStopped",
        }
    }
}

pub type RendererCommandEnvelope = MessageEnvelope<RendererCommand>;
pub type RendererEventEnvelope = MessageEnvelope<RendererEvent>;
