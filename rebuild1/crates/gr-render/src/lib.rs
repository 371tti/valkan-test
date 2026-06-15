#![deny(unsafe_op_in_unsafe_fn)]

pub mod import;
mod math;
pub mod protocol;
pub mod renderer;

pub use renderer::{
    NullRendererBackend, RendererBackend, RendererError, RendererResult, RendererThread,
    VulkanError, VulkanRendererBackend, spawn_renderer_thread,
};

/// Stable imports for applications and ECS extraction layers using the renderer protocol.
pub mod prelude {
    pub use crate::protocol::{
        AntiAliasingQualitySettings, CameraEffects, CameraSnapshot, CommandSink, DropReason,
        Exposure, FrameId, FrameSnapshot, FrameSnapshotBuilder, FramebufferMetering,
        FramebufferReadbackOptions, LightPacket, LoadedAsset, MaterialAlphaMode,
        MaterialDescriptor, MaterialHandle, MaterialTextureSlot, MeshHandle, MessageEnvelope,
        NativeSurfaceHandle, NonZeroExtent, PostQualitySettings, RenderItemPacket,
        RenderQualitySettings, RendererCommand, RendererEndpoint, RendererEvent,
        RendererEventEnvelope, RequestId, SceneBounds, SceneHandle, ShadowSofteningQualitySettings,
        SnapshotError, SsaoQualitySettings, SsrQualitySettings, SurfaceDescriptor,
        SurfaceGeneration, SurfaceId, TextureDescriptor, TextureHandle, TransportError, ViewId,
        ViewPacket, Win32SurfaceHandle, WindowId, renderer_transport,
    };
    pub use crate::renderer::{
        NullRendererBackend, RendererBackend, RendererError, RendererResult, RendererThread,
        VulkanError, VulkanRendererBackend, spawn_renderer_thread,
    };
}
