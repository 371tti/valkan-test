mod assets;
mod graph;
mod pipeline;
mod surface;
mod vulkan;

use std::{io, thread};

use async_trait::async_trait;
use thiserror::Error;

use crate::{
    import::import_asset_on_worker,
    protocol::{MessageEnvelope, RendererCommand, RendererEvent, RendererInbox, TransportError},
};

use self::assets::GpuAssetStore;
use self::surface::SurfaceRegistry;
pub use self::vulkan::{VulkanError, VulkanRendererBackend};

pub(crate) const SHADOW_MAP_SIZE: u32 = 2048;
pub(crate) const SHADOW_WORLD_SIZE: f32 = 70.0;
pub(crate) const SHADOW_VIEW_DISTANCE: f32 = 80.0;
pub(crate) const SHADOW_NEAR_PLANE: f32 = 1.0;
pub(crate) const SHADOW_FAR_PLANE: f32 = 180.0;

pub type RendererResult = Result<(), RendererError>;

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("failed to build renderer async runtime: {0}")]
    RuntimeBuild(io::Error),
    #[error("failed to spawn renderer thread: {0}")]
    ThreadSpawn(io::Error),
    #[error("renderer thread panicked")]
    ThreadPanic,
    #[error("renderer transport failed: {0}")]
    Transport(#[from] TransportError),
    #[error("vulkan renderer failed: {0}")]
    Vulkan(#[from] VulkanError),
}

#[async_trait]
pub trait RendererBackend: Send + 'static {
    /// Runs the renderer backend until command input closes or shutdown is received.
    async fn run(self, inbox: RendererInbox) -> RendererResult;
}

pub struct RendererThread {
    join: thread::JoinHandle<RendererResult>,
}

impl RendererThread {
    /// Returns whether the renderer thread has finished without consuming the handle.
    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }

    /// Waits for the renderer thread and returns its renderer result.
    pub fn join(self) -> RendererResult {
        tracing::info!("joining renderer thread");
        let result = self.join.join().map_err(|_| RendererError::ThreadPanic)?;
        tracing::info!("renderer thread joined");
        result
    }
}

/// Spawns a dedicated renderer thread and runs one backend inside it.
pub fn spawn_renderer_thread<B>(
    name: &'static str,
    backend: B,
    inbox: RendererInbox,
) -> Result<RendererThread, RendererError>
where
    B: RendererBackend,
{
    tracing::info!(thread = name, "spawning renderer thread");

    let join = thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || run_backend_on_thread(backend, inbox))
        .map_err(RendererError::ThreadSpawn)?;

    Ok(RendererThread { join })
}

/// Builds a thread-local async runtime and runs the renderer backend on it.
fn run_backend_on_thread<B>(backend: B, inbox: RendererInbox) -> RendererResult
where
    B: RendererBackend,
{
    tracing::trace!("building renderer async runtime");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(RendererError::RuntimeBuild)?;

    tracing::info!("renderer async runtime started");
    runtime.block_on(backend.run(inbox))
}

#[derive(Default)]
pub struct NullRendererBackend;

#[async_trait]
impl RendererBackend for NullRendererBackend {
    /// Processes protocol commands without creating Vulkan objects or presenting images.
    async fn run(self, mut inbox: RendererInbox) -> RendererResult {
        tracing::info!("null renderer backend starting");
        let mut assets = GpuAssetStore::default();
        let mut surfaces = SurfaceRegistry::default();

        inbox
            .send_event(MessageEnvelope::new(RendererEvent::RendererReady))
            .await?;
        tracing::info!("null renderer backend ready");

        while let Some(command) = inbox.recv_command().await {
            match command.payload {
                RendererCommand::ConfigureSurface { surface } => {
                    tracing::trace!(
                        window_id = surface.window_id.raw(),
                        surface_id = surface.surface_id.raw(),
                        generation = surface.generation.raw(),
                        width = surface.extent.width(),
                        height = surface.extent.height(),
                        platform = surface.native.platform().name(),
                        "configuring renderer surface"
                    );
                    let configured = surfaces.configure(surface);
                    let event = RendererEvent::SurfaceConfigured {
                        surface_id: configured.surface_id(),
                        generation: configured.generation(),
                        extent: configured.extent(),
                        platform: configured.native().platform(),
                    };
                    inbox.send_event(MessageEnvelope::new(event)).await?;
                }
                RendererCommand::ResizeSurface {
                    surface_id,
                    generation,
                    extent,
                } => {
                    tracing::trace!(
                        surface_id = surface_id.raw(),
                        generation = generation.raw(),
                        width = extent.width(),
                        height = extent.height(),
                        "resizing renderer surface"
                    );
                    if surfaces.resize(surface_id, generation, extent) {
                        let event = RendererEvent::SurfaceResized {
                            surface_id,
                            generation,
                            extent,
                        };
                        inbox.send_event(MessageEnvelope::new(event)).await?;
                    } else {
                        tracing::trace!(
                            surface_id = surface_id.raw(),
                            "ignored resize for unknown renderer surface"
                        );
                        let event = RendererEvent::ValidationWarning {
                            message: format!(
                                "resize ignored for unknown surface {}",
                                surface_id.raw()
                            ),
                        };
                        inbox.send_event(MessageEnvelope::new(event)).await?;
                    }
                }
                RendererCommand::SubmitFrame { snapshot } => {
                    tracing::trace!(
                        frame_id = snapshot.frame_id.raw(),
                        views = snapshot.views.len(),
                        draws = snapshot.draws.len(),
                        render_items = snapshot.render_items.len(),
                        lights = snapshot.lights.len(),
                        "presenting null renderer frame"
                    );
                    if let Some(reason) =
                        surfaces.frame_drop_reason(snapshot.surface_id, snapshot.surface_generation)
                    {
                        let event = MessageEnvelope::new(RendererEvent::FrameDropped {
                            frame_id: snapshot.frame_id,
                            reason,
                        })
                        .with_frame_id(snapshot.frame_id);
                        inbox.send_event(event).await?;
                        continue;
                    }

                    let event = MessageEnvelope::new(RendererEvent::FramePresented {
                        frame_id: snapshot.frame_id,
                    })
                    .with_frame_id(snapshot.frame_id);
                    inbox.send_event(event).await?;
                    assets.collect_deferred_destroys();
                }
                RendererCommand::LoadAsset { path } => {
                    tracing::trace!(path = %path.display(), "null renderer loading asset");
                    let event = match import_asset_on_worker(path).await {
                        Ok(imported) => {
                            let asset = assets.upload_imported_scene(&imported);
                            RendererEvent::AssetLoaded {
                                request_id: command.request_id,
                                asset,
                            }
                        }
                        Err(error) => RendererEvent::AssetLoadFailed {
                            request_id: command.request_id,
                            reason: error.to_string(),
                        },
                    };
                    inbox.send_event(MessageEnvelope::new(event)).await?;
                }
                RendererCommand::UnloadAsset { asset } => {
                    tracing::trace!(asset = ?asset, "null renderer unloading asset");
                    if !assets.unload(asset) {
                        let event = RendererEvent::ValidationWarning {
                            message: format!("asset unload ignored for stale handle: {asset:?}"),
                        };
                        inbox.send_event(MessageEnvelope::new(event)).await?;
                    }
                }
                RendererCommand::Shutdown => {
                    tracing::info!("null renderer backend stopping");
                    inbox
                        .send_event(MessageEnvelope::new(RendererEvent::RendererStopped))
                        .await?;
                    break;
                }
                other => {
                    tracing::trace!(command = other.name(), "null renderer ignored command");
                }
            }
        }

        tracing::info!("null renderer backend stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, num::NonZeroIsize};

    use crate::protocol::{
        DropReason, FrameId, FrameSnapshotBuilder, MessageEnvelope, NativeSurfaceHandle,
        NonZeroExtent, RendererCommand, RendererEvent, RequestId, SceneHandle, SurfaceDescriptor,
        SurfaceGeneration, SurfaceId, ViewId, ViewPacket, Win32SurfaceHandle, WindowId,
        renderer_transport,
    };

    use super::*;

    // Verifies that the null renderer presents a submitted validated frame snapshot.
    #[tokio::test]
    async fn null_renderer_presents_submitted_frame() {
        let (mut endpoint, inbox) = renderer_transport(8);
        let thread = spawn_renderer_thread("test-renderer", NullRendererBackend, inbox)
            .expect("renderer thread should spawn");

        let frame = FrameId::from_raw(1).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene id is non-zero");
        let window_id = WindowId::from_raw(1).expect("test window id is non-zero");
        let surface_id = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let extent = NonZeroExtent::new(16, 16).expect("test extent is non-zero");
        let surface = test_surface(window_id, surface_id, generation, extent);
        let mut builder = FrameSnapshotBuilder::new(frame, scene, surface_id, generation);
        builder.add_view(ViewPacket::new(view, extent));
        let snapshot = builder.build().expect("snapshot has one view");

        endpoint
            .send(MessageEnvelope::new(RendererCommand::ConfigureSurface {
                surface,
            }))
            .await
            .expect("command channel should accept surface config");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::SubmitFrame {
                snapshot,
            }))
            .await
            .expect("command channel should accept frame");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("command channel should accept shutdown");

        let mut presented = false;
        while let Some(event) = endpoint.recv_event().await {
            match event.payload {
                RendererEvent::FramePresented { frame_id } => {
                    presented = frame_id == frame;
                }
                RendererEvent::RendererStopped => break,
                _ => {}
            }
        }

        thread.join().expect("renderer thread should exit cleanly");
        assert!(presented);
    }

    // Verifies that surface lifecycle commands are processed on the renderer thread.
    #[tokio::test]
    async fn null_renderer_tracks_surface_lifecycle() {
        let (mut endpoint, inbox) = renderer_transport(8);
        let thread = spawn_renderer_thread("test-renderer", NullRendererBackend, inbox)
            .expect("renderer thread should spawn");
        let window_id = WindowId::from_raw(1).expect("test window id is non-zero");
        let surface_id = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let first_generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");
        let resized_generation =
            SurfaceGeneration::from_raw(2).expect("test generation is non-zero");
        let initial_extent = NonZeroExtent::new(16, 16).expect("test extent is non-zero");
        let resized_extent = NonZeroExtent::new(32, 24).expect("test extent is non-zero");
        let surface = test_surface(window_id, surface_id, first_generation, initial_extent);

        endpoint
            .send(MessageEnvelope::new(RendererCommand::ConfigureSurface {
                surface,
            }))
            .await
            .expect("command channel should accept surface config");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::ResizeSurface {
                surface_id,
                generation: resized_generation,
                extent: resized_extent,
            }))
            .await
            .expect("command channel should accept resize");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("command channel should accept shutdown");

        let mut configured = false;
        let mut resized = false;
        while let Some(event) = endpoint.recv_event().await {
            match event.payload {
                RendererEvent::SurfaceConfigured {
                    surface_id: configured_id,
                    generation,
                    extent,
                    ..
                } => {
                    configured = configured_id == surface_id
                        && generation == first_generation
                        && extent == initial_extent;
                }
                RendererEvent::SurfaceResized {
                    surface_id: resized_id,
                    generation,
                    extent,
                } => {
                    resized = resized_id == surface_id
                        && generation == resized_generation
                        && extent == resized_extent;
                }
                RendererEvent::RendererStopped => break,
                _ => {}
            }
        }

        thread.join().expect("renderer thread should exit cleanly");
        assert!(configured);
        assert!(resized);
    }

    // Verifies that stale snapshots are reported as drops instead of being presented.
    #[tokio::test]
    async fn null_renderer_drops_stale_surface_generation() {
        let (mut endpoint, inbox) = renderer_transport(8);
        let thread = spawn_renderer_thread("test-renderer", NullRendererBackend, inbox)
            .expect("renderer thread should spawn");
        let window_id = WindowId::from_raw(1).expect("test window id is non-zero");
        let surface_id = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let current_generation =
            SurfaceGeneration::from_raw(2).expect("test generation is non-zero");
        let stale_generation = SurfaceGeneration::from_raw(1).expect("test generation is non-zero");
        let extent = NonZeroExtent::new(16, 16).expect("test extent is non-zero");
        let surface = test_surface(window_id, surface_id, current_generation, extent);
        let frame = FrameId::from_raw(1).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene id is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let mut builder = FrameSnapshotBuilder::new(frame, scene, surface_id, stale_generation);
        builder.add_view(ViewPacket::new(view, extent));
        let snapshot = builder.build().expect("snapshot has one view");

        endpoint
            .send(MessageEnvelope::new(RendererCommand::ConfigureSurface {
                surface,
            }))
            .await
            .expect("command channel should accept surface config");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::SubmitFrame {
                snapshot,
            }))
            .await
            .expect("command channel should accept frame");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("command channel should accept shutdown");

        let mut dropped = false;
        while let Some(event) = endpoint.recv_event().await {
            match event.payload {
                RendererEvent::FrameDropped { reason, .. } => {
                    dropped = matches!(
                        reason,
                        DropReason::StaleSurfaceGeneration {
                            surface_id: dropped_surface,
                            submitted,
                            current,
                        } if dropped_surface == surface_id
                            && submitted == stale_generation
                            && current == current_generation
                    );
                }
                RendererEvent::RendererStopped => break,
                _ => {}
            }
        }

        thread.join().expect("renderer thread should exit cleanly");
        assert!(dropped);
    }

    // Verifies that explicit Stage 5 asset manifests produce handles without fallback geometry.
    #[tokio::test]
    async fn null_renderer_loads_explicit_asset_manifest() {
        let path = temp_scene_path("load");
        fs::write(
            &path,
            "rebuild1-scene\ntexture solid 255 255 255 255\nmaterial cutout base_color=0 alpha_cutoff=0.5\nmesh plane\n",
        )
        .expect("test scene manifest should be writable");
        let (mut endpoint, inbox) = renderer_transport(8);
        let thread = spawn_renderer_thread("test-renderer", NullRendererBackend, inbox)
            .expect("renderer thread should spawn");
        let request_id = RequestId::from_raw(1).expect("test request id is non-zero");

        endpoint
            .send(
                MessageEnvelope::new(RendererCommand::LoadAsset { path: path.clone() })
                    .with_request_id(request_id),
            )
            .await
            .expect("command channel should accept asset load");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("command channel should accept shutdown");

        let mut loaded = false;
        while let Some(event) = endpoint.recv_event().await {
            match event.payload {
                RendererEvent::AssetLoaded {
                    request_id: Some(id),
                    asset,
                } => {
                    loaded = id == request_id
                        && asset.scene.is_some()
                        && asset.meshes.len() == 1
                        && asset.materials.len() == 1
                        && asset.textures.len() == 1;
                }
                RendererEvent::RendererStopped => break,
                _ => {}
            }
        }

        thread.join().expect("renderer thread should exit cleanly");
        let _ = fs::remove_file(path);
        assert!(loaded);
    }

    // Verifies that missing assets return failure instead of a hidden cube or placeholder.
    #[tokio::test]
    async fn null_renderer_reports_missing_asset() {
        let (mut endpoint, inbox) = renderer_transport(8);
        let thread = spawn_renderer_thread("test-renderer", NullRendererBackend, inbox)
            .expect("renderer thread should spawn");
        let request_id = RequestId::from_raw(1).expect("test request id is non-zero");
        let path = temp_scene_path("missing");

        endpoint
            .send(
                MessageEnvelope::new(RendererCommand::LoadAsset { path })
                    .with_request_id(request_id),
            )
            .await
            .expect("command channel should accept asset load");
        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("command channel should accept shutdown");

        let mut failed = false;
        while let Some(event) = endpoint.recv_event().await {
            match event.payload {
                RendererEvent::AssetLoadFailed {
                    request_id: Some(id),
                    reason,
                } => {
                    failed = id == request_id && reason.contains("does not exist");
                }
                RendererEvent::RendererStopped => break,
                _ => {}
            }
        }

        thread.join().expect("renderer thread should exit cleanly");
        assert!(failed);
    }

    /// Creates a sendable test surface descriptor with non-zero Win32 handles.
    fn test_surface(
        window_id: WindowId,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
    ) -> SurfaceDescriptor {
        let native = NativeSurfaceHandle::Win32(Win32SurfaceHandle::new(
            NonZeroIsize::new(1).expect("test hwnd is non-zero"),
            None,
        ));
        SurfaceDescriptor::new(window_id, surface_id, generation, extent, native)
    }

    /// Returns a process-unique temporary scene manifest path for renderer tests.
    fn temp_scene_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rebuild1-{label}-{}.r1scene", std::process::id()))
    }
}
