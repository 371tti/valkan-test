use std::num::NonZeroIsize;

use thiserror::Error;

use crate::{
    protocol::{
        DrawPacket, FrameId, FrameSnapshot, FrameSnapshotBuilder, MessageEnvelope,
        NativeSurfaceHandle, NonZeroExtent, RendererCommand, RendererEndpoint, RendererEvent,
        SceneHandle, SurfaceDescriptor, SurfaceGeneration, SurfaceId, TransportError, ViewId,
        ViewPacket, Win32SurfaceHandle, WindowId, renderer_transport,
    },
    renderer::{NullRendererBackend, RendererError, spawn_renderer_thread},
};

#[derive(Debug, Error)]
pub enum HeadlessRunError {
    #[error("failed to communicate with renderer: {0}")]
    Transport(#[from] TransportError),
    #[error("renderer task failed: {0}")]
    Renderer(#[from] RendererError),
    #[error("renderer event stream closed before {0}")]
    EventStreamClosed(&'static str),
    #[error("headless frame was not presented")]
    FrameNotPresented,
    #[error("failed to build frame snapshot: {0}")]
    Snapshot(#[from] crate::protocol::SnapshotError),
}

/// Runs one protocol-driven frame through the null renderer backend.
pub async fn run_headless_once() -> Result<(), HeadlessRunError> {
    tracing::info!("starting headless renderer run");

    let (mut endpoint, inbox) = renderer_transport(16);
    let renderer = spawn_renderer_thread("rebuild1-renderer", NullRendererBackend, inbox)?;

    wait_until_ready(&mut endpoint).await?;

    let surface = headless_surface_descriptor();
    endpoint
        .send(MessageEnvelope::new(RendererCommand::ConfigureSurface {
            surface,
        }))
        .await?;

    let snapshot = build_minimal_snapshot()?;
    let frame_id = snapshot.frame_id;
    tracing::trace!(frame_id = frame_id.raw(), "submitting headless frame");

    endpoint
        .send(MessageEnvelope::new(RendererCommand::SubmitFrame {
            snapshot,
        }))
        .await?;
    tracing::trace!("requesting renderer shutdown after headless frame");
    endpoint
        .send(MessageEnvelope::new(RendererCommand::Shutdown))
        .await?;

    let presented = wait_until_stopped(&mut endpoint, frame_id).await?;
    renderer.join()?;

    if presented {
        tracing::info!(frame_id = frame_id.raw(), "completed headless renderer run");
        Ok(())
    } else {
        tracing::info!(
            frame_id = frame_id.raw(),
            "headless renderer run did not present frame"
        );
        Err(HeadlessRunError::FrameNotPresented)
    }
}

/// Waits until the renderer announces that it can receive commands.
async fn wait_until_ready(endpoint: &mut RendererEndpoint) -> Result<(), HeadlessRunError> {
    tracing::trace!("waiting for renderer ready event");

    while let Some(event) = endpoint.recv_event().await {
        if matches!(event.payload, RendererEvent::RendererReady) {
            tracing::trace!("renderer ready event received");
            return Ok(());
        }
    }

    Err(HeadlessRunError::EventStreamClosed("RendererReady"))
}

/// Waits until the renderer stops and reports whether the requested frame presented.
async fn wait_until_stopped(
    endpoint: &mut RendererEndpoint,
    frame_id: FrameId,
) -> Result<bool, HeadlessRunError> {
    tracing::trace!(
        frame_id = frame_id.raw(),
        "waiting for renderer stopped event"
    );
    let mut presented = false;

    while let Some(event) = endpoint.recv_event().await {
        match event.payload {
            RendererEvent::FramePresented {
                frame_id: presented_id,
            } => {
                tracing::trace!(
                    expected_frame_id = frame_id.raw(),
                    presented_frame_id = presented_id.raw(),
                    "renderer presented frame"
                );
                presented |= presented_id == frame_id;
            }
            RendererEvent::RendererStopped => {
                tracing::trace!(presented, "renderer stopped event received");
                return Ok(presented);
            }
            _ => {}
        }
    }

    Err(HeadlessRunError::EventStreamClosed("RendererStopped"))
}

/// Builds the smallest valid frame snapshot accepted by the protocol.
fn build_minimal_snapshot() -> Result<FrameSnapshot, HeadlessRunError> {
    tracing::trace!("building minimal headless frame snapshot");

    let frame = FrameId::from_raw(1).expect("literal frame id is non-zero");
    let scene = SceneHandle::from_raw(1).expect("literal scene handle is non-zero");
    let surface = SurfaceId::from_raw(1).expect("literal surface id is non-zero");
    let generation = SurfaceGeneration::from_raw(1).expect("literal generation is non-zero");
    let view = ViewId::from_raw(1).expect("literal view id is non-zero");
    let extent = NonZeroExtent::new(640, 360).expect("literal extent is non-zero");

    let mut builder = FrameSnapshotBuilder::new(frame, scene, surface, generation);
    builder.add_view(ViewPacket::new(view, extent));
    builder.add_draw(DrawPacket::debug_triangle());

    Ok(builder.build()?)
}

/// Builds the fixed surface descriptor used by the null renderer headless path.
fn headless_surface_descriptor() -> SurfaceDescriptor {
    let window = WindowId::from_raw(1).expect("literal window id is non-zero");
    let surface = SurfaceId::from_raw(1).expect("literal surface id is non-zero");
    let generation = SurfaceGeneration::from_raw(1).expect("literal generation is non-zero");
    let extent = NonZeroExtent::new(640, 360).expect("literal extent is non-zero");
    let native = NativeSurfaceHandle::Win32(Win32SurfaceHandle::new(
        NonZeroIsize::new(1).expect("literal hwnd is non-zero"),
        None,
    ));

    SurfaceDescriptor::new(window, surface, generation, extent, native)
}
