use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use gr_render::{
    protocol::{
        FrameId, FrameSnapshot, FrameSnapshotBuilder, FramebufferReadbackOptions, LightPacket,
        LoadedAsset, MessageEnvelope, NativeSurfaceHandle, NonZeroExtent, RenderItemPacket,
        RenderQualitySettings, RendererCommand, RendererEndpoint, RendererEvent, SceneHandle,
        SnapshotError, SurfaceDescriptor, SurfaceGeneration, SurfaceId, TransportError, ViewId,
        ViewPacket, Win32SurfaceHandle, WindowId,
    },
    renderer::{RendererError, RendererThread, VulkanRendererBackend, spawn_renderer_thread},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use thiserror::Error;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId as WinitWindowId},
};

use super::camera::{CameraKey, FreeCamera};
use super::camera_effects::{CameraEffectController, CameraMetering};

const DEFAULT_WINDOW_ID: u64 = 1;
const DEFAULT_SURFACE_ID: u64 = 1;
const DEFAULT_SURFACE_GENERATION: u64 = 1;
const DEFAULT_SCENE_ID: u64 = 1;
const DEFAULT_VIEW_ID: u64 = 1;
const DEFAULT_WINDOW_WIDTH: u32 = 1280;
const DEFAULT_WINDOW_HEIGHT: u32 = 720;
const DEFAULT_WINDOW_TITLE: &str = "rebuild1";
const DEFAULT_MODEL_ASSET_PATH: &str = "assets/model.glb";
const DEFAULT_WINDOW_LIGHT_INTENSITY: f32 = 1.15;
const MIN_WINDOW_LIGHT_INTENSITY: f32 = 0.0;
const MAX_WINDOW_LIGHT_INTENSITY: f32 = 24.0;
const LIGHT_CHANGE_STOPS_PER_SECOND: f32 = 1.6;
const WINDOW_ASSET_ENV: &str = "REBUILD1_WINDOW_ASSET";
const WINDOW_TRANSPORT_CAPACITY: usize = 32;
const WINDOW_SMOKE_PRESENTED_FRAME_LIMIT: u64 = 6;
const SHUTDOWN_RETRY_SLEEP: Duration = Duration::from_millis(1);
const EVENT_PUMP_INTERVAL: Duration = Duration::from_millis(16);
const WINDOW_MAX_FPS_ENV: &str = "REBUILD1_MAX_FPS";
const DEFAULT_WINDOW_MAX_FPS: u32 = 120;
const MIN_WINDOW_MAX_FPS: u32 = 15;
const MAX_WINDOW_MAX_FPS: u32 = 240;
const INITIAL_WINDOW_QUALITY: WindowQualityPreset = WindowQualityPreset::Performance;

#[derive(Debug, Error)]
pub enum WindowedRunError {
    #[error("failed to run winit event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("failed to create native window: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("failed to read native window handle: {0}")]
    Handle(#[from] raw_window_handle::HandleError),
    #[error("native surface handles are not supported for {display} display and {window} window")]
    UnsupportedNativeSurface {
        display: &'static str,
        window: &'static str,
    },
    #[error("failed to communicate with renderer: {0}")]
    Transport(#[from] TransportError),
    #[error("renderer task failed: {0}")]
    Renderer(#[from] RendererError),
    #[error("failed to build window frame snapshot: {0}")]
    Snapshot(#[from] SnapshotError),
    #[error("window asset load failed: {reason}")]
    AssetLoadFailed { reason: String },
    #[error("static protocol id {name} was configured as zero")]
    StaticIdZero { name: &'static str },
}

#[derive(Clone, Debug)]
pub struct WindowConfig {
    title: String,
    extent: NonZeroExtent,
    window_id: WindowId,
    surface_id: SurfaceId,
    initial_generation: SurfaceGeneration,
    asset_path: Option<PathBuf>,
    capture_mouse_on_start: bool,
    presented_frame_limit: Option<u64>,
    frame_limit_requires_loaded_asset: bool,
}

impl WindowConfig {
    /// Creates a window config from already validated renderer-facing values.
    pub fn new(
        title: impl Into<String>,
        extent: NonZeroExtent,
        window_id: WindowId,
        surface_id: SurfaceId,
        initial_generation: SurfaceGeneration,
    ) -> Self {
        Self {
            title: title.into(),
            extent,
            window_id,
            surface_id,
            initial_generation,
            asset_path: None,
            capture_mouse_on_start: true,
            presented_frame_limit: None,
            frame_limit_requires_loaded_asset: false,
        }
    }

    /// Sets the app-owned asset path sent through the renderer protocol after surface setup.
    pub fn with_asset_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.asset_path = Some(path.into());
        self
    }

    /// Disables automatic cursor capture for non-interactive smoke runs.
    pub fn without_initial_mouse_capture(mut self) -> Self {
        self.capture_mouse_on_start = false;
        self
    }

    /// Stops the window loop after a bounded number of presented frames.
    pub fn with_presented_frame_limit(mut self, limit: u64) -> Self {
        self.presented_frame_limit = Some(limit.max(1));
        self
    }

    /// Makes the frame limit count frames only after the configured asset has loaded.
    pub fn require_loaded_asset_before_frame_limit(mut self) -> Self {
        self.frame_limit_requires_loaded_asset = true;
        self
    }

    /// Returns the initial physical window size requested from winit.
    fn physical_size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.extent.width(), self.extent.height())
    }
}

impl Default for WindowConfig {
    /// Creates the default single-window configuration for early renderer wiring.
    fn default() -> Self {
        let extent = NonZeroExtent::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            .expect("default window extent is non-zero");
        let window_id =
            WindowId::from_raw(DEFAULT_WINDOW_ID).expect("default window id is non-zero");
        let surface_id =
            SurfaceId::from_raw(DEFAULT_SURFACE_ID).expect("default surface id is non-zero");
        let generation = SurfaceGeneration::from_raw(DEFAULT_SURFACE_GENERATION)
            .expect("default surface generation is non-zero");

        let config = Self::new(
            DEFAULT_WINDOW_TITLE,
            extent,
            window_id,
            surface_id,
            generation,
        );

        match default_model_asset_path() {
            Some(path) => config.with_asset_path(path),
            None => config,
        }
    }
}

/// Runs a native winit window that talks to the Vulkan renderer through protocol commands.
pub fn run_windowed() -> Result<(), WindowedRunError> {
    run_windowed_with_config(WindowConfig::default())
}

/// Runs a bounded native window smoke check that exits after presenting real frames.
pub fn run_windowed_smoke() -> Result<(), WindowedRunError> {
    let config = WindowConfig::default()
        .without_initial_mouse_capture()
        .with_presented_frame_limit(WINDOW_SMOKE_PRESENTED_FRAME_LIMIT)
        .require_loaded_asset_before_frame_limit();
    run_windowed_with_config(config)
}

/// Runs a native winit window with an explicit protocol-facing window config.
pub fn run_windowed_with_config(config: WindowConfig) -> Result<(), WindowedRunError> {
    tracing::info!(
        window_id = config.window_id.raw(),
        width = config.extent.width(),
        height = config.extent.height(),
        title = %config.title,
        "starting windowed renderer run"
    );

    let (endpoint, inbox) = gr_render::protocol::renderer_transport(WINDOW_TRANSPORT_CAPACITY);
    let renderer = spawn_renderer_thread("rebuild1-renderer", VulkanRendererBackend, inbox)?;
    let event_loop = EventLoop::<WindowUserEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let _event_pump = WindowEventPump::start(event_loop.create_proxy());
    let mut app = WindowedApp::new(endpoint, config);

    tracing::trace!("entering winit event loop");
    let run_result = event_loop.run_app(&mut app);
    tracing::trace!("winit event loop returned");
    let shutdown_result = app.request_shutdown();
    let app_error = app.take_error();
    let shutdown_failed = shutdown_result.is_err();
    let renderer_result = if shutdown_failed {
        drop(app);
        renderer.join()
    } else {
        join_renderer_with_live_events(renderer, &mut app)
    };

    if let Some(error) = app_error {
        return Err(error);
    }

    run_result?;
    shutdown_result?;
    renderer_result?;

    tracing::info!("completed windowed renderer run");
    Ok(())
}

/// Joins the renderer while keeping the app-side event receiver drained.
fn join_renderer_with_live_events(
    renderer: RendererThread,
    app: &mut WindowedApp,
) -> Result<(), RendererError> {
    tracing::trace!("joining renderer while draining app-side events");

    while !renderer.is_finished() {
        app.drain_renderer_events()?;
        thread::sleep(Duration::from_millis(1));
    }

    app.drain_renderer_events()?;
    renderer.join()
}

struct WindowedApp {
    endpoint: RendererEndpoint,
    config: WindowConfig,
    window: Option<Window>,
    error: Option<WindowedRunError>,
    shutdown_sent: bool,
    shutdown_pending: bool,
    in_flight_resize: Option<ResizeRequest>,
    pending_resize: Option<ResizeRequest>,
    drawable_surface: Option<DrawableSurface>,
    frame_in_flight: bool,
    asset_request_sent: bool,
    loaded_asset: Option<LoadedAsset>,
    camera: FreeCamera,
    camera_effects: CameraEffectController,
    latest_framebuffer_metering: Option<CameraMetering>,
    quality_preset: WindowQualityPreset,
    light_intensity: f32,
    light_brighter: bool,
    light_darker: bool,
    last_frame_at: Instant,
    frame_interval: Duration,
    next_frame_at: Instant,
    mouse_captured: bool,
    presented_frames: u64,
    presented_loaded_asset_frames: u64,
    next_frame_raw: u64,
    next_surface_generation_raw: u64,
}

#[derive(Clone, Copy, Debug)]
enum WindowUserEvent {
    PumpRendererEvents,
}

struct WindowEventPump {
    running: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WindowQualityPreset {
    Performance,
    Interactive,
    Balanced,
    HighQuality,
}

impl WindowQualityPreset {
    fn from_key(physical_key: PhysicalKey) -> Option<Self> {
        match physical_key {
            PhysicalKey::Code(KeyCode::Digit1) | PhysicalKey::Code(KeyCode::Numpad1) => {
                Some(Self::Performance)
            }
            PhysicalKey::Code(KeyCode::Digit2) | PhysicalKey::Code(KeyCode::Numpad2) => {
                Some(Self::Interactive)
            }
            PhysicalKey::Code(KeyCode::Digit3) | PhysicalKey::Code(KeyCode::Numpad3) => {
                Some(Self::Balanced)
            }
            PhysicalKey::Code(KeyCode::Digit4) | PhysicalKey::Code(KeyCode::Numpad4) => {
                Some(Self::HighQuality)
            }
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Interactive => "interactive",
            Self::Balanced => "balanced",
            Self::HighQuality => "high_quality",
        }
    }

    fn settings(self) -> RenderQualitySettings {
        match self {
            Self::Performance => RenderQualitySettings::performance(),
            Self::Interactive => RenderQualitySettings::interactive(),
            Self::Balanced => RenderQualitySettings::balanced(),
            Self::HighQuality => RenderQualitySettings::high_quality(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DrawableSurface {
    extent: NonZeroExtent,
    generation: SurfaceGeneration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResizeRequest {
    extent: NonZeroExtent,
    generation: SurfaceGeneration,
}

impl WindowedApp {
    /// Creates the winit-side app state that owns the window and renderer endpoint.
    fn new(endpoint: RendererEndpoint, config: WindowConfig) -> Self {
        tracing::trace!(
            window_id = config.window_id.raw(),
            width = config.extent.width(),
            height = config.extent.height(),
            "creating windowed app state"
        );
        let next_surface_generation_raw = config
            .initial_generation
            .raw()
            .checked_add(1)
            .unwrap_or(DEFAULT_SURFACE_GENERATION);

        let now = Instant::now();
        let frame_interval = configured_frame_interval();
        tracing::info!(
            max_fps = (1.0 / frame_interval.as_secs_f64()).round() as u32,
            "configured window frame pacing"
        );

        Self {
            endpoint,
            config,
            window: None,
            error: None,
            shutdown_sent: false,
            shutdown_pending: false,
            in_flight_resize: None,
            pending_resize: None,
            drawable_surface: None,
            frame_in_flight: false,
            asset_request_sent: false,
            loaded_asset: None,
            camera: FreeCamera::default(),
            camera_effects: CameraEffectController::default(),
            latest_framebuffer_metering: None,
            quality_preset: INITIAL_WINDOW_QUALITY,
            light_intensity: DEFAULT_WINDOW_LIGHT_INTENSITY,
            light_brighter: false,
            light_darker: false,
            last_frame_at: now,
            frame_interval,
            next_frame_at: now,
            mouse_captured: false,
            presented_frames: 0,
            presented_loaded_asset_frames: 0,
            next_frame_raw: 1,
            next_surface_generation_raw,
        }
    }

    /// Returns a stored callback error so the outer runner can report it.
    fn take_error(&mut self) -> Option<WindowedRunError> {
        self.error.take()
    }

    /// Creates the OS window and sends the first surface configuration command.
    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), WindowedRunError> {
        if self.window.is_some() {
            tracing::trace!("window creation skipped because window already exists");
            return Ok(());
        }

        tracing::trace!("creating native window");
        let attrs = WindowAttributes::default()
            .with_title(self.config.title.clone())
            .with_inner_size(self.config.physical_size());
        let window = event_loop.create_window(attrs)?;
        let size = window.inner_size();
        tracing::info!(
            window_id = self.config.window_id.raw(),
            width = size.width,
            height = size.height,
            "native window created"
        );

        self.window = Some(window);
        if self.config.capture_mouse_on_start {
            self.set_mouse_captured(true);
        }
        self.configure_framebuffer_readback()?;
        self.configure_quality_preset(self.quality_preset)?;
        self.configure_current_surface()?;
        self.send_initial_asset_load()?;

        Ok(())
    }

    /// Requests renderer-owned final framebuffer metering for app-side camera effects.
    fn configure_framebuffer_readback(&self) -> Result<(), WindowedRunError> {
        self.send_command(RendererCommand::SetFramebufferReadback {
            options: FramebufferReadbackOptions::camera_metering(),
        })?;
        Ok(())
    }

    /// Sends the current window-selected renderer quality profile.
    fn configure_quality_preset(
        &self,
        preset: WindowQualityPreset,
    ) -> Result<(), WindowedRunError> {
        tracing::info!(
            quality = preset.label(),
            "sending window renderer quality preset"
        );
        self.send_command(RendererCommand::SetQualitySettings {
            settings: preset.settings(),
        })?;
        Ok(())
    }

    /// Sends the first surface descriptor for the current native window.
    fn configure_current_surface(&self) -> Result<(), WindowedRunError> {
        let Some(window) = &self.window else {
            return Ok(());
        };

        self.configure_surface(window)
    }

    /// Converts a winit physical size into a protocol extent and rejects minimized windows.
    fn extent_from_size(size: PhysicalSize<u32>) -> Option<NonZeroExtent> {
        NonZeroExtent::new(size.width, size.height)
    }

    /// Sends the initial surface descriptor when the native window is drawable.
    fn configure_surface(&self, window: &Window) -> Result<(), WindowedRunError> {
        let size = window.inner_size();
        let Some(extent) = Self::extent_from_size(size) else {
            tracing::trace!(
                width = size.width,
                height = size.height,
                "surface configure skipped for zero-sized window"
            );
            return Ok(());
        };

        let native = native_surface_handle(window)?;
        tracing::trace!(
            window_id = self.config.window_id.raw(),
            surface_id = self.config.surface_id.raw(),
            width = extent.width(),
            height = extent.height(),
            generation = self.config.initial_generation.raw(),
            platform = native.platform().name(),
            "sending surface configure command"
        );
        let surface = SurfaceDescriptor::new(
            self.config.window_id,
            self.config.surface_id,
            self.config.initial_generation,
            extent,
            native,
        );
        self.send_command(RendererCommand::ConfigureSurface { surface })?;
        Ok(())
    }

    /// Sends the optional app-selected model load request after surface configuration.
    fn send_initial_asset_load(&mut self) -> Result<(), WindowedRunError> {
        if self.asset_request_sent {
            return Ok(());
        }

        let Some(path) = self.config.asset_path.clone() else {
            tracing::trace!("window model asset load skipped because no asset path is configured");
            return Ok(());
        };

        tracing::info!(path = %path.display(), "requesting window model asset load");
        self.send_command(RendererCommand::LoadAsset { path })?;
        self.asset_request_sent = true;
        Ok(())
    }

    /// Sends a resize command when the native window has a drawable extent.
    fn resize_surface(&mut self, size: PhysicalSize<u32>) -> Result<(), TransportError> {
        let Some(extent) = Self::extent_from_size(size) else {
            tracing::trace!(
                width = size.width,
                height = size.height,
                "surface resize skipped for zero-sized window"
            );
            self.drawable_surface = None;
            return Ok(());
        };

        self.drawable_surface = None;
        tracing::trace!(
            surface_id = self.config.surface_id.raw(),
            width = extent.width(),
            height = extent.height(),
            "sending surface resize command"
        );
        self.queue_resize(extent)
    }

    /// Sends one resize immediately or stores the newest resize while one is in flight.
    fn queue_resize(&mut self, extent: NonZeroExtent) -> Result<(), TransportError> {
        if self
            .in_flight_resize
            .is_some_and(|request| request.extent == extent)
            || self
                .pending_resize
                .is_some_and(|request| request.extent == extent)
        {
            tracing::trace!(
                surface_id = self.config.surface_id.raw(),
                width = extent.width(),
                height = extent.height(),
                "surface resize skipped because the extent is already queued"
            );
            return Ok(());
        }

        let request = ResizeRequest {
            extent,
            generation: self.next_surface_generation(),
        };

        if self.in_flight_resize.is_some() {
            tracing::trace!(
                surface_id = self.config.surface_id.raw(),
                width = extent.width(),
                height = extent.height(),
                generation = request.generation.raw(),
                "stored latest resize while previous resize is in flight"
            );
            self.pending_resize = Some(request);
            return Ok(());
        }

        self.try_send_resize(request)
    }

    /// Requests another redraw when the renderer is ready to accept a frame.
    fn request_redraw_if_ready(&self) {
        if self.is_shutting_down() {
            return;
        }

        if self.drawable_surface.is_none() || self.frame_in_flight {
            return;
        }

        let now = Instant::now();
        if now < self.next_frame_at {
            return;
        }

        if let Some(window) = &self.window {
            tracing::trace!("requesting redraw for next protocol frame");
            window.request_redraw();
        }
    }

    /// Keeps winit asleep until either renderer events arrive or the next frame budget opens.
    fn apply_frame_pacing_control_flow(&self, event_loop: &ActiveEventLoop) {
        if self.is_shutting_down() || self.frame_in_flight || self.drawable_surface.is_none() {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }

        let now = Instant::now();
        if self.next_frame_at > now {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    /// Builds and sends one minimal frame snapshot for the current drawable surface.
    fn submit_frame(&mut self) -> Result<(), WindowedRunError> {
        if self.is_shutting_down() {
            tracing::trace!("frame submit skipped because shutdown is in progress");
            return Ok(());
        }

        if self.frame_in_flight {
            tracing::trace!("frame submit skipped because a frame is already in flight");
            return Ok(());
        }

        let Some(drawable) = self.drawable_surface else {
            tracing::trace!("frame submit skipped because no drawable extent is configured");
            return Ok(());
        };

        let delta_time = self.update_camera_for_frame();
        self.update_light_for_frame(delta_time);
        let snapshot = self.build_frame_snapshot(drawable, delta_time)?;
        let frame_id = snapshot.frame_id;
        let command =
            MessageEnvelope::new(RendererCommand::SubmitFrame { snapshot }).with_frame_id(frame_id);

        match self.endpoint.try_send(command) {
            Ok(()) => {
                self.frame_in_flight = true;
                self.next_frame_at = Instant::now() + self.frame_interval;
                tracing::trace!(
                    frame_id = frame_id.raw(),
                    surface_id = self.config.surface_id.raw(),
                    generation = drawable.generation.raw(),
                    width = drawable.extent.width(),
                    height = drawable.extent.height(),
                    "submitted window frame snapshot"
                );
                Ok(())
            }
            Err(TransportError::Full) => {
                tracing::trace!(
                    frame_id = frame_id.raw(),
                    "frame submit skipped because command channel is full"
                );
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Builds the smallest user/app frame snapshot needed for swapchain rendering.
    fn build_frame_snapshot(
        &mut self,
        drawable: DrawableSurface,
        delta_time: f32,
    ) -> Result<FrameSnapshot, WindowedRunError> {
        let frame_id = self.next_frame_id()?;
        let scene = SceneHandle::from_raw(DEFAULT_SCENE_ID)
            .ok_or(WindowedRunError::StaticIdZero { name: "scene" })?;
        let view = ViewId::from_raw(DEFAULT_VIEW_ID)
            .ok_or(WindowedRunError::StaticIdZero { name: "view" })?;

        let mut builder =
            FrameSnapshotBuilder::new(frame_id, scene, self.config.surface_id, drawable.generation);
        builder
            .add_view(ViewPacket::new(view, drawable.extent).with_camera(self.camera.snapshot()));
        let light = LightPacket::new(self.light_intensity);
        let screen_metering = self.screen_metering_for_frame();
        let camera_effects: gr_render::prelude::CameraEffects =
            self.camera_effects.update(screen_metering, delta_time);
        builder.add_light(light).set_camera_effects(camera_effects);
        self.add_loaded_model_items(&mut builder);
        Ok(builder.build()?)
    }

    /// Returns the latest renderer framebuffer metering, or a neutral initial camera sample.
    fn screen_metering_for_frame(&self) -> CameraMetering {
        self.latest_framebuffer_metering
            .unwrap_or_else(CameraMetering::neutral_screen)
    }

    /// Adds every loaded mesh/material pair to the frame snapshot and returns the draw count.
    fn add_loaded_model_items(&self, builder: &mut FrameSnapshotBuilder) -> usize {
        let Some(asset) = self.loaded_asset.as_ref() else {
            return 0;
        };
        let mut count = 0;

        for (&mesh, &material) in asset.meshes.iter().zip(asset.materials.iter()) {
            builder.add_render_item(RenderItemPacket::new(mesh, material));
            count += 1;
        }

        count
    }

    /// Advances the app-side free camera once per submitted protocol frame.
    fn update_camera_for_frame(&mut self) -> f32 {
        let now = Instant::now();
        let delta_time = now
            .duration_since(self.last_frame_at)
            .as_secs_f32()
            .min(1.0 / 15.0);
        self.last_frame_at = now;
        self.camera.update(delta_time);
        delta_time
    }

    /// Updates app-owned light intensity from the arrow-key state before frame extraction.
    fn update_light_for_frame(&mut self, delta_time: f32) {
        if !self.light_brighter && !self.light_darker {
            return;
        }

        let previous = self.light_intensity;
        let multiplier = 2.0_f32.powf(delta_time.max(0.0) * LIGHT_CHANGE_STOPS_PER_SECOND);

        if self.light_brighter {
            self.light_intensity *= multiplier;
        }
        if self.light_darker {
            self.light_intensity /= multiplier;
        }

        self.light_intensity = self
            .light_intensity
            .clamp(MIN_WINDOW_LIGHT_INTENSITY, MAX_WINDOW_LIGHT_INTENSITY);

        if (self.light_intensity - previous).abs() > f32::EPSILON {
            tracing::trace!(
                light_intensity = self.light_intensity,
                "updated window light intensity"
            );
        }
    }

    /// Allocates the next non-zero protocol frame id for the window loop.
    fn next_frame_id(&mut self) -> Result<FrameId, WindowedRunError> {
        let raw = self.next_frame_raw;
        self.next_frame_raw = match raw.checked_add(1) {
            Some(next) => next,
            None => 1,
        };

        FrameId::from_raw(raw).ok_or(WindowedRunError::StaticIdZero { name: "frame" })
    }

    /// Allocates the next protocol surface generation for a resize request.
    fn next_surface_generation(&mut self) -> SurfaceGeneration {
        let raw = self.next_surface_generation_raw;
        self.next_surface_generation_raw = raw.checked_add(1).unwrap_or(DEFAULT_SURFACE_GENERATION);

        SurfaceGeneration::from_raw(raw).expect("surface generation counter never yields zero")
    }

    /// Tries to send one resize and keeps it pending when the bounded channel is full.
    fn try_send_resize(&mut self, request: ResizeRequest) -> Result<(), TransportError> {
        let result = self.send_command(RendererCommand::ResizeSurface {
            surface_id: self.config.surface_id,
            generation: request.generation,
            extent: request.extent,
        });

        match result {
            Ok(()) => {
                self.in_flight_resize = Some(request);
                Ok(())
            }
            Err(TransportError::Full) => {
                tracing::trace!(
                    surface_id = self.config.surface_id.raw(),
                    generation = request.generation.raw(),
                    width = request.extent.width(),
                    height = request.extent.height(),
                    "stored latest resize because command channel is full"
                );
                self.pending_resize = Some(request);
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// Sends the latest pending resize after the renderer acknowledges the previous one.
    fn flush_pending_resize(&mut self) -> Result<(), TransportError> {
        if self.is_shutting_down() {
            self.pending_resize = None;
            return Ok(());
        }

        if self.in_flight_resize.is_some() {
            return Ok(());
        }

        let Some(request) = self.pending_resize.take() else {
            return Ok(());
        };

        self.try_send_resize(request)
    }

    /// Sends one renderer command from the synchronous winit callback boundary.
    fn send_command(&self, command: RendererCommand) -> Result<(), TransportError> {
        self.endpoint.try_send(MessageEnvelope::new(command))
    }

    /// Returns whether the app is already winding down the renderer connection.
    fn is_shutting_down(&self) -> bool {
        self.shutdown_pending || self.shutdown_sent
    }

    /// Queues renderer shutdown once, draining events while waiting for command capacity.
    fn request_shutdown(&mut self) -> Result<(), TransportError> {
        if self.shutdown_sent {
            tracing::trace!("renderer shutdown request skipped because it was already sent");
            return Ok(());
        }

        if !self.shutdown_pending {
            tracing::info!("requesting renderer shutdown");
            self.shutdown_pending = true;
        }

        loop {
            match self.send_command(RendererCommand::Shutdown) {
                Ok(()) => {
                    self.shutdown_pending = false;
                    self.shutdown_sent = true;
                    return Ok(());
                }
                Err(TransportError::Full) => {
                    tracing::trace!("shutdown send is waiting for renderer command capacity");
                    self.drain_renderer_events()?;
                    thread::sleep(SHUTDOWN_RETRY_SLEEP);
                }
                Err(TransportError::Closed) => {
                    tracing::trace!(
                        "shutdown send skipped because renderer command channel closed"
                    );
                    self.shutdown_pending = false;
                    self.shutdown_sent = true;
                    return Ok(());
                }
            }
        }
    }

    /// Stores the first callback error and asks the event loop to stop.
    fn fail_and_exit(&mut self, error: WindowedRunError, event_loop: &ActiveEventLoop) {
        tracing::info!(error = %error, "windowed app exiting after callback error");

        if self.error.is_none() {
            self.error = Some(error);
        }

        let _ = self.request_shutdown();
        event_loop.exit();
    }

    /// Returns true when the incoming winit id belongs to the single owned window.
    fn owns_window(&self, window_id: WinitWindowId) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.id() == window_id)
    }

    /// Removes pending renderer events so the bounded event channel remains usable.
    fn drain_renderer_events(&mut self) -> Result<(), TransportError> {
        let mut drained = 0;

        while let Some(event) = self.endpoint.try_recv_event() {
            let frame_id = event.frame_id;

            match event.payload {
                RendererEvent::SurfaceConfigured {
                    surface_id,
                    generation,
                    extent,
                    ..
                } if surface_id == self.config.surface_id => {
                    tracing::trace!(
                        surface_id = surface_id.raw(),
                        generation = generation.raw(),
                        width = extent.width(),
                        height = extent.height(),
                        "renderer configured drawable surface"
                    );
                    self.drawable_surface = Some(DrawableSurface { extent, generation });
                    self.frame_in_flight = false;
                }
                RendererEvent::SurfaceResized {
                    surface_id,
                    generation,
                    extent,
                } if surface_id == self.config.surface_id => {
                    tracing::trace!(
                        surface_id = surface_id.raw(),
                        generation = generation.raw(),
                        width = extent.width(),
                        height = extent.height(),
                        "renderer completed surface resize"
                    );
                    self.drawable_surface = Some(DrawableSurface { extent, generation });
                    self.frame_in_flight = false;
                    self.in_flight_resize = None;
                    if self
                        .pending_resize
                        .is_some_and(|request| request.extent == extent)
                    {
                        self.pending_resize = None;
                    }
                }
                RendererEvent::FramePresented {
                    frame_id: presented_id,
                } => {
                    self.presented_frames += 1;
                    if self.loaded_asset.is_some() {
                        self.presented_loaded_asset_frames += 1;
                    }
                    tracing::trace!(
                        frame_id = presented_id.raw(),
                        presented_frames = self.presented_frames,
                        presented_loaded_asset_frames = self.presented_loaded_asset_frames,
                        "renderer presented window frame"
                    );
                    self.frame_in_flight = false;
                }
                RendererEvent::FrameDropped {
                    frame_id: dropped_id,
                    reason,
                } => {
                    tracing::trace!(
                        frame_id = dropped_id.raw(),
                        reason = reason.name(),
                        "renderer dropped window frame"
                    );
                    self.frame_in_flight = false;
                }
                RendererEvent::FramebufferReadback { readback }
                    if readback.surface_id == self.config.surface_id =>
                {
                    tracing::trace!(
                        frame_id = readback.frame_id.raw(),
                        generation = readback.generation.raw(),
                        width = readback.extent.width(),
                        height = readback.extent.height(),
                        average_luminance = readback.metering.average_luminance(),
                        center_luminance = readback.metering.center_luminance(),
                        highlight_fraction = readback.metering.highlight_fraction(),
                        "received framebuffer metering"
                    );
                    self.latest_framebuffer_metering =
                        Some(CameraMetering::from_framebuffer(readback.metering));
                }
                RendererEvent::AssetLoaded { asset, .. } => {
                    let bounds = asset.bounds;
                    tracing::info!(
                        scene = asset.scene.map(|scene| scene.raw()),
                        meshes = asset.meshes.len(),
                        materials = asset.materials.len(),
                        textures = asset.textures.len(),
                        bounds_center = ?bounds.map(|bounds| bounds.center()),
                        bounds_radius = ?bounds.map(|bounds| bounds.radius()),
                        "window model asset loaded"
                    );
                    if let Some(bounds) = bounds {
                        self.camera.frame_bounds(bounds);
                    }
                    self.loaded_asset = Some(asset);
                    self.frame_in_flight = false;
                }
                RendererEvent::AssetLoadFailed { reason, .. } => {
                    tracing::info!(reason, "window model asset load failed");
                    if self.config.frame_limit_requires_loaded_asset {
                        self.error = Some(WindowedRunError::AssetLoadFailed { reason });
                    }
                }
                RendererEvent::ValidationWarning { message } if frame_id.is_some() => {
                    tracing::trace!(
                        frame_id = frame_id.map(|id| id.raw()),
                        message,
                        "renderer skipped a submitted window frame"
                    );
                    self.frame_in_flight = false;
                }
                _ => {}
            }

            drained += 1;
        }

        if drained > 0 {
            tracing::trace!(drained, "drained renderer events");
        }

        self.flush_pending_resize()?;
        Ok(())
    }

    /// Returns true once a configured non-interactive run has presented enough frames.
    fn presented_frame_limit_reached(&self) -> bool {
        let Some(limit) = self.config.presented_frame_limit else {
            return false;
        };

        if self.config.frame_limit_requires_loaded_asset && self.config.asset_path.is_some() {
            self.presented_loaded_asset_frames >= limit
        } else {
            self.presented_frames >= limit
        }
    }

    /// Drains renderer events and schedules a redraw from a winit wake event.
    fn pump_renderer_events(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Err(error) = self.drain_renderer_events() {
            self.fail_and_exit(error.into(), event_loop);
            return;
        }

        if self.error.is_some() {
            let _ = self.request_shutdown();
            event_loop.exit();
            return;
        }

        if self.is_shutting_down() {
            return;
        }

        if self.presented_frame_limit_reached() {
            tracing::info!(
                presented_frames = self.presented_frames,
                presented_loaded_asset_frames = self.presented_loaded_asset_frames,
                "window smoke frame limit reached"
            );
            if let Err(error) = self.request_shutdown() {
                self.fail_and_exit(error.into(), event_loop);
                return;
            }
            event_loop.exit();
            return;
        }

        self.request_redraw_if_ready();
        self.apply_frame_pacing_control_flow(event_loop);
    }

    /// Captures or releases the mouse cursor for old-style free camera look controls.
    fn set_mouse_captured(&mut self, captured: bool) {
        let Some(window) = &self.window else {
            self.mouse_captured = captured;
            return;
        };

        if captured {
            let grab_result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(error) = grab_result {
                tracing::info!(error = %error, "failed to capture mouse cursor");
                self.mouse_captured = false;
                window.set_cursor_visible(true);
                return;
            }
        } else if let Err(error) = window.set_cursor_grab(CursorGrabMode::None) {
            tracing::info!(error = %error, "failed to release mouse cursor");
        }

        self.mouse_captured = captured;
        window.set_cursor_visible(!captured);
    }

    /// Converts a winit keyboard input into one quality, light, or camera state update.
    fn handle_keyboard(
        &mut self,
        physical_key: PhysicalKey,
        state: ElementState,
    ) -> Result<(), TransportError> {
        let pressed = state == ElementState::Pressed;

        if pressed {
            if let Some(preset) = WindowQualityPreset::from_key(physical_key) {
                self.set_quality_preset(preset)?;
                return Ok(());
            }
        }

        match physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.light_brighter = pressed;
                return Ok(());
            }
            PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.light_darker = pressed;
                return Ok(());
            }
            _ => {}
        }

        if physical_key == PhysicalKey::Code(KeyCode::Escape) && state == ElementState::Pressed {
            self.set_mouse_captured(false);
        }

        self.camera.set_key(camera_key(physical_key), pressed);
        Ok(())
    }

    /// Sends a renderer quality update selected by a number key.
    fn set_quality_preset(&mut self, preset: WindowQualityPreset) -> Result<(), TransportError> {
        if self.quality_preset == preset {
            tracing::trace!(
                quality = preset.label(),
                "window renderer quality preset unchanged"
            );
            return Ok(());
        }

        match self.send_command(RendererCommand::SetQualitySettings {
            settings: preset.settings(),
        }) {
            Ok(()) => {
                self.quality_preset = preset;
                tracing::info!(
                    quality = preset.label(),
                    "updated window renderer quality preset"
                );
                Ok(())
            }
            Err(TransportError::Full) => {
                tracing::trace!(
                    quality = preset.label(),
                    "quality preset update skipped because command channel is full"
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// Converts raw captured mouse movement into camera look deltas.
    fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        if !self.mouse_captured {
            return;
        }

        self.camera
            .add_mouse_delta([delta.0 as f32, delta.1 as f32]);
    }

    /// Converts mouse wheel input into camera movement speed changes.
    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(position) => (position.y as f32 / 32.0).clamp(-4.0, 4.0),
        };
        self.camera.adjust_speed_multiplier(delta);
    }
}

impl WindowEventPump {
    /// Starts a small wake thread so renderer channel events are drained without user input.
    fn start(proxy: EventLoopProxy<WindowUserEvent>) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let thread_running = Arc::clone(&running);
        let join = thread::spawn(move || {
            while thread_running.load(Ordering::Relaxed) {
                thread::sleep(EVENT_PUMP_INTERVAL);
                if proxy
                    .send_event(WindowUserEvent::PumpRendererEvents)
                    .is_err()
                {
                    break;
                }
            }
        });

        Self {
            running,
            join: Some(join),
        }
    }
}

impl Drop for WindowEventPump {
    /// Stops the wake thread when the winit event loop has returned.
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Returns the default model path when the app-level sample asset is present.
fn default_model_asset_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(WINDOW_ASSET_ENV).map(PathBuf::from) {
        return path.is_file().then_some(path);
    }

    let path = PathBuf::from(DEFAULT_MODEL_ASSET_PATH);
    path.is_file().then_some(path)
}

/// Returns the app-side frame pacing interval used to avoid rendering faster than needed.
fn configured_frame_interval() -> Duration {
    let fps = std::env::var(WINDOW_MAX_FPS_ENV)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(DEFAULT_WINDOW_MAX_FPS)
        .clamp(MIN_WINDOW_MAX_FPS, MAX_WINDOW_MAX_FPS);

    Duration::from_secs_f64(1.0 / fps as f64)
}

/// Copies winit raw handles into a sendable protocol surface handle.
fn native_surface_handle(window: &Window) -> Result<NativeSurfaceHandle, WindowedRunError> {
    let raw_display = window.display_handle()?.as_raw();
    let raw_window = window.window_handle()?.as_raw();
    tracing::trace!(
        display_handle = display_handle_name(raw_display),
        window_handle = window_handle_name(raw_window),
        "read native window handles"
    );

    match (raw_display, raw_window) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(handle)) => Ok(
            NativeSurfaceHandle::Win32(Win32SurfaceHandle::new(handle.hwnd, handle.hinstance)),
        ),
        (raw_display, raw_window) => Err(WindowedRunError::UnsupportedNativeSurface {
            display: display_handle_name(raw_display),
            window: window_handle_name(raw_window),
        }),
    }
}

/// Returns a compact platform name for a raw display handle.
fn display_handle_name(handle: RawDisplayHandle) -> &'static str {
    match handle {
        RawDisplayHandle::UiKit(_) => "uikit",
        RawDisplayHandle::AppKit(_) => "appkit",
        RawDisplayHandle::Orbital(_) => "orbital",
        RawDisplayHandle::Ohos(_) => "ohos",
        RawDisplayHandle::Xlib(_) => "xlib",
        RawDisplayHandle::Xcb(_) => "xcb",
        RawDisplayHandle::Wayland(_) => "wayland",
        RawDisplayHandle::Drm(_) => "drm",
        RawDisplayHandle::Gbm(_) => "gbm",
        RawDisplayHandle::Windows(_) => "windows",
        RawDisplayHandle::Web(_) => "web",
        RawDisplayHandle::Android(_) => "android",
        RawDisplayHandle::Haiku(_) => "haiku",
        _ => "unknown",
    }
}

/// Returns a compact platform name for a raw window handle.
fn window_handle_name(handle: RawWindowHandle) -> &'static str {
    match handle {
        RawWindowHandle::UiKit(_) => "uikit",
        RawWindowHandle::AppKit(_) => "appkit",
        RawWindowHandle::Orbital(_) => "orbital",
        RawWindowHandle::OhosNdk(_) => "ohos-ndk",
        RawWindowHandle::Xlib(_) => "xlib",
        RawWindowHandle::Xcb(_) => "xcb",
        RawWindowHandle::Wayland(_) => "wayland",
        RawWindowHandle::Drm(_) => "drm",
        RawWindowHandle::Gbm(_) => "gbm",
        RawWindowHandle::Win32(_) => "win32",
        RawWindowHandle::WinRt(_) => "winrt",
        RawWindowHandle::Web(_) => "web",
        RawWindowHandle::WebCanvas(_) => "web-canvas",
        RawWindowHandle::WebOffscreenCanvas(_) => "web-offscreen-canvas",
        RawWindowHandle::AndroidNdk(_) => "android-ndk",
        RawWindowHandle::Haiku(_) => "haiku",
        _ => "unknown",
    }
}

impl ApplicationHandler<WindowUserEvent> for WindowedApp {
    /// Creates the native window once the platform event loop allows it.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        tracing::trace!("winit resumed");

        if let Err(error) = self.create_window(event_loop) {
            self.fail_and_exit(error, event_loop);
        }
    }

    /// Converts window lifecycle changes into renderer protocol commands.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WinitWindowId,
        event: WindowEvent,
    ) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if !self.owns_window(window_id) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("window close requested");

                if let Err(error) = self.request_shutdown() {
                    self.fail_and_exit(error.into(), event_loop);
                    return;
                }

                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                tracing::trace!(width = size.width, height = size.height, "window resized");

                if let Err(error) = self.resize_surface(size) {
                    self.fail_and_exit(error.into(), event_loop);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Err(error) = self.handle_keyboard(event.physical_key, event.state) {
                    self.fail_and_exit(error.into(), event_loop);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.set_mouse_captured(true);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.submit_frame() {
                    self.fail_and_exit(error, event_loop);
                }
            }
            _ => {}
        }
    }

    /// Drains renderer events between platform event batches.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.pump_renderer_events(event_loop);
    }

    /// Wakes the app to drain renderer events even when the OS has no window events.
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: WindowUserEvent) {
        match event {
            WindowUserEvent::PumpRendererEvents => self.pump_renderer_events(event_loop),
        }
    }

    /// Routes raw device motion into the captured free camera.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.handle_mouse_motion(delta);
        }
    }
}

/// Maps physical keyboard keys to app-level camera controls.
fn camera_key(physical_key: PhysicalKey) -> CameraKey {
    match physical_key {
        PhysicalKey::Code(KeyCode::KeyW) => CameraKey::Forward,
        PhysicalKey::Code(KeyCode::KeyS) => CameraKey::Back,
        PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => CameraKey::Left,
        PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
            CameraKey::Right
        }
        PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::KeyQ) => CameraKey::Down,
        PhysicalKey::Code(KeyCode::Space) | PhysicalKey::Code(KeyCode::KeyE) => CameraKey::Up,
        PhysicalKey::Code(KeyCode::ControlLeft) => CameraKey::Sprint,
        PhysicalKey::Code(KeyCode::Escape) => CameraKey::Stop,
        _ => CameraKey::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use gr_render::protocol::renderer_transport;

    #[test]
    fn number_keys_select_window_quality_presets() {
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Digit1)),
            Some(WindowQualityPreset::Performance)
        );
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Numpad1)),
            Some(WindowQualityPreset::Performance)
        );
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Digit2)),
            Some(WindowQualityPreset::Interactive)
        );
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Digit3)),
            Some(WindowQualityPreset::Balanced)
        );
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Digit4)),
            Some(WindowQualityPreset::HighQuality)
        );
        assert_eq!(
            WindowQualityPreset::from_key(PhysicalKey::Code(KeyCode::Digit5)),
            None
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pressed_number_key_sends_quality_command() {
        let config = WindowConfig::default();
        let (endpoint, mut inbox) = renderer_transport(2);
        let mut app = WindowedApp::new(endpoint, config);

        app.handle_keyboard(PhysicalKey::Code(KeyCode::Digit2), ElementState::Pressed)
            .expect("quality key should send renderer command");

        let command = inbox
            .recv_command()
            .await
            .expect("quality command should be received");

        assert!(matches!(
            command.payload,
            RendererCommand::SetQualitySettings { settings }
                if settings == RenderQualitySettings::interactive()
        ));
    }

    // Verifies that window shutdown does not use tokio's blocking send inside a runtime.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_retries_when_command_channel_is_full_inside_runtime() {
        let config = WindowConfig::default();
        let (endpoint, mut inbox) = renderer_transport(1);
        endpoint
            .try_send(MessageEnvelope::new(RendererCommand::ResizeSurface {
                surface_id: config.surface_id,
                generation: config.initial_generation,
                extent: config.extent,
            }))
            .expect("test command channel should accept the prefilled command");

        let receiver = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime should build");

            runtime.block_on(async move {
                let _ = inbox
                    .recv_command()
                    .await
                    .expect("prefilled resize command should be received");
                let shutdown = inbox
                    .recv_command()
                    .await
                    .expect("shutdown command should be received after retry");
                assert!(matches!(shutdown.payload, RendererCommand::Shutdown));
            });
        });

        let mut app = WindowedApp::new(endpoint, config);
        app.request_shutdown()
            .expect("shutdown request should retry until command capacity is available");
        receiver.join().expect("receiver thread should not panic");
    }

    // Verifies that vertical arrow keys are reserved for light control like the old demo.
    #[test]
    fn vertical_arrows_are_not_camera_movement_keys() {
        assert_eq!(
            camera_key(PhysicalKey::Code(KeyCode::ArrowUp)),
            CameraKey::Other
        );
        assert_eq!(
            camera_key(PhysicalKey::Code(KeyCode::ArrowDown)),
            CameraKey::Other
        );
    }

    // Verifies that the app-side light state changes continuously while a light key is held.
    #[test]
    fn held_light_keys_change_window_light_intensity() {
        let config = WindowConfig::default();
        let (endpoint, _inbox) = renderer_transport(1);
        let mut app = WindowedApp::new(endpoint, config);
        let initial = app.light_intensity;

        app.light_brighter = true;
        app.update_light_for_frame(0.5);
        assert!(app.light_intensity > initial);

        app.light_brighter = false;
        app.light_darker = true;
        app.update_light_for_frame(1.0);
        assert!(app.light_intensity < initial);
    }
}
