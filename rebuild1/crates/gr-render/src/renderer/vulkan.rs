mod buffer;
mod debug;
mod frame;
mod immediate;
mod material;
mod mesh;
mod post;
mod readback;
mod swapchain;

use std::{collections::BTreeMap, ffi::CStr};

use ash::{Device, Entry, Instance, khr, vk};
use thiserror::Error;

use crate::{
    import::{ImportedScene, import_asset_on_worker},
    protocol::{
        AssetHandle, DropReason, FrameId, FramebufferReadback, FramebufferReadbackOptions,
        LoadedAsset, MessageEnvelope, NativeSurfaceHandle, NativeSurfacePlatform, NonZeroExtent,
        RenderItemPacket, RenderQualitySettings, RendererCommand, RendererEvent, RendererInbox,
        SurfaceDescriptor, SurfaceGeneration, SurfaceId, TransportError, WindowId,
    },
};

use self::{
    debug::{VulkanDebug, VulkanDebugConfig},
    frame::{FramePresentStatus, VulkanFrames},
    material::VulkanMaterialStore,
    mesh::VulkanMeshStore,
    readback::{FramebufferReadbackConfig, FramebufferReadbackState},
    swapchain::{ShadowResources, ShadowSamplerFallback, VulkanSwapchain},
};
use super::{RendererBackend, RendererResult, assets::GpuAssetStore};

const APP_NAME: &CStr = c"rebuild1";
const ENGINE_NAME: &CStr = c"rebuild1";

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("failed to load Vulkan entry: {0}")]
    EntryLoad(#[from] ash::LoadingError),
    #[error("Vulkan call failed: {0:?}")]
    Vk(#[from] vk::Result),
    #[error("native surface platform is not supported: {0:?}")]
    UnsupportedSurface(NativeSurfacePlatform),
    #[error("no Vulkan device supports graphics and present for the configured surface")]
    NoSuitableDevice,
    #[error("the selected Vulkan queue family cannot present to the configured surface")]
    SelectedQueueCannotPresent,
    #[error("logical device is required before creating a swapchain")]
    LogicalDeviceMissing,
    #[error("surface has no supported image formats")]
    SurfaceFormatsUnavailable,
    #[error("surface has no supported present modes")]
    PresentModesUnavailable,
    #[error("surface extent is zero or unavailable")]
    SurfaceExtentUnavailable,
    #[error("surface does not support color attachment swapchain images")]
    ColorAttachmentUnsupported,
    #[error("surface has no supported composite alpha mode")]
    CompositeAlphaUnavailable,
    #[error("swapchain image index {index} is out of range for {count} swapchain images")]
    SwapchainImageIndexOutOfRange { index: usize, count: usize },
    #[error("frame slot index {index} is out of range for {count} frame slots")]
    FrameSlotIndexOutOfRange { index: usize, count: usize },
    #[error("no compatible host-visible Vulkan memory type was found")]
    MemoryTypeUnavailable,
    #[error("failed to read compiled shader code: {0}")]
    ShaderCodeRead(std::io::Error),
    #[error("shader interface contract failed: {0}")]
    ShaderInterface(String),
    #[error("failed to compile render graph: {0}")]
    GraphCompile(String),
}

#[derive(Default)]
pub struct VulkanRendererBackend;

impl RendererBackend for VulkanRendererBackend {
    /// Runs the Vulkan backend and owns Vulkan objects on the renderer thread.
    async fn run(self, mut inbox: RendererInbox) -> RendererResult {
        tracing::info!("vulkan renderer backend starting");
        let mut context = VulkanContext::new()?;

        inbox
            .send_event(MessageEnvelope::new(RendererEvent::RendererReady))
            .await?;
        tracing::info!("vulkan renderer backend ready");

        while let Some(command) = inbox.recv_command().await {
            match command.payload {
                RendererCommand::ConfigureSurface { surface } => {
                    let configured = context.configure_surface(surface)?;
                    let event = RendererEvent::SurfaceConfigured {
                        surface_id: configured.surface_id,
                        generation: configured.generation,
                        extent: configured.extent,
                        platform: configured.platform,
                    };
                    inbox.send_event(MessageEnvelope::new(event)).await?;
                }
                RendererCommand::ResizeSurface {
                    surface_id,
                    generation,
                    extent,
                } => {
                    if let Some(configured) =
                        context.resize_surface(surface_id, generation, extent)?
                    {
                        let event = RendererEvent::SurfaceResized {
                            surface_id: configured.surface_id,
                            generation: configured.generation,
                            extent: configured.extent,
                        };
                        inbox.send_event(MessageEnvelope::new(event)).await?;
                    } else {
                        send_warning(
                            &inbox,
                            format!("resize ignored for unknown surface {}", surface_id.raw()),
                        )
                        .await?;
                    }
                }
                RendererCommand::SubmitFrame { snapshot } => {
                    let frame_id = snapshot.frame_id;
                    match context.present_frame(snapshot)? {
                        FramePresentResult::Presented { readback } => {
                            if let Some(readback) = readback {
                                let readback_frame_id = readback.frame_id;
                                let event =
                                    MessageEnvelope::new(RendererEvent::FramebufferReadback {
                                        readback,
                                    })
                                    .with_frame_id(readback_frame_id);
                                inbox.send_event(event).await?;
                            }
                            let event =
                                MessageEnvelope::new(RendererEvent::FramePresented { frame_id })
                                    .with_frame_id(frame_id);
                            inbox.send_event(event).await?;
                        }
                        FramePresentResult::Dropped(reason) => {
                            send_frame_dropped(&inbox, frame_id, reason).await?;
                        }
                    }
                }
                RendererCommand::LoadAsset { path } => {
                    tracing::trace!(path = %path.display(), "Vulkan renderer loading asset");
                    let event = match import_asset_on_worker(path).await {
                        Ok(imported) => {
                            let asset = context.register_imported_asset(&imported)?;
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
                    tracing::trace!(asset = ?asset, "Vulkan renderer unloading asset");
                    if !context.unload_asset(asset) {
                        let event = RendererEvent::ValidationWarning {
                            message: format!("asset unload ignored for stale handle: {asset:?}"),
                        };
                        inbox.send_event(MessageEnvelope::new(event)).await?;
                    }
                }
                RendererCommand::SetFramebufferReadback { options } => {
                    context.set_framebuffer_readback(options)?;
                }
                RendererCommand::SetQualitySettings { settings } => {
                    context.set_quality_settings(settings);
                }
                RendererCommand::Shutdown => {
                    tracing::info!("vulkan renderer backend stopping");
                    inbox
                        .send_event(MessageEnvelope::new(RendererEvent::RendererStopped))
                        .await?;
                    break;
                }
                other => {
                    tracing::trace!(command = other.name(), "vulkan renderer ignored command");
                }
            }
        }

        tracing::info!("vulkan renderer backend stopped");
        Ok(())
    }
}

struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    debug: Option<VulkanDebug>,
    surface_loader: khr::surface::Instance,
    win32_surface_loader: khr::win32_surface::Instance,
    device: Option<VulkanDevice>,
    surfaces: BTreeMap<SurfaceId, VulkanSurface>,
    assets: GpuAssetStore,
    framebuffer_readback: FramebufferReadbackOptions,
    quality: RenderQualitySettings,
}

impl VulkanContext {
    /// Creates the Vulkan entry, instance, extension loaders, and empty surface registry.
    fn new() -> Result<Self, VulkanError> {
        tracing::trace!("loading Vulkan entry");
        let entry = load_entry()?;
        let debug_config = VulkanDebugConfig::new(&entry)?;
        let instance = create_instance(&entry, &debug_config)?;
        let debug = VulkanDebug::create(&entry, &instance, debug_config.debug_utils_enabled())?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let win32_surface_loader = khr::win32_surface::Instance::new(&entry, &instance);

        tracing::info!(
            validation_layer = debug_config.validation_layer_enabled(),
            debug_utils = debug_config.debug_utils_enabled(),
            "created Vulkan instance"
        );
        Ok(Self {
            _entry: entry,
            instance,
            debug,
            surface_loader,
            win32_surface_loader,
            device: None,
            surfaces: BTreeMap::new(),
            assets: GpuAssetStore::default(),
            framebuffer_readback: FramebufferReadbackOptions::default(),
            quality: RenderQualitySettings::default(),
        })
    }

    /// Creates or replaces the Vulkan surface for one protocol window.
    fn configure_surface(
        &mut self,
        descriptor: SurfaceDescriptor,
    ) -> Result<ConfiguredSurface, VulkanError> {
        tracing::trace!(
            window_id = descriptor.window_id.raw(),
            surface_id = descriptor.surface_id.raw(),
            generation = descriptor.generation.raw(),
            width = descriptor.extent.width(),
            height = descriptor.extent.height(),
            platform = descriptor.native.platform().name(),
            "configuring Vulkan surface"
        );

        self.wait_device_idle()?;
        self.destroy_surface_for_id(descriptor.surface_id);
        let mut surface = self.create_surface(descriptor)?;
        if let Err(error) = self.ensure_device_for_surface(surface.handle) {
            self.destroy_surface(surface);
            return Err(error);
        }

        let swapchain =
            match self.create_swapchain(surface.handle, surface.extent, vk::SwapchainKHR::null()) {
                Ok(swapchain) => swapchain,
                Err(error) => {
                    self.destroy_surface(surface);
                    return Err(error);
                }
            };

        surface.extent = swapchain.extent;
        surface.swapchain = Some(swapchain);
        let configured = surface.info();
        self.surfaces.insert(surface.surface_id, surface);

        tracing::info!(
            surface_id = configured.surface_id.raw(),
            generation = configured.generation.raw(),
            width = configured.extent.width(),
            height = configured.extent.height(),
            platform = configured.platform.name(),
            "created Vulkan surface and swapchain"
        );
        Ok(configured)
    }

    /// Recreates the swapchain for a configured Vulkan surface and stores the new extent.
    fn resize_surface(
        &mut self,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
    ) -> Result<Option<ConfiguredSurface>, VulkanError> {
        let Some(existing) = self.surfaces.get(&surface_id) else {
            tracing::trace!(
                surface_id = surface_id.raw(),
                "Vulkan surface resize missed"
            );
            return Ok(None);
        };

        if existing.extent == extent && existing.swapchain.is_some() {
            tracing::trace!(
                surface_id = surface_id.raw(),
                generation = generation.raw(),
                width = extent.width(),
                height = extent.height(),
                "Vulkan surface resize skipped because extent is unchanged"
            );
            let mut configured = existing.info();
            if existing.generation != generation {
                let Some(surface) = self.surfaces.get_mut(&surface_id) else {
                    return Ok(None);
                };
                surface.generation = generation;
                configured.generation = generation;
            }
            return Ok(Some(configured));
        }

        self.wait_device_idle()?;
        let Some(mut surface) = self.surfaces.remove(&surface_id) else {
            tracing::trace!(
                surface_id = surface_id.raw(),
                "Vulkan surface resize skipped because the surface disappeared"
            );
            return Ok(None);
        };
        let old_swapchain = surface.swapchain.take();
        let old_swapchain_handle = old_swapchain
            .as_ref()
            .map_or(vk::SwapchainKHR::null(), |swapchain| swapchain.handle);

        let new_swapchain =
            match self.create_swapchain(surface.handle, extent, old_swapchain_handle) {
                Ok(swapchain) => swapchain,
                Err(error) => {
                    surface.swapchain = old_swapchain;
                    self.surfaces.insert(surface_id, surface);
                    return Err(error);
                }
            };

        if let Some(old_swapchain) = old_swapchain {
            self.destroy_swapchain(old_swapchain);
        }

        surface.extent = new_swapchain.extent;
        surface.generation = generation;
        surface.swapchain = Some(new_swapchain);
        let configured = surface.info();
        tracing::trace!(
            surface_id = surface_id.raw(),
            generation = generation.raw(),
            width = configured.extent.width(),
            height = configured.extent.height(),
            "recreated Vulkan swapchain for resized surface"
        );
        self.surfaces.insert(surface_id, surface);
        Ok(Some(configured))
    }

    /// Renders and presents a submitted frame snapshot against its target surface.
    fn present_frame(
        &mut self,
        snapshot: crate::protocol::FrameSnapshot,
    ) -> Result<FramePresentResult, VulkanError> {
        tracing::trace!(
            frame_id = snapshot.frame_id.raw(),
            surface_id = snapshot.surface_id.raw(),
            generation = snapshot.surface_generation.raw(),
            views = snapshot.views.len(),
            render_items = snapshot.render_items.len(),
            lights = snapshot.lights.len(),
            "presenting Vulkan frame snapshot"
        );

        let Some(surface) = self.surfaces.get(&snapshot.surface_id) else {
            tracing::trace!(
                frame_id = snapshot.frame_id.raw(),
                surface_id = snapshot.surface_id.raw(),
                "Vulkan frame dropped because its surface is not configured"
            );
            return Ok(FramePresentResult::Dropped(DropReason::NoSurface {
                surface_id: snapshot.surface_id,
            }));
        };

        if surface.generation != snapshot.surface_generation {
            tracing::trace!(
                frame_id = snapshot.frame_id.raw(),
                surface_id = snapshot.surface_id.raw(),
                submitted = snapshot.surface_generation.raw(),
                current = surface.generation.raw(),
                "Vulkan frame dropped because surface generation is stale"
            );
            return Ok(FramePresentResult::Dropped(
                DropReason::StaleSurfaceGeneration {
                    surface_id: snapshot.surface_id,
                    submitted: snapshot.surface_generation,
                    current: surface.generation,
                },
            ));
        }

        self.trace_render_item_asset_summary(&snapshot.render_items);
        let surface_id = snapshot.surface_id;
        let Some(mut surface) = self.surfaces.remove(&surface_id) else {
            tracing::trace!(
                frame_id = snapshot.frame_id.raw(),
                surface_id = surface_id.raw(),
                "Vulkan frame dropped because the selected surface disappeared"
            );
            return Ok(FramePresentResult::Dropped(DropReason::NoSurface {
                surface_id,
            }));
        };
        let result = if let Some(swapchain) = surface.swapchain.as_mut() {
            let Some(device) = self.device.as_mut() else {
                self.surfaces.insert(surface_id, surface);
                return Err(VulkanError::LogicalDeviceMissing);
            };
            match device.present_frame(swapchain, &snapshot) {
                Ok(FramePresentStatus::Presented { readback }) => FramePresentResult::Presented {
                    readback: readback.map(|sample| {
                        FramebufferReadback::new(
                            sample.frame_id,
                            snapshot.surface_id,
                            snapshot.surface_generation,
                            surface.extent,
                            sample.metering,
                        )
                    }),
                },
                Ok(FramePresentStatus::SwapchainOutOfDate) => {
                    FramePresentResult::Dropped(DropReason::SwapchainOutOfDate { surface_id })
                }
                Err(error) => {
                    self.surfaces.insert(surface_id, surface);
                    return Err(error);
                }
            }
        } else {
            tracing::trace!(
                surface_id = surface_id.raw(),
                frame_id = snapshot.frame_id.raw(),
                "Vulkan frame skipped because the surface has no swapchain"
            );
            FramePresentResult::Dropped(DropReason::NoSurface { surface_id })
        };

        let retired = self.assets.collect_deferred_destroys();
        if let Some(device) = self.device.as_mut() {
            device.destroy_retired_assets(&retired);
        }
        self.surfaces.insert(surface_id, surface);
        Ok(result)
    }

    /// Traces one aggregate asset-readiness summary for the submitted draw packets.
    fn trace_render_item_asset_summary(&self, items: &[RenderItemPacket]) {
        if !tracing::enabled!(tracing::Level::TRACE) {
            return;
        }

        let drawable = items
            .iter()
            .filter(|item| self.assets.can_draw(item.mesh, item.material))
            .count();
        tracing::trace!(
            render_items = items.len(),
            drawable,
            missing = items.len().saturating_sub(drawable),
            "checked render item asset readiness"
        );
    }

    /// Registers worker-imported scene metadata and returns protocol asset handles.
    fn register_imported_asset(
        &mut self,
        imported: &ImportedScene,
    ) -> Result<LoadedAsset, VulkanError> {
        let asset = self.assets.upload_imported_scene(imported);
        let texture_records = self.assets.texture_descriptors(&asset.textures);
        let material_records = self.assets.material_descriptors(&asset.materials);
        if let Some(device) = self.device.as_mut() {
            device.upload_imported_meshes(&self.instance, &asset.meshes, imported.meshes())?;
            device.upload_imported_textures(&texture_records)?;
            device.upload_imported_materials(&material_records)?;
        }
        Ok(asset)
    }

    /// Invalidates one renderer asset handle and queues its GPU resources for deferred destroy.
    fn unload_asset(&mut self, asset: AssetHandle) -> bool {
        let unloaded = self.assets.unload(asset);
        tracing::trace!(
            asset = ?asset,
            pending_destroy_count = self.assets.pending_destroy_count(),
            "queued Vulkan asset unload"
        );
        unloaded
    }

    /// Stores app-visible framebuffer readback policy and applies it to the active device.
    fn set_framebuffer_readback(
        &mut self,
        options: FramebufferReadbackOptions,
    ) -> Result<(), VulkanError> {
        self.framebuffer_readback = options;
        let Some(device) = self.device.as_mut() else {
            tracing::trace!("stored framebuffer readback options before Vulkan device creation");
            return Ok(());
        };

        device.set_framebuffer_readback_options(options);
        if let Some(swapchain) = self
            .surfaces
            .values()
            .find_map(|surface| surface.swapchain.as_ref())
        {
            device.configure_framebuffer_readback(swapchain)?;
        }
        Ok(())
    }

    /// Stores renderer quality policy and applies it to all later submitted frames.
    fn set_quality_settings(&mut self, settings: RenderQualitySettings) {
        tracing::info!(
            ssao_intensity = settings.ssao().intensity(),
            aa_threshold = settings.anti_aliasing().edge_threshold(),
            post_contrast = settings.post().contrast(),
            "updated Vulkan renderer quality settings"
        );
        self.quality = settings;
        if let Some(device) = self.device.as_mut() {
            device.set_quality_settings(settings);
        }
    }

    /// Creates the logical device once and verifies that later surfaces are present-capable.
    fn ensure_device_for_surface(&mut self, surface: vk::SurfaceKHR) -> Result<(), VulkanError> {
        if let Some(device) = &self.device {
            return device.ensure_surface_support(&self.surface_loader, surface);
        }

        let mut device = create_device_for_surface(&self.instance, &self.surface_loader, surface)?;
        device.set_framebuffer_readback_options(self.framebuffer_readback);
        device.set_quality_settings(self.quality);
        tracing::info!(
            queue_family_index = device.queue_family_index,
            device_name = %device.name,
            "created Vulkan logical device"
        );
        self.device = Some(device);
        Ok(())
    }

    /// Creates a swapchain for one surface using the selected logical device.
    fn create_swapchain(
        &mut self,
        surface: vk::SurfaceKHR,
        extent: NonZeroExtent,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<VulkanSwapchain, VulkanError> {
        let physical_device = self
            .device
            .as_ref()
            .ok_or(VulkanError::LogicalDeviceMissing)?
            .physical_device;
        let support =
            swapchain::query_surface_support(&self.surface_loader, physical_device, surface)?;
        let config = swapchain::choose_swapchain_config(&support, extent)?;
        let device = self
            .device
            .as_mut()
            .ok_or(VulkanError::LogicalDeviceMissing)?;
        let swapchain = device.create_swapchain(surface, config, old_swapchain)?;
        device.configure_framebuffer_readback(&swapchain)?;

        Ok(swapchain)
    }

    /// Creates a platform Vulkan surface from one renderer protocol descriptor.
    fn create_surface(&self, descriptor: SurfaceDescriptor) -> Result<VulkanSurface, VulkanError> {
        match descriptor.native {
            NativeSurfaceHandle::Win32(handle) => {
                let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(handle.hwnd().get() as vk::HWND)
                    .hinstance(handle.hinstance().map_or(0, |value| value.get()) as vk::HINSTANCE);

                // Safety: the app thread owns the winit window for the whole configured surface
                // lifetime, and `SurfaceDescriptor` contains non-zero Win32 handles captured from
                // raw-window-handle while that window was alive.
                let surface = unsafe {
                    self.win32_surface_loader
                        .create_win32_surface(&create_info, None)
                }?;

                Ok(VulkanSurface {
                    window_id: descriptor.window_id,
                    surface_id: descriptor.surface_id,
                    generation: descriptor.generation,
                    extent: descriptor.extent,
                    platform: NativeSurfacePlatform::Win32,
                    handle: surface,
                    swapchain: None,
                })
            }
        }
    }

    /// Destroys the Vulkan surface owned by one protocol surface id if it exists.
    fn destroy_surface_for_id(&mut self, surface_id: SurfaceId) {
        if let Some(surface) = self.surfaces.remove(&surface_id) {
            self.destroy_surface(surface);
        }
    }

    /// Destroys one Vulkan surface handle with the matching KHR surface loader.
    fn destroy_surface(&self, surface: VulkanSurface) {
        if let Some(swapchain) = surface.swapchain {
            self.destroy_swapchain(swapchain);
        }

        tracing::trace!(
            window_id = surface.window_id.raw(),
            surface_id = surface.surface_id.raw(),
            generation = surface.generation.raw(),
            platform = surface.platform.name(),
            "destroying Vulkan surface"
        );

        // Safety: every stored surface was created by this instance and is removed exactly once
        // before the Vulkan instance is destroyed.
        unsafe {
            self.surface_loader.destroy_surface(surface.handle, None);
        }
    }

    /// Destroys one swapchain through the logical device that created it.
    fn destroy_swapchain(&self, swapchain: VulkanSwapchain) {
        if let Some(device) = &self.device {
            device.destroy_swapchain(swapchain);
        }
    }

    /// Waits for the logical device when it exists.
    fn wait_device_idle(&self) -> Result<(), VulkanError> {
        if let Some(device) = &self.device {
            device.wait_idle()?;
        }

        Ok(())
    }
}

impl Drop for VulkanContext {
    /// Destroys Vulkan surfaces first, then destroys the Vulkan instance.
    fn drop(&mut self) {
        tracing::trace!("destroying Vulkan context");

        if let Err(error) = self.wait_device_idle() {
            tracing::warn!(error = %error, "failed to wait for Vulkan device idle during drop");
        }

        let surfaces = std::mem::take(&mut self.surfaces);
        for surface in surfaces.into_values() {
            self.destroy_surface(surface);
        }

        if let Some(device) = self.device.take() {
            device.destroy();
        }

        if let Some(debug) = self.debug.take() {
            debug.destroy();
        }

        // Safety: all child Vulkan surfaces and the logical device owned by this context were
        // destroyed above, and the debug messenger has been destroyed before its parent instance.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

struct VulkanSurface {
    window_id: WindowId,
    surface_id: SurfaceId,
    generation: SurfaceGeneration,
    extent: NonZeroExtent,
    platform: NativeSurfacePlatform,
    handle: vk::SurfaceKHR,
    swapchain: Option<VulkanSwapchain>,
}

impl VulkanSurface {
    /// Returns the protocol-facing surface facts that can be copied into events.
    fn info(&self) -> ConfiguredSurface {
        ConfiguredSurface {
            surface_id: self.surface_id,
            generation: self.generation,
            extent: self.extent,
            platform: self.platform,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ConfiguredSurface {
    surface_id: SurfaceId,
    generation: SurfaceGeneration,
    extent: NonZeroExtent,
    platform: NativeSurfacePlatform,
}

struct VulkanDevice {
    name: String,
    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Device,
    queue_family_index: u32,
    graphics_queue: vk::Queue,
    swapchain_loader: khr::swapchain::Device,
    frames: VulkanFrames,
    materials: VulkanMaterialStore,
    meshes: VulkanMeshStore,
    shadow_fallback: ShadowSamplerFallback,
    shadows: Option<ShadowResources>,
    shadow_cache: ShadowCacheState,
    readback: FramebufferReadbackState,
    quality: RenderQualitySettings,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ShadowFrameSignature {
    camera_eye_bucket: [i32; 3],
    camera_forward_bucket: [i32; 3],
    fov_bucket: i32,
    caster_hash: u64,
    translucent_casters: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ShadowFrameData {
    pub(super) view_proj: [[f32; 16]; crate::renderer::graph::SHADOW_CASCADE_COUNT],
    pub(super) splits: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct ShadowCacheState {
    dirty: bool,
    signature: Option<ShadowFrameSignature>,
    frame_data: Option<ShadowFrameData>,
}

impl ShadowCacheState {
    /// Creates a cache state that forces the first real shadow frame to populate the maps.
    fn dirty() -> Self {
        Self {
            dirty: true,
            signature: None,
            frame_data: None,
        }
    }

    /// Marks cached shadow maps unusable after assets or shadow resources change.
    fn invalidate(&mut self) {
        self.dirty = true;
        self.signature = None;
        self.frame_data = None;
    }

    /// Returns whether the submitted frame needs to render shadow maps again.
    fn needs_refresh(self, signature: Option<ShadowFrameSignature>) -> bool {
        signature.is_some_and(|signature| {
            self.dirty || self.signature != Some(signature) || self.frame_data.is_none()
        })
    }

    /// Returns the shadow matrices that match the currently cached shadow-map contents.
    fn frame_data(self) -> Option<ShadowFrameData> {
        self.frame_data
    }

    /// Stores the signature that was rendered into the persistent shadow resources.
    fn mark_refreshed(
        &mut self,
        signature: Option<ShadowFrameSignature>,
        frame_data: Option<ShadowFrameData>,
    ) {
        if let Some(signature) = signature {
            self.signature = Some(signature);
            self.frame_data = frame_data;
            self.dirty = false;
        }
    }
}

impl VulkanDevice {
    /// Uploads imported mesh geometry into backend-local vertex and index buffers.
    fn upload_imported_meshes(
        &mut self,
        instance: &Instance,
        handles: &[crate::protocol::MeshHandle],
        meshes: &[crate::import::ImportedMesh],
    ) -> Result<(), VulkanError> {
        self.meshes.upload_imported_meshes(
            instance,
            &self.device,
            self.physical_device,
            handles,
            meshes,
        )?;
        self.shadow_cache.invalidate();
        Ok(())
    }

    /// Uploads imported texture payloads into backend-local sampled images.
    fn upload_imported_textures(
        &mut self,
        textures: &[(
            crate::protocol::TextureHandle,
            crate::protocol::TextureDescriptor,
        )],
    ) -> Result<(), VulkanError> {
        self.materials.upload_imported_textures(
            &self.device,
            &self.memory_properties,
            self.queue_family_index,
            self.graphics_queue,
            textures,
        )?;
        self.shadow_cache.invalidate();
        Ok(())
    }

    /// Uploads material parameter buffers and texture descriptors.
    fn upload_imported_materials(
        &mut self,
        materials: &[(
            crate::protocol::MaterialHandle,
            crate::protocol::MaterialDescriptor,
        )],
    ) -> Result<(), VulkanError> {
        self.materials.upload_imported_materials(
            &self.device,
            &self.memory_properties,
            materials,
        )?;
        self.shadow_cache.invalidate();
        Ok(())
    }

    /// Destroys backend-local resources whose protocol handles have passed deferred retirement.
    fn destroy_retired_assets(&mut self, retired: &[AssetHandle]) {
        self.meshes.destroy_retired(&self.device, retired);
        self.materials.destroy_retired(&self.device, retired);
        if !retired.is_empty() {
            self.shadow_cache.invalidate();
        }
    }

    /// Updates the framebuffer readback cadence requested by the app protocol.
    fn set_framebuffer_readback_options(&mut self, options: FramebufferReadbackOptions) {
        self.readback.set_options(options);
    }

    /// Updates shader quality constants applied while recording later frames.
    fn set_quality_settings(&mut self, settings: RenderQualitySettings) {
        self.quality = settings;
    }

    /// Rebuilds readback buffers for the currently configured swapchain.
    fn configure_framebuffer_readback(
        &mut self,
        swapchain: &VulkanSwapchain,
    ) -> Result<(), VulkanError> {
        self.readback.configure(
            &self.device,
            &self.memory_properties,
            FramebufferReadbackConfig {
                image_count: swapchain.image_count(),
                extent: swapchain.extent,
                format: swapchain.format,
                transfer_src_supported: swapchain.transfer_src_supported(),
            },
        )
    }

    /// Verifies that the chosen queue family can present to a newly configured surface.
    fn ensure_surface_support(
        &self,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(), VulkanError> {
        let supported = get_surface_support(
            surface_loader,
            self.physical_device,
            self.queue_family_index,
            surface,
        )?;

        if supported {
            Ok(())
        } else {
            Err(VulkanError::SelectedQueueCannotPresent)
        }
    }

    /// Destroys the logical device after all device-owned resources have been released.
    fn destroy(mut self) {
        tracing::trace!(
            queue_family_index = self.queue_family_index,
            device_name = %self.name,
            "destroying Vulkan logical device"
        );

        if let Err(error) = self.wait_idle() {
            tracing::warn!(error = %error, "failed to wait for Vulkan device idle before destroy");
        }

        if let Some(shadows) = self.shadows.take() {
            self.destroy_shadow_resources(shadows);
        }
        self.shadow_fallback.destroy(&self.device);
        self.frames.destroy(&self.device);
        self.readback.destroy(&self.device);
        self.meshes.destroy(&self.device);
        self.materials.destroy(&self.device);

        // Safety: all swapchains and frame resources owned by this context were destroyed before
        // the device is destroyed, and no custom allocation callbacks are used.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[derive(Clone, Debug)]
struct DeviceCandidate {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    score: u32,
    name: String,
}

#[derive(Clone, Debug)]
enum FramePresentResult {
    Presented {
        readback: Option<FramebufferReadback>,
    },
    Dropped(DropReason),
}

/// Sends one validation warning event from the Vulkan backend.
async fn send_warning(inbox: &RendererInbox, message: String) -> Result<(), TransportError> {
    tracing::trace!(message, "sending Vulkan validation warning");
    inbox
        .send_event(MessageEnvelope::new(RendererEvent::ValidationWarning {
            message,
        }))
        .await
}

/// Sends one dropped-frame event and tags the envelope with the submitted frame id.
async fn send_frame_dropped(
    inbox: &RendererInbox,
    frame_id: FrameId,
    reason: DropReason,
) -> Result<(), TransportError> {
    tracing::trace!(
        frame_id = frame_id.raw(),
        reason = reason.name(),
        "sending Vulkan dropped-frame event"
    );
    inbox
        .send_event(
            MessageEnvelope::new(RendererEvent::FrameDropped { frame_id, reason })
                .with_frame_id(frame_id),
        )
        .await
}

/// Loads Vulkan function pointers from the platform Vulkan loader.
fn load_entry() -> Result<Entry, VulkanError> {
    // Safety: loading the Vulkan loader is the required first step before creating an instance;
    // the returned `Entry` owns the loaded function table for all later calls in this context.
    unsafe { Entry::load() }.map_err(VulkanError::EntryLoad)
}

/// Creates the Vulkan instance with surface extensions and optional debug utilities.
fn create_instance(entry: &Entry, debug: &VulkanDebugConfig) -> Result<Instance, VulkanError> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(APP_NAME)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(ENGINE_NAME)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let mut extensions = vec![
        khr::surface::NAME.as_ptr(),
        khr::win32_surface::NAME.as_ptr(),
    ];
    debug.append_extensions(&mut extensions);

    let layer_names = debug.layer_names();
    let mut debug_create_info = debug::messenger_create_info();
    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extensions)
        .enabled_layer_names(&layer_names);

    if debug.debug_utils_enabled() {
        create_info = create_info.push_next(&mut debug_create_info);
    }

    tracing::trace!("creating Vulkan instance");
    // Safety: `create_info` points to static extension names and local application info that live
    // for the duration of the call; no custom allocation callbacks are used.
    unsafe { entry.create_instance(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Selects a Vulkan device that supports graphics commands and presentation to one surface.
fn create_device_for_surface(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<VulkanDevice, VulkanError> {
    let candidate = select_device_candidate(instance, surface_loader, surface)?;
    let priorities = [1.0_f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(candidate.queue_family_index)
        .queue_priorities(&priorities)];
    let extensions = [khr::swapchain::NAME.as_ptr()];
    let features = vk::PhysicalDeviceFeatures::default().independent_blend(true);
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    tracing::trace!(
        queue_family_index = candidate.queue_family_index,
        device_name = %candidate.name,
        "creating Vulkan logical device"
    );
    // Safety: `candidate` was selected from this instance, queue family zero is requested with one
    // priority value, and the swapchain extension name is a static Vulkan C string.
    let device = unsafe { instance.create_device(candidate.physical_device, &create_info, None) }?;
    // Safety: the device was created with one queue from `candidate.queue_family_index`.
    let graphics_queue = unsafe { device.get_device_queue(candidate.queue_family_index, 0) };
    let swapchain_loader = khr::swapchain::Device::new(instance, &device);
    let memory_properties = buffer::memory_properties(instance, candidate.physical_device);
    let frames = match VulkanFrames::create(&device, candidate.queue_family_index) {
        Ok(frames) => frames,
        Err(error) => {
            // Safety: no device resources escaped because frame creation failed before returning
            // the logical device to the context.
            unsafe {
                device.destroy_device(None);
            }
            return Err(error);
        }
    };
    let materials = match VulkanMaterialStore::create(&device) {
        Ok(materials) => materials,
        Err(error) => {
            frames.destroy(&device);
            // Safety: no device resources escape because material resource creation failed before
            // returning the logical device to the context.
            unsafe {
                device.destroy_device(None);
            }
            return Err(error);
        }
    };
    let meshes = match VulkanMeshStore::create(
        instance,
        &device,
        candidate.physical_device,
        frames.slot_count(),
        materials.material_set_layout(),
    ) {
        Ok(meshes) => meshes,
        Err(error) => {
            materials.destroy(&device);
            frames.destroy(&device);
            // Safety: no device resources escape because mesh resource creation failed before
            // returning the logical device to the context.
            unsafe {
                device.destroy_device(None);
            }
            return Err(error);
        }
    };
    let shadow_fallback = match ShadowSamplerFallback::create(
        &device,
        &memory_properties,
        candidate.queue_family_index,
        graphics_queue,
        &meshes,
    ) {
        Ok(fallback) => fallback,
        Err(error) => {
            meshes.destroy(&device);
            materials.destroy(&device);
            frames.destroy(&device);
            // Safety: no device resources escape because fallback creation failed before
            // returning the logical device to the context.
            unsafe {
                device.destroy_device(None);
            }
            return Err(error);
        }
    };
    Ok(VulkanDevice {
        name: candidate.name,
        physical_device: candidate.physical_device,
        memory_properties,
        device,
        queue_family_index: candidate.queue_family_index,
        graphics_queue,
        swapchain_loader,
        frames,
        materials,
        meshes,
        shadow_fallback,
        shadows: None,
        shadow_cache: ShadowCacheState::dirty(),
        readback: FramebufferReadbackState::default(),
        quality: RenderQualitySettings::default(),
    })
}

/// Picks the highest-scored physical device that can draw and present to the surface.
fn select_device_candidate(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<DeviceCandidate, VulkanError> {
    let physical_devices = enumerate_physical_devices(instance)?;
    tracing::trace!(
        physical_device_count = physical_devices.len(),
        "enumerated Vulkan physical devices"
    );

    let mut best = None;
    for physical_device in physical_devices {
        let properties = get_physical_device_properties(instance, physical_device);
        let name = physical_device_name(&properties);
        let score = physical_device_score(&properties);
        let features = get_physical_device_features(instance, physical_device);
        let has_swapchain = device_supports_swapchain(instance, physical_device)?;
        let has_independent_blend = features.independent_blend == vk::TRUE;
        let queue_family_index =
            find_graphics_present_queue(instance, surface_loader, physical_device, surface)?;

        tracing::trace!(
            device_name = %name,
            score,
            has_swapchain,
            has_independent_blend,
            queue_family_index,
            "evaluated Vulkan physical device"
        );

        let Some(queue_family_index) = queue_family_index else {
            continue;
        };

        if !has_swapchain || !has_independent_blend {
            continue;
        }

        let candidate = DeviceCandidate {
            physical_device,
            queue_family_index,
            score,
            name,
        };

        if best
            .as_ref()
            .is_none_or(|best: &DeviceCandidate| candidate.score > best.score)
        {
            best = Some(candidate);
        }
    }

    best.ok_or(VulkanError::NoSuitableDevice)
}

/// Finds one queue family that supports graphics commands and presentation to the surface.
fn find_graphics_present_queue(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<Option<u32>, VulkanError> {
    let queue_families = get_queue_family_properties(instance, physical_device);

    for (index, queue_family) in queue_families.iter().enumerate() {
        let queue_family_index = index as u32;
        let supports_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let supports_present =
            get_surface_support(surface_loader, physical_device, queue_family_index, surface)?;

        tracing::trace!(
            queue_family_index,
            supports_graphics,
            supports_present,
            queue_count = queue_family.queue_count,
            "evaluated Vulkan queue family"
        );

        if supports_graphics && supports_present {
            return Ok(Some(queue_family_index));
        }
    }

    Ok(None)
}

/// Returns whether a physical device exposes the swapchain extension required for presentation.
fn device_supports_swapchain(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<bool, VulkanError> {
    let extensions = enumerate_device_extensions(instance, physical_device)?;
    Ok(extensions
        .iter()
        .any(|extension| extension_name_matches(extension, khr::swapchain::NAME)))
}

/// Gives simple preference to discrete GPUs while keeping selection deterministic and readable.
fn physical_device_score(properties: &vk::PhysicalDeviceProperties) -> u32 {
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1_000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 250,
        vk::PhysicalDeviceType::CPU => 100,
        _ => 0,
    }
}

/// Converts Vulkan's fixed C string device name into an owned Rust string for logs.
fn physical_device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    // Safety: Vulkan guarantees that `device_name` is a null-terminated C string.
    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Compares a Vulkan extension property name with a required static extension name.
fn extension_name_matches(property: &vk::ExtensionProperties, expected: &CStr) -> bool {
    // Safety: Vulkan guarantees that `extension_name` is a null-terminated C string.
    (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == expected
}

/// Enumerates Vulkan physical devices for this instance.
fn enumerate_physical_devices(instance: &Instance) -> Result<Vec<vk::PhysicalDevice>, VulkanError> {
    // Safety: `instance` is alive for the whole query and no output pointers escape the call.
    unsafe { instance.enumerate_physical_devices() }.map_err(VulkanError::Vk)
}

/// Reads immutable properties for a Vulkan physical device.
fn get_physical_device_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceProperties {
    // Safety: `physical_device` came from `instance.enumerate_physical_devices`.
    unsafe { instance.get_physical_device_properties(physical_device) }
}

/// Reads core feature support needed by the current renderer pipelines.
fn get_physical_device_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceFeatures {
    // Safety: `physical_device` came from `instance.enumerate_physical_devices`.
    unsafe { instance.get_physical_device_features(physical_device) }
}

/// Reads queue family properties for a Vulkan physical device.
fn get_queue_family_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<vk::QueueFamilyProperties> {
    // Safety: `physical_device` came from `instance.enumerate_physical_devices`.
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
}

/// Queries whether one queue family can present to one Vulkan surface.
fn get_surface_support(
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    surface: vk::SurfaceKHR,
) -> Result<bool, VulkanError> {
    // Safety: `surface` was created by this instance, and the queue index is taken from that
    // physical device's queue family list before this function is called.
    unsafe {
        surface_loader.get_physical_device_surface_support(
            physical_device,
            queue_family_index,
            surface,
        )
    }
    .map_err(VulkanError::Vk)
}

/// Enumerates device extension properties for one Vulkan physical device.
fn enumerate_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<vk::ExtensionProperties>, VulkanError> {
    // Safety: `physical_device` came from `instance.enumerate_physical_devices`.
    unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(VulkanError::Vk)
}
