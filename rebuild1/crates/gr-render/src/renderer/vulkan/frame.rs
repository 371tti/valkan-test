use std::cmp::Ordering;

use ash::{Device, khr, vk};

use crate::{
    math::{dot3, sub3},
    protocol::{CameraSnapshot, FrameSnapshot, RenderItemPacket, RenderQualitySettings},
    renderer::graph::{
        BarrierLocation, FrameGraphPlan, GraphPass, PassOutput, ResourceBarrier, ResourceState,
        shadow_blur_h_pass_index, shadow_blur_v_pass_index, shadow_pass_index,
        translucent_shadow_pass_index,
    },
};

use super::{
    VulkanDevice, VulkanError,
    material::VulkanMaterialStore,
    mesh::{
        EmissiveLightUniforms, MAX_EMISSIVE_LIGHTS, MAX_LOCAL_SHADOW_CASTERS, MeshDrawOptions,
        MeshPassResources, MeshPipelineSet, ShadowCascadeCull, VulkanMeshStore,
    },
    readback::{FramebufferReadbackCopy, FramebufferReadbackSample, record_image_to_buffer},
    shadow::{
        mesh_frame_uniform_for_frame, shadow_cascade_cull, shadow_frame_data,
        shadow_frame_signature,
    },
    swapchain::{ShadowResources, VulkanSwapchain},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.015, 0.018, 0.026, 1.0];

pub(super) struct VulkanFrames {
    command_pool: vk::CommandPool,
    slots: Vec<VulkanFrameSlot>,
    image_render_finished: Vec<vk::Semaphore>,
    cursor: usize,
}

#[derive(Clone, Copy)]
struct VulkanFrameSlot {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    submitted: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ActiveFrame {
    slot_index: usize,
    image_index: u32,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
}

pub(super) enum FrameAcquire {
    Ready(ActiveFrame),
    SwapchainOutOfDate,
}

pub(super) enum FramePresentStatus {
    Presented {
        readback: Option<FramebufferReadbackSample>,
    },
    SwapchainOutOfDate,
}

impl VulkanFrames {
    /// Creates reusable command buffers and sync objects for bounded frames in flight.
    pub(super) fn create(device: &Device, queue_family_index: u32) -> Result<Self, VulkanError> {
        let command_pool = create_command_pool(device, queue_family_index)?;
        let command_buffers =
            match allocate_command_buffers(device, command_pool, MAX_FRAMES_IN_FLIGHT as u32) {
                Ok(command_buffers) => command_buffers,
                Err(error) => {
                    destroy_command_pool(device, command_pool);
                    return Err(error);
                }
            };
        let slots = match create_frame_slots(device, command_buffers) {
            Ok(slots) => slots,
            Err(error) => {
                destroy_command_pool(device, command_pool);
                return Err(error);
            }
        };

        tracing::info!(
            frames_in_flight = slots.len(),
            queue_family_index,
            "created Vulkan frame resources"
        );

        Ok(Self {
            command_pool,
            slots,
            image_render_finished: Vec::new(),
            cursor: 0,
        })
    }

    /// Waits for all frame fences before swapchain-sized resources are destroyed.
    pub(super) fn wait_for_idle(&self, device: &Device) -> Result<(), VulkanError> {
        let fences: Vec<_> = self
            .slots
            .iter()
            .filter(|slot| slot.submitted)
            .map(|slot| slot.in_flight)
            .collect();
        wait_for_fences(device, &fences)
    }

    /// Returns the number of reusable frame slots owned by this frame system.
    pub(super) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Acquires the next swapchain image and returns the current reusable frame slot.
    fn acquire(
        &mut self,
        device: &Device,
        swapchain_loader: &khr::swapchain::Device,
        swapchain: &VulkanSwapchain,
    ) -> Result<FrameAcquire, VulkanError> {
        self.ensure_image_render_finished(device, swapchain.image_count())?;

        let slot_index = self.cursor;
        let slot = self.slots[slot_index];
        if slot.submitted {
            wait_for_fences(device, &[slot.in_flight])?;
            self.slots[slot_index].submitted = false;
        }

        let image =
            match acquire_swapchain_image(swapchain_loader, swapchain, slot.image_available)? {
                AcquireImage::Ready {
                    image_index,
                    suboptimal,
                } => {
                    if suboptimal {
                        tracing::trace!(image_index, "acquired suboptimal Vulkan swapchain image");
                    }
                    image_index
                }
                AcquireImage::SwapchainOutOfDate => return Ok(FrameAcquire::SwapchainOutOfDate),
            };

        reset_command_buffer(device, slot.command_buffer)?;
        let render_finished = self.render_finished_for_image(image)?;

        Ok(FrameAcquire::Ready(ActiveFrame {
            slot_index,
            image_index: image,
            command_buffer: slot.command_buffer,
            image_available: slot.image_available,
            render_finished,
            in_flight: slot.in_flight,
        }))
    }

    /// Advances the reusable frame slot cursor after a submit/present attempt.
    fn advance(&mut self) {
        self.cursor = (self.cursor + 1) % self.slots.len();
    }

    /// Marks the frame slot as queued so future reuse waits for its fence.
    fn mark_submitted(&mut self, frame: ActiveFrame) {
        self.slots[frame.slot_index].submitted = true;
    }

    /// Rebuilds per-swapchain-image render completion semaphores when image count changes.
    fn ensure_image_render_finished(
        &mut self,
        device: &Device,
        image_count: usize,
    ) -> Result<(), VulkanError> {
        if self.image_render_finished.len() == image_count {
            return Ok(());
        }

        tracing::trace!(
            previous_count = self.image_render_finished.len(),
            image_count,
            "rebuilding per-image Vulkan present semaphores"
        );

        destroy_semaphores(device, std::mem::take(&mut self.image_render_finished));
        self.image_render_finished = create_semaphores(device, image_count)?;
        Ok(())
    }

    /// Returns the render completion semaphore tied to one acquired swapchain image.
    fn render_finished_for_image(&self, image_index: u32) -> Result<vk::Semaphore, VulkanError> {
        let index = image_index as usize;
        self.image_render_finished.get(index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index,
                count: self.image_render_finished.len(),
            },
        )
    }

    /// Destroys sync objects and the command pool owned by the frame system.
    pub(super) fn destroy(self, device: &Device) {
        tracing::trace!(
            frames_in_flight = self.slots.len(),
            image_semaphore_count = self.image_render_finished.len(),
            "destroying Vulkan frame resources"
        );

        destroy_frame_slots(device, self.slots);
        destroy_semaphores(device, self.image_render_finished);
        destroy_command_pool(device, self.command_pool);
    }
}

impl VulkanDevice {
    /// Records, submits, and presents the current swapchain frame.
    pub(super) fn present_frame(
        &mut self,
        swapchain: &mut VulkanSwapchain,
        snapshot: &FrameSnapshot,
    ) -> Result<FramePresentStatus, VulkanError> {
        let features = frame_feature_flags(&self.materials, &snapshot.render_items);
        let camera = active_camera(snapshot);
        let scene_metadata_required = scene_material_metadata_required(self.quality);
        if features.has_shadow_casters {
            self.ensure_shadow_resources()?;
        }
        let shadow_signature = features.has_shadow_casters.then(|| {
            shadow_frame_signature(
                camera,
                &snapshot.render_items,
                features.has_translucent_shadow_casters,
            )
        });
        let refresh_shadows = self.shadow_cache.needs_refresh(shadow_signature);
        let cached_shadow_data = self.shadow_cache.frame_data();
        let current_shadow_data =
            if refresh_shadows || (features.has_shadow_casters && cached_shadow_data.is_none()) {
                features
                    .has_shadow_casters
                    .then(|| shadow_frame_data(camera, swapchain.extent_2d()))
            } else {
                None
            };
        let shadow_data = if refresh_shadows {
            current_shadow_data
        } else {
            cached_shadow_data.or(current_shadow_data)
        };
        let frame = match self
            .frames
            .acquire(&self.device, &self.swapchain_loader, swapchain)?
        {
            FrameAcquire::Ready(frame) => frame,
            FrameAcquire::SwapchainOutOfDate => return Ok(FramePresentStatus::SwapchainOutOfDate),
        };

        tracing::trace!(
            frame_slot = frame.slot_index,
            image_index = frame.image_index,
            "building swapchain render graph"
        );
        let readback = self
            .readback
            .prepare_frame(&self.device, frame.image_index)?;
        let shadows = self.shadows.as_ref();
        let initial_states = swapchain.graph_initial_states(frame.image_index, shadows)?;
        let graph = if refresh_shadows {
            FrameGraphPlan::standard_frame_with_readback_and_scene_metadata(
                DEFAULT_CLEAR_COLOR,
                initial_states,
                readback.copy.is_some(),
                features.has_shadow_casters,
                features.has_translucent_shadow_casters,
                scene_metadata_required,
            )
        } else {
            FrameGraphPlan::standard_frame_with_shadow_refresh_and_scene_metadata(
                DEFAULT_CLEAR_COLOR,
                initial_states,
                readback.copy.is_some(),
                features.has_shadow_casters,
                features.has_translucent_shadow_casters,
                false,
                scene_metadata_required,
            )
        }
        .map_err(|error| VulkanError::GraphCompile(error.to_string()))?;
        trace_compiled_graph("standard_frame_executor", &graph);

        self.meshes.write_frame_uniform(
            &self.device,
            frame.slot_index,
            mesh_frame_uniform_for_frame(
                camera,
                frame_light_intensity(snapshot),
                self.quality,
                swapchain.extent_2d(),
                features.has_shadow_casters,
                features.has_translucent_shadow_casters,
                shadow_data,
                emissive_light_uniforms(&self.materials, &self.meshes, snapshot, camera),
            ),
        )?;
        let scene_pass_resources = shadows.map_or_else(
            || self.shadow_fallback.mesh_pass_resources(),
            ShadowResources::mesh_pass_resources,
        );
        record_graph_command_buffer(
            &self.device,
            frame,
            swapchain,
            shadows,
            scene_pass_resources,
            &graph,
            &self.materials,
            &self.meshes,
            snapshot,
            readback.copy,
            features,
            self.quality,
        )?;
        submit_frame(&self.device, self.graphics_queue, frame)?;
        swapchain.apply_graph_final_states(frame.image_index, &graph)?;
        if let Some(shadows) = self.shadows.as_mut() {
            shadows.apply_graph_final_states(&graph);
        }
        if refresh_shadows {
            self.shadow_cache
                .mark_refreshed(shadow_signature, current_shadow_data);
        }
        if readback.copy.is_some() {
            self.readback
                .mark_copy_recorded(frame.image_index, snapshot.frame_id);
        }
        self.frames.mark_submitted(frame);
        let status = match present_frame(
            &self.swapchain_loader,
            self.graphics_queue,
            swapchain,
            frame,
        )? {
            FramePresentStatus::Presented { .. } => FramePresentStatus::Presented {
                readback: readback.latest,
            },
            FramePresentStatus::SwapchainOutOfDate => FramePresentStatus::SwapchainOutOfDate,
        };
        self.frames.advance();

        Ok(status)
    }

    /// Waits until all device work is idle before resource teardown or swapchain recreation.
    pub(super) fn wait_idle(&self) -> Result<(), VulkanError> {
        tracing::trace!("waiting for Vulkan device idle");
        self.frames.wait_for_idle(&self.device)?;

        // Safety: the logical device is alive and owned by the renderer thread while all child
        // resources are still valid.
        unsafe { self.device.device_wait_idle() }.map_err(VulkanError::Vk)
    }

    /// Creates fixed shadow resources the first time a real frame needs them.
    fn ensure_shadow_resources(&mut self) -> Result<(), VulkanError> {
        if self.shadows.is_some() {
            return Ok(());
        }

        self.shadows = Some(self.create_shadow_resources()?);
        self.shadow_cache.invalidate();
        Ok(())
    }
}

enum AcquireImage {
    Ready { image_index: u32, suboptimal: bool },
    SwapchainOutOfDate,
}

/// Creates the command pool used by per-frame command buffers.
fn create_command_pool(
    device: &Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, VulkanError> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    // Safety: the queue family index was selected from this device's physical queue families.
    unsafe { device.create_command_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates primary command buffers reused by the frame slots.
fn allocate_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>, VulkanError> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);

    // Safety: the command pool belongs to this device and remains alive until after buffers are
    // implicitly freed with the pool.
    unsafe { device.allocate_command_buffers(&allocate_info) }.map_err(VulkanError::Vk)
}

/// Creates one frame slot for every allocated command buffer.
fn create_frame_slots(
    device: &Device,
    command_buffers: Vec<vk::CommandBuffer>,
) -> Result<Vec<VulkanFrameSlot>, VulkanError> {
    let mut slots = Vec::with_capacity(command_buffers.len());

    for command_buffer in command_buffers {
        match create_frame_slot(device, command_buffer) {
            Ok(slot) => slots.push(slot),
            Err(error) => {
                destroy_frame_slots(device, slots);
                return Err(error);
            }
        }
    }

    Ok(slots)
}

/// Creates the semaphores and fence paired with one command buffer.
fn create_frame_slot(
    device: &Device,
    command_buffer: vk::CommandBuffer,
) -> Result<VulkanFrameSlot, VulkanError> {
    let image_available = create_semaphore(device)?;
    let in_flight = match create_signaled_fence(device) {
        Ok(fence) => fence,
        Err(error) => {
            destroy_semaphore(device, image_available);
            return Err(error);
        }
    };

    Ok(VulkanFrameSlot {
        command_buffer,
        image_available,
        in_flight,
        submitted: false,
    })
}

/// Creates one binary semaphore for every swapchain image that can be presented.
fn create_semaphores(device: &Device, count: usize) -> Result<Vec<vk::Semaphore>, VulkanError> {
    let mut semaphores = Vec::with_capacity(count);

    for _ in 0..count {
        match create_semaphore(device) {
            Ok(semaphore) => semaphores.push(semaphore),
            Err(error) => {
                destroy_semaphores(device, semaphores);
                return Err(error);
            }
        }
    }

    Ok(semaphores)
}

/// Creates one binary semaphore for acquire/submit ordering.
fn create_semaphore(device: &Device) -> Result<vk::Semaphore, VulkanError> {
    let create_info = vk::SemaphoreCreateInfo::default();

    // Safety: no pointers are stored beyond the call and no custom allocation callbacks are used.
    unsafe { device.create_semaphore(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates a fence that starts signaled so the first frame can begin immediately.
fn create_signaled_fence(device: &Device) -> Result<vk::Fence, VulkanError> {
    let create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    // Safety: no pointers are stored beyond the call and no custom allocation callbacks are used.
    unsafe { device.create_fence(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Acquires one image from the swapchain using the current frame's acquire semaphore.
fn acquire_swapchain_image(
    swapchain_loader: &khr::swapchain::Device,
    swapchain: &VulkanSwapchain,
    image_available: vk::Semaphore,
) -> Result<AcquireImage, VulkanError> {
    // Safety: the swapchain is alive, the semaphore is unsignaled for the current frame slot, and
    // no fence is used for image acquisition.
    match unsafe {
        swapchain_loader.acquire_next_image(
            swapchain.handle,
            u64::MAX,
            image_available,
            vk::Fence::null(),
        )
    } {
        Ok((image_index, suboptimal)) => Ok(AcquireImage::Ready {
            image_index,
            suboptimal,
        }),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(AcquireImage::SwapchainOutOfDate),
        Err(error) => Err(VulkanError::Vk(error)),
    }
}

/// Records the current graph plan into one frame command buffer.
fn record_graph_command_buffer(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    shadows: Option<&ShadowResources>,
    scene_pass_resources: &MeshPassResources,
    graph: &FrameGraphPlan,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    readback: Option<FramebufferReadbackCopy>,
    features: FrameFeatureFlags,
    quality: RenderQualitySettings,
) -> Result<(), VulkanError> {
    tracing::trace!(
        passes = graph.pass_count(),
        barriers = graph.barrier_count(),
        "recording render graph"
    );

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // Safety: the command buffer belongs to the current frame slot and is reset before recording.
    unsafe {
        device.begin_command_buffer(frame.command_buffer, &begin_info)?;
        let mut state = FrameRecordState::new(features, quality);
        for pass in graph.passes() {
            record_barriers_for_location(
                device,
                frame.command_buffer,
                swapchain,
                shadows,
                frame.image_index,
                graph,
                BarrierLocation::BeforePass(pass.name()),
            )?;
            record_graph_pass(
                device,
                frame,
                swapchain,
                shadows,
                scene_pass_resources,
                pass,
                materials,
                meshes,
                snapshot,
                readback,
                &mut state,
            )?;
        }
        record_barriers_for_location(
            device,
            frame.command_buffer,
            swapchain,
            shadows,
            frame.image_index,
            graph,
            BarrierLocation::AfterGraph,
        )?;
        device.end_command_buffer(frame.command_buffer)?;
    }

    Ok(())
}

#[derive(Clone, Copy, Default)]
struct FrameFeatureFlags {
    has_shadow_casters: bool,
    has_translucent_shadow_casters: bool,
    has_transparent_scene_items: bool,
}

struct FrameRecordState {
    features: FrameFeatureFlags,
    quality: RenderQualitySettings,
}

impl FrameRecordState {
    /// Carries frame-wide feature flags plus stats discovered while recording graph passes.
    fn new(features: FrameFeatureFlags, quality: RenderQualitySettings) -> Self {
        Self { features, quality }
    }
}

/// Scans one extracted frame for feature flags that let shaders skip unused work.
fn frame_feature_flags(
    materials: &VulkanMaterialStore,
    items: &[RenderItemPacket],
) -> FrameFeatureFlags {
    let mut flags = FrameFeatureFlags::default();
    for item in items {
        if !item.flags.visible {
            continue;
        }
        let casts_shadow = item.flags.casts_shadow;
        let is_transparent = materials.is_transparent(item.material);
        let casts_translucent_shadow =
            casts_shadow && materials.casts_translucent_shadow(item.material);
        flags.has_transparent_scene_items |= is_transparent;
        flags.has_translucent_shadow_casters |= casts_translucent_shadow;
        flags.has_shadow_casters |= casts_translucent_shadow
            || (casts_shadow && materials.casts_opaque_shadow(item.material));
        if flags.has_shadow_casters
            && flags.has_translucent_shadow_casters
            && flags.has_transparent_scene_items
        {
            break;
        }
    }
    flags
}

/// Records the backend body for one compiled graph pass.
fn record_graph_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    shadows: Option<&ShadowResources>,
    scene_pass_resources: &MeshPassResources,
    pass: &GraphPass,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    readback: Option<FramebufferReadbackCopy>,
    state: &mut FrameRecordState,
) -> Result<(), VulkanError> {
    let name = pass.name();
    if let Some(cascade_index) = shadow_pass_index(name) {
        let shadows = required_shadow_resources(shadows, name)?;
        return record_shadow_pass(
            device,
            frame,
            shadows,
            cascade_index,
            materials,
            meshes,
            &snapshot.render_items,
            active_camera(snapshot),
        );
    }
    if let Some(cascade_index) = shadow_blur_h_pass_index(name) {
        let shadows = required_shadow_resources(shadows, name)?;
        return record_shadow_moment_blur_pass(device, frame, shadows, cascade_index, true);
    }
    if let Some(cascade_index) = shadow_blur_v_pass_index(name) {
        let shadows = required_shadow_resources(shadows, name)?;
        return record_shadow_moment_blur_pass(device, frame, shadows, cascade_index, false);
    }
    if let Some(cascade_index) = translucent_shadow_pass_index(name) {
        let shadows = required_shadow_resources(shadows, name)?;
        if !state.features.has_translucent_shadow_casters {
            tracing::trace!(
                cascade_index,
                "translucent shadow pass skipped because no translucent casters are live"
            );
            return Ok(());
        }
        return record_translucent_shadow_pass(
            device,
            frame,
            shadows,
            cascade_index,
            materials,
            meshes,
            &snapshot.render_items,
            active_camera(snapshot),
        );
    }

    match name {
        "scene" => record_scene_pass(
            device,
            frame,
            swapchain,
            scene_pass_resources,
            pass,
            materials,
            meshes,
            snapshot,
            state,
        ),
        "post" => record_post_pass(device, frame, swapchain, snapshot, state),
        "framebuffer_readback" => record_framebuffer_readback_pass(
            device,
            frame.command_buffer,
            swapchain,
            frame.image_index,
            readback,
        ),
        "present" => {
            tracing::trace!("graph present pass has no command body");
            Ok(())
        }
        other => Err(VulkanError::GraphCompile(format!(
            "graph pass {other} has no Vulkan executor"
        ))),
    }
}

/// Records graph barriers that belong at one command-buffer location.
fn record_barriers_for_location(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    swapchain: &VulkanSwapchain,
    shadows: Option<&ShadowResources>,
    image_index: u32,
    graph: &FrameGraphPlan,
    location: BarrierLocation,
) -> Result<(), VulkanError> {
    for &barrier in graph.barriers_at(location) {
        let (image, aspect) = graph_image(swapchain, shadows, barrier.resource(), image_index)?;
        record_graph_barrier(device, command_buffer, image, aspect, barrier);
    }

    Ok(())
}

/// Resolves a graph resource to the Vulkan image owned by either fixed shadows or the swapchain.
fn graph_image(
    swapchain: &VulkanSwapchain,
    shadows: Option<&ShadowResources>,
    resource: crate::renderer::graph::GraphResource,
    image_index: u32,
) -> Result<(vk::Image, vk::ImageAspectFlags), VulkanError> {
    if let Some(image) = shadows.and_then(|shadows| shadows.graph_image(resource)) {
        return Ok(image);
    }

    swapchain.graph_image(resource, image_index)
}

/// Returns real shadow resources for graph passes that cannot execute against the fallback set.
fn required_shadow_resources<'a>(
    shadows: Option<&'a ShadowResources>,
    pass_name: &str,
) -> Result<&'a ShadowResources, VulkanError> {
    shadows.ok_or_else(|| {
        VulkanError::GraphCompile(format!(
            "graph pass {pass_name} requested real shadow resources after they were omitted"
        ))
    })
}

/// Returns whether post effects will sample scene normal/roughness metadata this frame.
fn scene_material_metadata_required(quality: RenderQualitySettings) -> bool {
    quality.ssao().intensity() > 0.0
        || quality.ssr().intensity() > 0.0
        || quality.anti_aliasing().blend() > 0.0
}

/// Records one graph-owned image barrier outside pass recording.
fn record_graph_barrier(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    barrier: ResourceBarrier,
) {
    tracing::trace!(
        resource = barrier.resource().name(),
        from = barrier.from().name(),
        to = barrier.to().name(),
        "recording render graph barrier"
    );

    let image_barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(access_for_source_state(barrier.from()))
        .dst_access_mask(access_for_destination_state(barrier.to()))
        .old_layout(layout_for_state(barrier.from()))
        .new_layout(layout_for_state(barrier.to()))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range(aspect));
    let image_barriers = [image_barrier];

    // Safety: `command_buffer` is currently recording, `image` is owned by the swapchain resource
    // set, and the graph plan constrains this barrier to the declared resource state.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            stage_for_source_state(barrier.from()),
            stage_for_destination_state(barrier.to()),
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );
    }
}

/// Records scene draws into the graph-owned scene color and depth targets.
fn record_scene_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    pass_resources: &MeshPassResources,
    pass: &GraphPass,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let metadata_required = scene_material_metadata_required(state.quality);
    let clear_values = scene_clear_values(pass, metadata_required)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain.extent_2d());
    let render_pass = if metadata_required {
        swapchain.scene_render_pass()
    } else {
        swapchain.scene_fast_render_pass()
    };
    let framebuffer = if metadata_required {
        swapchain.scene_framebuffer()
    } else {
        swapchain.scene_fast_framebuffer()
    };
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place scene color and scene depth in attachment layouts before this
    // render pass begins, and the framebuffer owns those exact image views.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let camera = active_camera(snapshot);
        let mesh_options =
            MeshDrawOptions::scene(swapchain.extent_2d(), camera, snapshot.optimization);
        let opaque_pipeline = if metadata_required {
            swapchain.mesh_pipeline()
        } else {
            swapchain.mesh_fast_pipeline()
        };
        let transparent_pipeline = if metadata_required {
            swapchain.transparent_mesh_pipeline()
        } else {
            swapchain.transparent_mesh_fast_pipeline()
        };
        let mut opaque_count = 0_usize;
        let mut transparent_count = 0_usize;
        for item in &snapshot.render_items {
            if materials.is_transparent(item.material) {
                continue;
            }
            if meshes.bind_and_draw(
                device,
                frame.command_buffer,
                opaque_pipeline,
                materials,
                Some(pass_resources),
                frame.slot_index,
                item,
                mesh_options,
            )? {
                opaque_count += 1;
            }
        }
        for item in &snapshot.render_items {
            if !materials.is_transparent(item.material) {
                continue;
            }
            if meshes.bind_and_draw(
                device,
                frame.command_buffer,
                transparent_pipeline,
                materials,
                Some(pass_resources),
                frame.slot_index,
                item,
                mesh_options,
            )? {
                transparent_count += 1;
            }
        }
        tracing::trace!(
            opaque_count,
            transparent_count,
            metadata_required,
            "recorded scene mesh draw groups"
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records moment shadow data for items that explicitly cast shadows.
fn record_shadow_pass(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    cascade_index: usize,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    items: &[RenderItemPacket],
    camera: CameraSnapshot,
) -> Result<(), VulkanError> {
    let clear_values = [color_clear_value([1.0, 1.0, 1.0, 1.0]), depth_clear_value()];
    let shadow_extent = shadows.extent_2d(cascade_index)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.shadow_render_pass())
        .framebuffer(shadows.shadow_framebuffer(cascade_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place the moment map in color attachment layout before this pass.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let caster_count = record_mesh_draws(
            device,
            frame,
            materials,
            meshes,
            shadows.shadow_pipeline(),
            None,
            items,
            shadow_extent,
            cascade_index,
            shadow_cascade_cull(camera, cascade_index),
            MeshDrawFilter::OpaqueShadowCasters,
        )?;
        if caster_count == 0 {
            tracing::trace!(
                cascade_index,
                "opaque shadow pass cleared without mesh draws because no casters are live"
            );
        }
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records one separable blur pass over shadow moments for a single cascade.
fn record_shadow_moment_blur_pass(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    cascade_index: usize,
    horizontal: bool,
) -> Result<(), VulkanError> {
    let clear_values = [color_clear_value([1.0, 1.0, 1.0, 1.0])];
    let shadow_extent = shadows.extent_2d(cascade_index)?;
    let framebuffer = if horizontal {
        shadows.blur_h_framebuffer(cascade_index)?
    } else {
        shadows.blur_v_framebuffer(cascade_index)?
    };
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.blur_render_pass())
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place the blur source in shader-read layout and the blur target in
    // color-attachment layout before this fullscreen pass begins.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        if horizontal {
            shadows.blur_pipeline().draw_horizontal(
                device,
                frame.command_buffer,
                cascade_index,
                shadow_extent,
            );
        } else {
            shadows.blur_pipeline().draw_vertical(
                device,
                frame.command_buffer,
                cascade_index,
                shadow_extent,
            );
        }
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records multiplicative transmittance for transparent shadow casters.
fn record_translucent_shadow_pass(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    cascade_index: usize,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    items: &[RenderItemPacket],
    camera: CameraSnapshot,
) -> Result<(), VulkanError> {
    let clear_values = [color_clear_value([1.0, 1.0, 1.0, 1.0])];
    let shadow_extent = shadows.extent_2d(cascade_index)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.translucent_render_pass())
        .framebuffer(shadows.translucent_framebuffer(cascade_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place transmittance in color attachment layout and opaque cascade
    // depth in shader-read layout. The fragment shader samples opaque depth explicitly.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let caster_count = record_mesh_draws(
            device,
            frame,
            materials,
            meshes,
            shadows.translucent_pipeline(),
            Some(shadows.translucent_pass_resources()),
            items,
            shadow_extent,
            cascade_index,
            shadow_cascade_cull(camera, cascade_index),
            MeshDrawFilter::TranslucentShadowCasters,
        )?;
        if caster_count == 0 {
            tracing::trace!(
                cascade_index,
                "translucent shadow pass cleared to full transmittance because no casters are live"
            );
        }
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records the post pass that samples scene color and writes the swapchain image.
fn record_post_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let clear_values = [color_clear_value([0.0, 0.0, 0.0, 1.0])];
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain.extent_2d());
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.post_render_pass())
        .framebuffer(swapchain.post_framebuffer_for_image(frame.image_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place scene color in shader-read layout and the swapchain image in
    // color-attachment layout before this render pass begins.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.post_pipeline().draw(
            device,
            frame.command_buffer,
            swapchain.extent_2d(),
            snapshot.camera_effects,
            active_camera(snapshot),
            state.quality,
            state.features.has_transparent_scene_items,
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records the app-requested final framebuffer copy after post and before presentation.
fn record_framebuffer_readback_pass(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    swapchain: &VulkanSwapchain,
    image_index: u32,
    readback: Option<FramebufferReadbackCopy>,
) -> Result<(), VulkanError> {
    let Some(copy) = readback else {
        tracing::trace!("framebuffer readback pass skipped because no copy target is prepared");
        return Ok(());
    };
    let image = swapchain.image_for_index(image_index)?;

    tracing::trace!(
        image_index,
        width = copy.extent.width,
        height = copy.extent.height,
        "recording framebuffer readback copy"
    );
    record_image_to_buffer(device, command_buffer, image, copy);
    Ok(())
}

#[derive(Clone, Copy)]
enum MeshDrawFilter {
    OpaqueShadowCasters,
    TranslucentShadowCasters,
}

/// Records mesh-only passes without duplicating per-pass draw loops.
fn record_mesh_draws(
    device: &Device,
    frame: ActiveFrame,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    pipeline: MeshPipelineSet,
    pass_resources: Option<&MeshPassResources>,
    items: &[RenderItemPacket],
    extent: vk::Extent2D,
    cascade_index: usize,
    shadow_cull: ShadowCascadeCull,
    filter: MeshDrawFilter,
) -> Result<usize, VulkanError> {
    let mut drawn = 0;
    for item in items {
        if !shadow_filter_accepts(filter, materials, item) {
            continue;
        }
        if meshes.bind_and_draw(
            device,
            frame.command_buffer,
            pipeline,
            materials,
            pass_resources,
            frame.slot_index,
            item,
            MeshDrawOptions::shadow(extent, cascade_index, shadow_cull),
        )? {
            drawn += 1;
        }
    }

    Ok(drawn)
}

/// Returns whether one render item belongs in the selected shadow caster pass.
fn shadow_filter_accepts(
    filter: MeshDrawFilter,
    materials: &VulkanMaterialStore,
    item: &crate::protocol::RenderItemPacket,
) -> bool {
    item.flags.visible
        && item.flags.casts_shadow
        && match filter {
            MeshDrawFilter::OpaqueShadowCasters => materials.casts_opaque_shadow(item.material),
            MeshDrawFilter::TranslucentShadowCasters => {
                materials.casts_translucent_shadow(item.material)
            }
        }
}

/// Returns the first extracted camera, or the protocol default when no view is available.
fn active_camera(snapshot: &FrameSnapshot) -> CameraSnapshot {
    snapshot
        .views
        .first()
        .map(|view| view.camera)
        .unwrap_or_default()
}

/// Returns the renderer light intensity encoded in the first extracted light packet.
fn frame_light_intensity(snapshot: &FrameSnapshot) -> f32 {
    snapshot
        .lights
        .first()
        .map(|light| light.intensity)
        .unwrap_or(1.0)
        .max(0.0)
}

#[derive(Clone, Copy)]
struct EmissiveLightCandidate {
    score: f32,
    position_radius: [f32; 4],
    color: [f32; 4],
}

#[derive(Clone, Copy)]
struct LocalShadowCasterCandidate {
    score: f32,
    center_radius: [f32; 4],
}

/// Extracts a small, stable local-light set from emissive mesh materials.
fn emissive_light_uniforms(
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
) -> EmissiveLightUniforms {
    const MIN_EMISSIVE_BRIGHTNESS: f32 = 0.02;
    const EMISSIVE_LIGHT_COLOR_SCALE: f32 = 1.35;
    const EMISSIVE_LIGHT_RADIUS_SCALE: f32 = 4.5;
    const EMISSIVE_LIGHT_MIN_RADIUS: f32 = 1.5;
    const EMISSIVE_LIGHT_MAX_RADIUS: f32 = 72.0;

    let mut candidates = Vec::new();

    for item in &snapshot.render_items {
        if !item.flags.visible {
            continue;
        }

        let Some(emissive) = materials.emissive_factor(item.material) else {
            continue;
        };
        let brightness = max3(emissive);
        if brightness <= MIN_EMISSIVE_BRIGHTNESS {
            continue;
        }

        let Some(bounds) = meshes.bounds_for(item.mesh) else {
            continue;
        };
        let center = bounds.center();
        let source_radius = bounds.radius();
        let light_radius = (source_radius * (EMISSIVE_LIGHT_RADIUS_SCALE + brightness.sqrt()))
            .clamp(EMISSIVE_LIGHT_MIN_RADIUS, EMISSIVE_LIGHT_MAX_RADIUS);
        let camera_delta = sub3(center, camera.eye);
        let camera_distance_sq = dot3(camera_delta, camera_delta).max(1.0);
        let score = brightness * light_radius * light_radius / (1.0 + camera_distance_sq * 0.0025);

        candidates.push(EmissiveLightCandidate {
            score,
            position_radius: [center[0], center[1], center[2], light_radius],
            color: [
                emissive[0] * EMISSIVE_LIGHT_COLOR_SCALE,
                emissive[1] * EMISSIVE_LIGHT_COLOR_SCALE,
                emissive[2] * EMISSIVE_LIGHT_COLOR_SCALE,
                brightness,
            ],
        });
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let mut uniforms = EmissiveLightUniforms::disabled();
    for (index, candidate) in candidates.iter().take(MAX_EMISSIVE_LIGHTS).enumerate() {
        uniforms.position_radius[index] = candidate.position_radius;
        uniforms.color[index] = candidate.color;
    }
    uniforms.count[0] = candidates.len().min(MAX_EMISSIVE_LIGHTS) as f32;
    let selected_light_position_radius = uniforms.position_radius;
    let selected_light_count = uniforms.count[0] as usize;
    fill_local_shadow_casters(
        materials,
        meshes,
        snapshot,
        camera,
        &selected_light_position_radius,
        selected_light_count,
        &mut uniforms,
    );

    uniforms
}

fn fill_local_shadow_casters(
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
    light_position_radius: &[[f32; 4]; MAX_EMISSIVE_LIGHTS],
    light_count: usize,
    uniforms: &mut EmissiveLightUniforms,
) {
    if light_count == 0 {
        return;
    }

    const MIN_OCCLUDER_RADIUS: f32 = 0.08;
    const EMISSIVE_OCCLUDER_SKIP_BRIGHTNESS: f32 = 0.04;
    const LOCAL_SHADOW_INFLUENCE_SCALE: f32 = 1.15;

    let mut candidates = Vec::new();

    for item in &snapshot.render_items {
        if !item.flags.visible
            || !item.flags.casts_shadow
            || !materials.casts_opaque_shadow(item.material)
        {
            continue;
        }

        if materials
            .emissive_factor(item.material)
            .is_some_and(|emissive| max3(emissive) > EMISSIVE_OCCLUDER_SKIP_BRIGHTNESS)
        {
            continue;
        }

        let Some(bounds) = meshes.bounds_for(item.mesh) else {
            continue;
        };
        let center = bounds.center();
        let radius = bounds.radius();
        if radius < MIN_OCCLUDER_RADIUS {
            continue;
        }

        let mut light_score = 0.0_f32;
        for light in light_position_radius.iter().take(light_count) {
            let light_radius = light[3].max(0.001);
            let light_to_caster = sub3(center, [light[0], light[1], light[2]]);
            let distance_sq = dot3(light_to_caster, light_to_caster);
            let influence_radius = (light_radius + radius) * LOCAL_SHADOW_INFLUENCE_SCALE;

            if distance_sq > influence_radius * influence_radius {
                continue;
            }

            light_score = light_score.max((radius * radius) / (distance_sq + radius * radius));
        }

        if light_score <= 0.0 {
            continue;
        }

        let camera_delta = sub3(center, camera.eye);
        let camera_distance_sq = dot3(camera_delta, camera_delta).max(1.0);
        let camera_score = (radius * radius) / (camera_distance_sq * 0.015 + radius * radius);

        candidates.push(LocalShadowCasterCandidate {
            score: light_score * 0.8 + camera_score * 0.2,
            center_radius: [center[0], center[1], center[2], radius],
        });
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    for (index, candidate) in candidates.iter().take(MAX_LOCAL_SHADOW_CASTERS).enumerate() {
        uniforms.shadow_caster_center_radius[index] = candidate.center_radius;
    }
    uniforms.shadow_caster_count[0] = candidates.len().min(MAX_LOCAL_SHADOW_CASTERS) as f32;
}

fn max3(value: [f32; 3]) -> f32 {
    value[0].max(value[1]).max(value[2])
}

/// Traces a compiled graph plan without exposing Vulkan objects through the graph API.
fn trace_compiled_graph(label: &'static str, graph: &FrameGraphPlan) {
    if !tracing::enabled!(tracing::Level::TRACE) {
        return;
    }

    let resources = graph
        .resources()
        .iter()
        .map(|resource| resource.resource().name())
        .collect::<Vec<_>>();
    let passes = graph
        .passes()
        .iter()
        .map(|pass| pass.name())
        .collect::<Vec<_>>();
    let pass_outputs = graph
        .passes()
        .iter()
        .flat_map(|pass| {
            pass.writes().iter().map(move |output| {
                (
                    pass.name(),
                    output.resource().name(),
                    output.state().name(),
                    output.load().name(),
                    output.store().name(),
                )
            })
        })
        .collect::<Vec<_>>();
    let lifetimes = graph
        .lifetimes()
        .iter()
        .map(|lifetime| {
            (
                lifetime.resource().name(),
                lifetime.first_pass(),
                lifetime.last_pass(),
            )
        })
        .collect::<Vec<_>>();
    let alias_candidates = graph
        .optimization_hints()
        .transient_alias_candidates()
        .iter()
        .map(|candidate| (candidate.first().name(), candidate.second().name()))
        .collect::<Vec<_>>();

    tracing::trace!(
        label,
        resource_count = graph.resource_count(),
        pass_count = graph.pass_count(),
        resources = ?resources,
        passes = ?passes,
        pass_outputs = ?pass_outputs,
        transitions = graph.transition_count(),
        barriers = graph.barrier_count(),
        lifetimes = ?lifetimes,
        barrier_merge_candidates = graph.optimization_hints().barrier_merge_candidates(),
        transient_alias_candidates = ?alias_candidates,
        render_pass_merge_candidates = graph.optimization_hints().render_pass_merge_candidates(),
        "compiled render graph"
    );
}

/// Converts scene graph outputs into clear values matching the scene framebuffer attachments.
fn scene_clear_values(
    pass: &GraphPass,
    metadata_required: bool,
) -> Result<Vec<vk::ClearValue>, VulkanError> {
    let mut clear_values = vec![color_clear_value(clear_color_for_output(
        pass,
        crate::renderer::graph::GraphResource::SceneColor,
    )?)];

    if metadata_required {
        clear_values.push(color_clear_value(clear_color_for_output(
            pass,
            crate::renderer::graph::GraphResource::SceneNormalRoughness,
        )?));
        clear_values.push(color_clear_value(clear_color_for_output(
            pass,
            crate::renderer::graph::GraphResource::SceneTransparentNormalRoughness,
        )?));
    }

    clear_values.push(depth_clear_value());

    Ok(clear_values)
}

/// Returns the clear color declared for one scene output resource.
fn clear_color_for_output(
    pass: &GraphPass,
    resource: crate::renderer::graph::GraphResource,
) -> Result<[f32; 4], VulkanError> {
    pass.writes()
        .iter()
        .find(|output| output.resource() == resource)
        .map(PassOutput::clear_color)
        .ok_or_else(|| {
            VulkanError::GraphCompile(format!("scene pass has no {} output", resource.name()))
        })
}

/// Returns one color clear value for color/depth render pass attachments.
fn color_clear_value(color: [f32; 4]) -> vk::ClearValue {
    vk::ClearValue {
        color: vk::ClearColorValue { float32: color },
    }
}

/// Returns the canonical depth clear used by shadow and depth attachments.
fn depth_clear_value() -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    }
}

/// Returns the Vulkan layout corresponding to a graph resource state.
fn layout_for_state(state: ResourceState) -> vk::ImageLayout {
    match state {
        ResourceState::Undefined => vk::ImageLayout::UNDEFINED,
        ResourceState::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ResourceState::DepthAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ResourceState::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ResourceState::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ResourceState::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

/// Returns the source stage mask used when leaving a graph resource state.
fn stage_for_source_state(state: ResourceState) -> vk::PipelineStageFlags {
    match state {
        ResourceState::Undefined => vk::PipelineStageFlags::TOP_OF_PIPE,
        ResourceState::ColorAttachment => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ResourceState::DepthAttachment => {
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        }
        ResourceState::ShaderRead => vk::PipelineStageFlags::FRAGMENT_SHADER,
        ResourceState::TransferSrc => vk::PipelineStageFlags::TRANSFER,
        ResourceState::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    }
}

/// Returns the destination stage mask used when entering a graph resource state.
fn stage_for_destination_state(state: ResourceState) -> vk::PipelineStageFlags {
    match state {
        ResourceState::Undefined => vk::PipelineStageFlags::TOP_OF_PIPE,
        ResourceState::ColorAttachment => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ResourceState::DepthAttachment => {
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        }
        ResourceState::ShaderRead => vk::PipelineStageFlags::FRAGMENT_SHADER,
        ResourceState::TransferSrc => vk::PipelineStageFlags::TRANSFER,
        ResourceState::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    }
}

/// Returns the source access mask used when leaving a graph resource state.
fn access_for_source_state(state: ResourceState) -> vk::AccessFlags {
    match state {
        ResourceState::ColorAttachment => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ResourceState::DepthAttachment => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ResourceState::ShaderRead => vk::AccessFlags::SHADER_READ,
        ResourceState::TransferSrc => vk::AccessFlags::TRANSFER_READ,
        ResourceState::Undefined | ResourceState::Present => vk::AccessFlags::empty(),
    }
}

/// Returns the destination access mask used when entering a graph resource state.
fn access_for_destination_state(state: ResourceState) -> vk::AccessFlags {
    match state {
        ResourceState::ColorAttachment => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ResourceState::DepthAttachment => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ResourceState::ShaderRead => vk::AccessFlags::SHADER_READ,
        ResourceState::TransferSrc => vk::AccessFlags::TRANSFER_READ,
        ResourceState::Undefined | ResourceState::Present => vk::AccessFlags::empty(),
    }
}

/// Returns the subresource range touched by one graph image barrier.
fn subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

/// Submits one recorded frame and signals its in-flight fence on completion.
fn submit_frame(device: &Device, queue: vk::Queue, frame: ActiveFrame) -> Result<(), VulkanError> {
    let wait_semaphores = [frame.image_available];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let command_buffers = [frame.command_buffer];
    let signal_semaphores = [frame.render_finished];
    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);

    reset_fences(device, &[frame.in_flight])?;

    // Safety: the command buffer has ended recording, wait/signal semaphores are owned by this
    // frame slot, and the fence was reset immediately before submission.
    unsafe { device.queue_submit(queue, &[submit_info], frame.in_flight) }.map_err(VulkanError::Vk)
}

/// Presents the submitted swapchain image after rendering has signaled completion.
fn present_frame(
    swapchain_loader: &khr::swapchain::Device,
    queue: vk::Queue,
    swapchain: &VulkanSwapchain,
    frame: ActiveFrame,
) -> Result<FramePresentStatus, VulkanError> {
    let wait_semaphores = [frame.render_finished];
    let swapchains = [swapchain.handle];
    let image_indices = [frame.image_index];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    // Safety: the swapchain image was acquired for this frame and rendering signals
    // `render_finished` before presentation waits on it.
    match unsafe { swapchain_loader.queue_present(queue, &present_info) } {
        Ok(suboptimal) => {
            if suboptimal {
                tracing::trace!(
                    image_index = frame.image_index,
                    "presented suboptimal Vulkan swapchain image"
                );
            }
            Ok(FramePresentStatus::Presented { readback: None })
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(FramePresentStatus::SwapchainOutOfDate),
        Err(error) => Err(VulkanError::Vk(error)),
    }
}

/// Waits for the requested fences to signal.
fn wait_for_fences(device: &Device, fences: &[vk::Fence]) -> Result<(), VulkanError> {
    if fences.is_empty() {
        return Ok(());
    }

    // Safety: all fences were created by this device and remain alive for this wait.
    unsafe { device.wait_for_fences(fences, true, u64::MAX) }.map_err(VulkanError::Vk)
}

/// Resets fences after command recording has succeeded and before queue submission.
fn reset_fences(device: &Device, fences: &[vk::Fence]) -> Result<(), VulkanError> {
    // Safety: all fences were created by this device and are not in use after the matching wait.
    unsafe { device.reset_fences(fences) }.map_err(VulkanError::Vk)
}

/// Resets one reusable command buffer before recording a new frame into it.
fn reset_command_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
) -> Result<(), VulkanError> {
    // Safety: the command buffer belongs to the frame command pool and its in-flight fence has
    // signaled before this reset.
    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }
        .map_err(VulkanError::Vk)
}

/// Destroys all semaphores and fences owned by a set of frame slots.
fn destroy_frame_slots(device: &Device, slots: Vec<VulkanFrameSlot>) {
    for slot in slots {
        destroy_fence(device, slot.in_flight);
        destroy_semaphore(device, slot.image_available);
    }
}

/// Destroys a group of semaphores created by this frame system.
fn destroy_semaphores(device: &Device, semaphores: Vec<vk::Semaphore>) {
    for semaphore in semaphores {
        destroy_semaphore(device, semaphore);
    }
}

/// Destroys one semaphore created for frame synchronization.
fn destroy_semaphore(device: &Device, semaphore: vk::Semaphore) {
    if semaphore == vk::Semaphore::null() {
        return;
    }

    // Safety: the semaphore was created by this device and is destroyed after the device is idle.
    unsafe {
        device.destroy_semaphore(semaphore, None);
    }
}

/// Destroys one fence created for frame synchronization.
fn destroy_fence(device: &Device, fence: vk::Fence) {
    if fence == vk::Fence::null() {
        return;
    }

    // Safety: the fence was created by this device and is destroyed after the device is idle.
    unsafe {
        device.destroy_fence(fence, None);
    }
}

/// Destroys the frame command pool and all command buffers allocated from it.
fn destroy_command_pool(device: &Device, command_pool: vk::CommandPool) {
    if command_pool == vk::CommandPool::null() {
        return;
    }

    // Safety: all command buffer work has completed and command buffers are freed with the pool.
    unsafe {
        device.destroy_command_pool(command_pool, None);
    }
}
