use ash::{Device, khr, vk};

use crate::{
    math::{add3, cross3, dot3, mul3, normalize_or, sub3},
    protocol::{CameraSnapshot, FrameSnapshot, RenderItemPacket, RenderQualitySettings},
    renderer::{
        graph::{
            BarrierLocation, FrameGraphPlan, GraphPass, PassOutput, ResourceBarrier, ResourceState,
            SHADOW_CASCADE_COUNT,
        },
        shadow_cascade_size,
    },
};

use super::{
    ShadowFrameData, ShadowFrameSignature, VulkanDevice, VulkanError,
    material::VulkanMaterialStore,
    mesh::{
        MeshDrawOptions, MeshFrameUniform, MeshPassResources, MeshPipelineSet, ShadowCascadeCull,
        VulkanMeshStore,
    },
    readback::{FramebufferReadbackCopy, FramebufferReadbackSample, record_image_to_buffer},
    swapchain::{ShadowResources, VulkanSwapchain},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.015, 0.018, 0.026, 1.0];
const DEFAULT_AMBIENT_COLOR: [f32; 4] = [0.014, 0.017, 0.024, 0.55];
const SHADOW_SPLIT_LAMBDA: f32 = 0.78;
const SHADOW_SPLIT_NEAR_FLOOR: f32 = 1.0;
const SHADOW_RADIUS_PADDING: f32 = 1.04;
const SHADOW_MIN_RADIUS: f32 = 4.0;
const SHADOW_DEPTH_PADDING: f32 = 24.0;
const SHADOW_SIGNATURE_POSITION_STEP: f32 = 0.45;
const SHADOW_SIGNATURE_DIRECTION_STEP: f32 = 0.012;
const SHADOW_SIGNATURE_FOV_STEP: f32 = 0.010;

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
        if features.has_shadow_casters {
            self.ensure_shadow_resources()?;
        }
        let shadow_signature = shadow_frame_signature(snapshot, features);
        let refresh_shadows = self.shadow_cache.needs_refresh(shadow_signature);
        let cached_shadow_data = self.shadow_cache.frame_data();
        let current_shadow_data =
            if refresh_shadows || (features.has_shadow_casters && cached_shadow_data.is_none()) {
                shadow_frame_data(snapshot, swapchain.extent_2d(), features)
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
            FrameGraphPlan::standard_frame_with_readback(
                DEFAULT_CLEAR_COLOR,
                initial_states,
                readback.copy.is_some(),
                features.has_shadow_casters,
                features.has_translucent_shadow_casters,
            )
        } else {
            FrameGraphPlan::standard_frame_with_shadow_refresh(
                DEFAULT_CLEAR_COLOR,
                initial_states,
                readback.copy.is_some(),
                features.has_shadow_casters,
                features.has_translucent_shadow_casters,
                false,
            )
        }
        .map_err(|error| VulkanError::GraphCompile(error.to_string()))?;
        trace_compiled_graph("standard_frame_executor", &graph);

        self.meshes.write_frame_uniform(
            &self.device,
            frame.slot_index,
            mesh_frame_uniform_for_frame(snapshot, swapchain.extent_2d(), features, shadow_data),
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
        let casts_translucent_shadow =
            casts_shadow && materials.casts_translucent_shadow(item.material);
        flags.has_translucent_shadow_casters |= casts_translucent_shadow;
        flags.has_shadow_casters |= casts_translucent_shadow
            || (casts_shadow && materials.casts_opaque_shadow(item.material));
        if flags.has_shadow_casters && flags.has_translucent_shadow_casters {
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
    if let Some(cascade_index) = shadow_cascade_index(name) {
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
    if let Some(cascade_index) = translucent_shadow_cascade_index(name) {
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

/// Returns the opaque shadow cascade index encoded in a graph pass name.
fn shadow_cascade_index(pass_name: &str) -> Option<usize> {
    pass_name
        .strip_prefix("shadow_cascade_")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|index| *index < crate::renderer::graph::SHADOW_CASCADE_COUNT)
}

/// Returns the translucent shadow cascade index encoded in a graph pass name.
fn translucent_shadow_cascade_index(pass_name: &str) -> Option<usize> {
    pass_name
        .strip_prefix("translucent_shadow_")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|index| *index < crate::renderer::graph::SHADOW_CASCADE_COUNT)
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
) -> Result<(), VulkanError> {
    let clear_values = scene_clear_values(pass)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain.extent_2d());
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.scene_render_pass())
        .framebuffer(swapchain.scene_framebuffer())
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
        let mut opaque_count = 0_usize;
        let mut transparent_count = 0_usize;
        for item in &snapshot.render_items {
            if materials.is_transparent(item.material) {
                continue;
            }
            if meshes.bind_and_draw(
                device,
                frame.command_buffer,
                swapchain.mesh_pipeline(),
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
                swapchain.transparent_mesh_pipeline(),
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
            "recorded scene mesh draw groups"
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records mesh depth for items that explicitly cast shadows.
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
    let clear_values = [depth_clear_value()];
    let shadow_extent = shadows.extent_2d(cascade_index)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.shadow_render_pass())
        .framebuffer(shadows.shadow_framebuffer(cascade_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place the shadow map in depth attachment layout before this pass.
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
            Some(shadows.mesh_pass_resources()),
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

/// Builds the stable frame signature used to decide whether cached shadow maps can be reused.
fn shadow_frame_signature(
    snapshot: &FrameSnapshot,
    features: FrameFeatureFlags,
) -> Option<ShadowFrameSignature> {
    features.has_shadow_casters.then(|| {
        let camera = active_camera(snapshot);
        let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);

        ShadowFrameSignature {
            camera_eye_bucket: quantize3(camera.eye, SHADOW_SIGNATURE_POSITION_STEP),
            camera_forward_bucket: quantize3(forward, SHADOW_SIGNATURE_DIRECTION_STEP),
            fov_bucket: quantize_scalar(camera.fov_y_radians, SHADOW_SIGNATURE_FOV_STEP),
            caster_hash: shadow_caster_hash(&snapshot.render_items),
            translucent_casters: features.has_translucent_shadow_casters,
        }
    })
}

/// Builds the shadow matrices that correspond to freshly rendered shadow maps.
fn shadow_frame_data(
    snapshot: &FrameSnapshot,
    extent: vk::Extent2D,
    features: FrameFeatureFlags,
) -> Option<ShadowFrameData> {
    features.has_shadow_casters.then(|| {
        let aspect = if extent.height > 0 {
            extent.width as f32 / extent.height as f32
        } else {
            1.0
        };
        let camera = active_camera(snapshot);
        let light_dir = normalize_or(super::DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
        let splits = shadow_cascade_splits(camera);

        let projections = shadow_view_projections(camera, aspect, light_dir);
        ShadowFrameData {
            view_proj: std::array::from_fn(|index| projections[index].view_projection),
            splits,
            texel_world: shadow_cascade_metric_vec4(&projections, |projection| {
                projection.texel_world
            }),
            depth_span: shadow_cascade_metric_vec4(&projections, |projection| {
                projection.depth_span
            }),
        }
    })
}

/// Hashes the renderer-facing shadow caster set without depending on ECS state.
fn shadow_caster_hash(items: &[RenderItemPacket]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for item in items
        .iter()
        .filter(|item| item.flags.visible && item.flags.casts_shadow)
    {
        hash = fnv1a(hash, item.mesh.raw());
        hash = fnv1a(hash, item.material.raw());
        hash = fnv1a(hash, item.layer as u64);
        hash = fnv1a(hash, item.object_id.map_or(0, |id| id.raw()));
    }

    hash
}

/// Mixes one integer into a small deterministic FNV-1a hash.
fn fnv1a(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
}

/// Quantizes one vector so tiny camera changes do not force shadow-map redraws.
fn quantize3(value: [f32; 3], step: f32) -> [i32; 3] {
    [
        quantize_scalar(value[0], step),
        quantize_scalar(value[1], step),
        quantize_scalar(value[2], step),
    ]
}

/// Quantizes one finite scalar into a stable cache signature bucket.
fn quantize_scalar(value: f32, step: f32) -> i32 {
    if !value.is_finite() || step <= f32::EPSILON {
        return 0;
    }

    (value / step)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Returns the conservative camera-depth culling window for one shadow cascade.
fn shadow_cascade_cull(camera: CameraSnapshot, cascade_index: usize) -> ShadowCascadeCull {
    let (min_depth, max_depth) = shadow_cascade_depth_range(camera, cascade_index);

    ShadowCascadeCull::new(camera, min_depth, max_depth)
}

/// Returns the camera-space near/far depths that feed one cascade projection.
fn shadow_cascade_depth_range(camera: CameraSnapshot, cascade_index: usize) -> (f32, f32) {
    let splits = shadow_cascade_splits(camera);
    let min_depth = match cascade_index {
        0 => camera.near.max(0.03),
        index => splits[index.saturating_sub(1)].max(camera.near.max(0.03)),
    };
    let max_depth = splits[cascade_index.min(SHADOW_CASCADE_COUNT - 1)].max(min_depth + 1.0);

    (min_depth, max_depth)
}

/// Builds the frame uniform consumed by mesh vertex and fragment shaders.
fn mesh_frame_uniform_for_frame(
    snapshot: &FrameSnapshot,
    extent: vk::Extent2D,
    features: FrameFeatureFlags,
    shadow_data: Option<ShadowFrameData>,
) -> MeshFrameUniform {
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let camera = active_camera(snapshot);
    let light_intensity = snapshot
        .lights
        .first()
        .map(|light| light.intensity)
        .unwrap_or(1.0)
        .max(0.0);
    let light_dir = normalize_or(super::DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
    let shadow_data = shadow_data.unwrap_or_else(disabled_shadow_frame_data);

    MeshFrameUniform {
        view_proj: camera.view_projection(aspect),
        view: look_at_rh(camera.eye, camera.target, camera.up),
        shadow_view_proj: shadow_data.view_proj,
        shadow_cascade_splits: shadow_data.splits,
        shadow_cascade_texel_world: shadow_data.texel_world,
        shadow_cascade_depth_span: shadow_data.depth_span,
        camera_pos: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
        light_dir: [
            light_dir[0],
            light_dir[1],
            light_dir[2],
            if features.has_shadow_casters {
                1.0
            } else {
                0.0
            },
        ],
        light_color: [
            3.00 * light_intensity,
            2.65 * light_intensity,
            2.15 * light_intensity,
            if features.has_translucent_shadow_casters {
                1.0
            } else {
                0.0
            },
        ],
        ambient_color: DEFAULT_AMBIENT_COLOR,
    }
}

/// Returns inert shadow matrices for frames that have no live shadow casters.
fn disabled_shadow_frame_data() -> ShadowFrameData {
    ShadowFrameData {
        view_proj: [identity_mat4(); SHADOW_CASCADE_COUNT],
        splits: [20.0, 52.0, 128.0, 320.0],
        texel_world: [1.0, 1.0, 1.0, 1.0],
        depth_span: [1.0, 1.0, 1.0, 1.0],
    }
}

#[derive(Clone, Copy)]
struct ShadowCascadeProjection {
    view_projection: [f32; 16],
    texel_world: f32,
    depth_span: f32,
}

/// Packs cascade metrics into the vec4 layout consumed by mesh shaders.
fn shadow_cascade_metric_vec4(
    projections: &[ShadowCascadeProjection; SHADOW_CASCADE_COUNT],
    value: impl Fn(ShadowCascadeProjection) -> f32,
) -> [f32; 4] {
    let mut output = [0.0; 4];
    for (index, projection) in projections.iter().copied().enumerate() {
        output[index] = value(projection);
    }
    output
}

/// Returns an identity matrix for disabled shadow sampling state.
fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Builds directional-light projections for camera-distance cascades.
fn shadow_view_projections(
    camera: CameraSnapshot,
    aspect: f32,
    light_dir: [f32; 3],
) -> [ShadowCascadeProjection; SHADOW_CASCADE_COUNT] {
    let splits = shadow_cascade_splits(camera);
    let mut cascade_near = camera.near.max(0.03);

    std::array::from_fn(|cascade_index| {
        let cascade_far = splits[cascade_index].max(cascade_near + 1.0);
        let projection = shadow_view_projection(
            camera,
            aspect,
            light_dir,
            cascade_near,
            cascade_far,
            cascade_index,
        );
        cascade_near = cascade_far;
        projection
    })
}

/// Builds a stable directional-light projection for one camera cascade range.
fn shadow_view_projection(
    camera: CameraSnapshot,
    aspect: f32,
    light_dir: [f32; 3],
    cascade_near: f32,
    cascade_far: f32,
    cascade_index: usize,
) -> ShadowCascadeProjection {
    let frustum = camera_frustum_corners(camera, aspect, cascade_near, cascade_far);
    let (center, radius) = bounding_sphere(&frustum);
    let shadow_resolution = shadow_cascade_resolution(cascade_index);
    let radius = quantize_shadow_radius(
        (radius * SHADOW_RADIUS_PADDING).max(SHADOW_MIN_RADIUS),
        shadow_resolution,
    );
    let mut view = stable_light_view(light_dir, center, radius);

    snap_shadow_view_to_texels(&mut view, center, radius, shadow_resolution);
    let (near, far) = shadow_depth_range(&view, radius, &frustum, center);
    let texel_world = radius * 2.0 / shadow_resolution;
    let depth_span = (far - near).max(1.0);

    tracing::trace!(
        cascade_index,
        cascade_near,
        cascade_far,
        radius,
        texel_world,
        shadow_resolution,
        near,
        far,
        "built camera-cascade shadow projection"
    );

    ShadowCascadeProjection {
        view_projection: mat4_mul(
            orthographic_vulkan(radius * 2.0, radius * 2.0, near, far),
            view,
        ),
        texel_world,
        depth_span,
    }
}

fn shadow_cascade_splits(camera: CameraSnapshot) -> [f32; 4] {
    let near = camera.near.max(0.03);
    let far = shadow_coverage_distance(camera);
    let range = (far - near).max(1.0);
    let split_near = near.max(SHADOW_SPLIT_NEAR_FLOOR);
    let ratio = (far / split_near).max(1.0);
    let mut previous = near;

    std::array::from_fn(|index| {
        let t = (index + 1) as f32 / SHADOW_CASCADE_COUNT as f32;
        let uniform = near + range * t;
        let logarithmic = split_near * ratio.powf(t);
        let split = logarithmic * SHADOW_SPLIT_LAMBDA + uniform * (1.0 - SHADOW_SPLIT_LAMBDA);
        let split = if index + 1 == SHADOW_CASCADE_COUNT {
            far
        } else {
            split.clamp(
                previous + 1.0,
                far - (SHADOW_CASCADE_COUNT - index - 1) as f32,
            )
        };
        previous = split;
        split
    })
}

/// Returns the camera-local shadow distance so scene scale does not dilute cascade texels.
fn shadow_coverage_distance(camera: CameraSnapshot) -> f32 {
    const MAX_SHADOW_DISTANCE: f32 = 320.0;
    let near = camera.near.max(0.03);

    camera
        .far
        .max(near + SHADOW_CASCADE_COUNT as f32)
        .min(MAX_SHADOW_DISTANCE)
}

fn shadow_cascade_resolution(cascade_index: usize) -> f32 {
    shadow_cascade_size(cascade_index) as f32
}

fn quantize_shadow_radius(radius: f32, resolution: f32) -> f32 {
    let texel_target = (radius * 2.0 / resolution.max(1.0)).max(0.001);
    let step = (texel_target * 32.0).clamp(0.25, 8.0);

    (radius / step).ceil() * step
}

fn camera_frustum_corners(
    camera: CameraSnapshot,
    aspect: f32,
    near: f32,
    far: f32,
) -> [[f32; 3]; 8] {
    let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);
    let tan_y = (camera.fov_y_radians * 0.5).tan().max(0.001);
    let tan_x = tan_y * aspect.max(0.001);
    let mut corners = [[0.0; 3]; 8];

    for (plane, depth) in [near, far].into_iter().enumerate() {
        let center = add3(camera.eye, mul3(forward, depth));
        let x = mul3(right, tan_x * depth);
        let y = mul3(up, tan_y * depth);
        let base = plane * 4;
        corners[base] = add3(sub3(center, x), y);
        corners[base + 1] = add3(add3(center, x), y);
        corners[base + 2] = sub3(add3(center, x), y);
        corners[base + 3] = sub3(sub3(center, x), y);
    }

    corners
}

fn bounding_sphere(points: &[[f32; 3]]) -> ([f32; 3], f32) {
    let mut center = [0.0; 3];
    for point in points {
        center = add3(center, *point);
    }
    center = mul3(center, 1.0 / points.len().max(1) as f32);

    let radius = points
        .iter()
        .map(|point| distance_squared(center, *point))
        .fold(0.0_f32, f32::max)
        .sqrt();

    (center, radius)
}

fn stable_light_view(light_dir: [f32; 3], center: [f32; 3], radius: f32) -> [f32; 16] {
    let light_dir = normalize_or(light_dir, [0.0, -1.0, 0.0]);
    let eye = sub3(center, mul3(light_dir, radius * 3.0 + 16.0));
    let up = if dot3(light_dir, [0.0, 1.0, 0.0]).abs() > 0.92 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    look_at_rh(eye, center, up)
}

fn snap_shadow_view_to_texels(
    view: &mut [f32; 16],
    center: [f32; 3],
    radius: f32,
    resolution: f32,
) {
    let texel_world = radius * 2.0 / resolution.max(1.0);
    if texel_world <= f32::EPSILON {
        return;
    }

    let center_in_light_space = transform_point(*view, center);
    let snapped_x = (center_in_light_space[0] / texel_world).round() * texel_world;
    let snapped_y = (center_in_light_space[1] / texel_world).round() * texel_world;

    view[12] += snapped_x - center_in_light_space[0];
    view[13] += snapped_y - center_in_light_space[1];
}

fn shadow_depth_range(
    view: &[f32; 16],
    shadow_radius: f32,
    receiver_points: &[[f32; 3]],
    focus_center: [f32; 3],
) -> (f32, f32) {
    let mut min_depth = f32::INFINITY;
    let mut max_depth = f32::NEG_INFINITY;
    let mut include_depth = |center: [f32; 3], radius: f32| {
        let light_center = transform_point(*view, center);
        let depth = -light_center[2];
        min_depth = min_depth.min(depth - radius);
        max_depth = max_depth.max(depth + radius);
    };

    include_depth(focus_center, shadow_radius);
    for point in receiver_points {
        include_depth(*point, 0.0);
    }

    if !min_depth.is_finite() || !max_depth.is_finite() {
        return (0.1, (shadow_radius * 4.0).max(16.0));
    }

    let margin = (shadow_radius * 0.08).clamp(1.0, SHADOW_DEPTH_PADDING);
    let near = (min_depth - margin).max(0.05);
    let far = (max_depth + margin).max(near + 8.0);
    quantize_depth_range(near, far, shadow_radius)
}

fn quantize_depth_range(near: f32, far: f32, radius: f32) -> (f32, f32) {
    let step = (radius * 0.025).clamp(0.25, 8.0);
    let near = (near / step).floor() * step;
    let far = (far / step).ceil() * step;

    (near.max(0.05), far.max(near + step * 4.0))
}

fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    let x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
    let y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
    let z = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
    let w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];

    if w.abs() > f32::EPSILON {
        [x / w, y / w, z / w]
    } else {
        [x, y, z]
    }
}

fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];

    dx * dx + dy * dy + dz * dz
}

/// Builds a Vulkan clip-space orthographic projection with NDC depth in 0..1.
fn orthographic_vulkan(width: f32, height: f32, near: f32, far: f32) -> [f32; 16] {
    let z = 1.0 / (near - far);
    [
        2.0 / width,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / height,
        0.0,
        0.0,
        0.0,
        0.0,
        z,
        0.0,
        0.0,
        0.0,
        near * z,
        1.0,
    ]
}

/// Builds a right-handed view matrix from explicit camera basis vectors.
fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let forward = normalize_or(sub3(target, eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);

    [
        right[0],
        up[0],
        -forward[0],
        0.0,
        right[1],
        up[1],
        -forward[1],
        0.0,
        right[2],
        up[2],
        -forward[2],
        0.0,
        -dot3(right, eye),
        -dot3(up, eye),
        dot3(forward, eye),
        1.0,
    ]
}

/// Multiplies two column-major 4x4 matrices.
fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = a[row] * b[column * 4]
                + a[4 + row] * b[column * 4 + 1]
                + a[8 + row] * b[column * 4 + 2]
                + a[12 + row] * b[column * 4 + 3];
        }
    }
    out
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
fn scene_clear_values(pass: &GraphPass) -> Result<[vk::ClearValue; 4], VulkanError> {
    Ok([
        color_clear_value(clear_color_for_output(
            pass,
            crate::renderer::graph::GraphResource::SceneColor,
        )?),
        color_clear_value(clear_color_for_output(
            pass,
            crate::renderer::graph::GraphResource::SceneNormalRoughness,
        )?),
        color_clear_value(clear_color_for_output(
            pass,
            crate::renderer::graph::GraphResource::SceneTransparentNormalRoughness,
        )?),
        depth_clear_value(),
    ])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraSnapshot {
        CameraSnapshot::perspective(
            [0.0, 1.5, 6.0],
            [0.0, 1.5, 5.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            5000.0,
        )
        .expect("test camera is finite")
    }

    // Verifies that cascade coverage is camera-local even when the view far plane is huge.
    #[test]
    fn shadow_cascade_splits_keep_near_density_without_following_scene_scale() {
        let splits = shadow_cascade_splits(camera());

        assert!(splits[0] < 32.0);
        assert!(splits[0] < splits[1]);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert!(splits[3] < camera().far);
        assert_eq!(splits[3], shadow_coverage_distance(camera()));
    }

    // Verifies that near cascade density is higher than far cascade density.
    #[test]
    fn near_shadow_cascade_has_smaller_world_texels() {
        let camera = camera();
        let light_dir = normalize_or(
            super::super::DEFAULT_DIRECTIONAL_LIGHT_DIR,
            [0.0, -1.0, 0.0],
        );
        let splits = shadow_cascade_splits(camera);
        let near = shadow_view_projection(camera, 16.0 / 9.0, light_dir, camera.near, splits[0], 0);
        let far = shadow_view_projection(camera, 16.0 / 9.0, light_dir, splits[2], splits[3], 3);

        assert!(near.view_projection[0].abs() > far.view_projection[0].abs());
        assert!(near.texel_world < far.texel_world);
        assert!(near.depth_span > 0.0);
        assert!(far.depth_span > 0.0);
    }

    // Verifies that shader vec4 metrics pack all four cascade values.
    #[test]
    fn shadow_cascade_metrics_pack_four_values_into_vec4() {
        let projections = std::array::from_fn(|index| ShadowCascadeProjection {
            view_projection: identity_mat4(),
            texel_world: (index + 1) as f32,
            depth_span: 10.0 + index as f32,
        });

        assert_eq!(
            shadow_cascade_metric_vec4(&projections, |projection| projection.texel_world),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            shadow_cascade_metric_vec4(&projections, |projection| projection.depth_span),
            [10.0, 11.0, 12.0, 13.0]
        );
    }
}
