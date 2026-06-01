use ash::{Device, khr, vk};

use crate::{
    math::{add3, cross3, dot3, mul3, normalize_or, sub3},
    protocol::{CameraEffects, CameraSnapshot, DrawPacket, FrameSnapshot},
    renderer::{
        SHADOW_FAR_PLANE, SHADOW_MAP_SIZE, SHADOW_NEAR_PLANE, SHADOW_VIEW_DISTANCE,
        SHADOW_WORLD_SIZE,
        graph::{
            BarrierLocation, FrameGraphPlan, GraphPass, LoadOp, PassOutput, ResourceBarrier,
            ResourceState,
        },
    },
};

use super::{
    VulkanDevice, VulkanError,
    material::VulkanMaterialStore,
    mesh::{MeshFrameUniform, MeshPassResources, MeshPipelineSet, VulkanMeshStore},
    readback::{FramebufferReadbackCopy, FramebufferReadbackSample, record_image_to_buffer},
    swapchain::VulkanSwapchain,
    triangle::DebugTriangleResources,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.015, 0.018, 0.026, 1.0];
const DEFAULT_AMBIENT_COLOR: [f32; 4] = [0.014, 0.016, 0.020, 1.0];

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
        let graph = FrameGraphPlan::standard_frame_with_readback(
            DEFAULT_CLEAR_COLOR,
            swapchain.graph_initial_states(frame.image_index)?,
            readback.copy.is_some(),
        )
        .map_err(|error| VulkanError::GraphCompile(error.to_string()))?;
        trace_compiled_graph("standard_frame_executor", &graph);

        self.debug_triangle.write_frame_uniform(
            &self.device,
            frame.slot_index,
            debug_triangle_tint(&snapshot.draws),
        )?;
        self.meshes.write_frame_uniform(
            &self.device,
            frame.slot_index,
            mesh_frame_uniform_for_frame(snapshot, swapchain.extent_2d()),
        )?;
        record_graph_command_buffer(
            &self.device,
            frame,
            swapchain,
            &graph,
            &self.debug_triangle,
            &self.materials,
            &self.meshes,
            &snapshot.draws,
            snapshot.camera_effects,
            readback.copy,
        )?;
        submit_frame(&self.device, self.graphics_queue, frame)?;
        swapchain.apply_graph_final_states(frame.image_index, &graph)?;
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
    graph: &FrameGraphPlan,
    debug_triangle: &DebugTriangleResources,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    draws: &[DrawPacket],
    camera_effects: CameraEffects,
    readback: Option<FramebufferReadbackCopy>,
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
        for pass in graph.passes() {
            record_barriers_for_location(
                device,
                frame.command_buffer,
                swapchain,
                frame.image_index,
                graph,
                BarrierLocation::BeforePass(pass.name()),
            )?;
            record_graph_pass(
                device,
                frame,
                swapchain,
                pass,
                debug_triangle,
                materials,
                meshes,
                draws,
                camera_effects,
                readback,
            )?;
        }
        record_barriers_for_location(
            device,
            frame.command_buffer,
            swapchain,
            frame.image_index,
            graph,
            BarrierLocation::AfterGraph,
        )?;
        device.end_command_buffer(frame.command_buffer)?;
    }

    Ok(())
}

/// Records the backend body for one compiled graph pass.
fn record_graph_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    pass: &GraphPass,
    debug_triangle: &DebugTriangleResources,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    draws: &[DrawPacket],
    camera_effects: CameraEffects,
    readback: Option<FramebufferReadbackCopy>,
) -> Result<(), VulkanError> {
    match pass.name() {
        "shadow" => record_shadow_pass(device, frame, swapchain, materials, meshes, draws),
        "scene" => record_scene_pass(
            device,
            frame,
            swapchain,
            pass,
            debug_triangle,
            materials,
            meshes,
            draws,
        ),
        "post" => record_post_pass(device, frame, swapchain, camera_effects),
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
    image_index: u32,
    graph: &FrameGraphPlan,
    location: BarrierLocation,
) -> Result<(), VulkanError> {
    for barrier in graph
        .barriers()
        .iter()
        .copied()
        .filter(|barrier| barrier.location() == location)
    {
        let (image, aspect) = swapchain.graph_image(barrier.resource(), image_index)?;
        record_graph_barrier(device, command_buffer, image, aspect, barrier);
    }

    Ok(())
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
    pass: &GraphPass,
    debug_triangle: &DebugTriangleResources,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    draws: &[DrawPacket],
) -> Result<(), VulkanError> {
    let color_output = pass
        .color_output()
        .ok_or_else(|| VulkanError::GraphCompile("scene pass has no color output".into()))?;
    let clear_values = clear_values_for_output(color_output);
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
        for draw in draws {
            record_draw_packet(
                device,
                frame,
                swapchain,
                debug_triangle,
                materials,
                meshes,
                swapchain.mesh_pipeline(),
                Some(swapchain.mesh_pass_resources()),
                draw,
            )?;
        }
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records mesh depth for items that explicitly cast shadows.
fn record_shadow_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    draws: &[DrawPacket],
) -> Result<(), VulkanError> {
    let clear_values = [depth_clear_value()];
    let shadow_extent = swapchain.shadow_extent_2d();
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.shadow_render_pass())
        .framebuffer(swapchain.shadow_framebuffer())
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place the shadow map in depth attachment layout before this pass.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        record_mesh_draws(
            device,
            frame,
            materials,
            meshes,
            swapchain.shadow_pipeline(),
            None,
            draws,
            shadow_extent,
            MeshDrawFilter::ShadowCasters,
        )?;
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records the post pass that samples scene color and writes the swapchain image.
fn record_post_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    camera_effects: CameraEffects,
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
            camera_effects,
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

/// Records one high-level draw packet inside the active scene render pass.
fn record_draw_packet(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    debug_triangle: &DebugTriangleResources,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    mesh_pipeline: MeshPipelineSet,
    pass_resources: Option<&MeshPassResources>,
    draw: &DrawPacket,
) -> Result<(), VulkanError> {
    match draw {
        DrawPacket::DebugTriangle(_) => debug_triangle.bind_and_draw(
            device,
            frame.command_buffer,
            swapchain.debug_triangle_pipeline(),
            frame.slot_index,
            swapchain.extent_2d(),
        ),
        DrawPacket::Mesh(item) => meshes.bind_and_draw(
            device,
            frame.command_buffer,
            mesh_pipeline,
            materials,
            pass_resources,
            frame.slot_index,
            item,
            swapchain.extent_2d(),
        ),
    }
}

#[derive(Clone, Copy)]
enum MeshDrawFilter {
    ShadowCasters,
}

/// Records mesh-only passes without duplicating per-pass draw loops.
fn record_mesh_draws(
    device: &Device,
    frame: ActiveFrame,
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    pipeline: MeshPipelineSet,
    pass_resources: Option<&MeshPassResources>,
    draws: &[DrawPacket],
    extent: vk::Extent2D,
    filter: MeshDrawFilter,
) -> Result<(), VulkanError> {
    for draw in draws {
        let DrawPacket::Mesh(item) = draw else {
            continue;
        };
        if matches!(filter, MeshDrawFilter::ShadowCasters) && !item.flags.casts_shadow {
            continue;
        }
        meshes.bind_and_draw(
            device,
            frame.command_buffer,
            pipeline,
            materials,
            pass_resources,
            frame.slot_index,
            item,
            extent,
        )?;
    }

    Ok(())
}

/// Builds the frame uniform consumed by mesh vertex and fragment shaders.
fn mesh_frame_uniform_for_frame(
    snapshot: &FrameSnapshot,
    extent: vk::Extent2D,
) -> MeshFrameUniform {
    let aspect = if extent.height > 0 {
        extent.width as f32 / extent.height as f32
    } else {
        1.0
    };
    let camera = snapshot
        .views
        .first()
        .map(|view| view.camera)
        .unwrap_or_default();
    let light_intensity = snapshot
        .lights
        .first()
        .map(|light| light.intensity)
        .unwrap_or(1.0)
        .max(0.0);
    let light_dir = normalize_or([0.45, -1.0, 0.25], [0.0, -1.0, 0.0]);

    MeshFrameUniform {
        view_proj: camera.view_projection(aspect),
        shadow_view_proj: shadow_view_projection(camera, light_dir),
        camera_pos: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
        light_dir: [light_dir[0], light_dir[1], light_dir[2], 0.0],
        light_color: [
            1.0 * light_intensity,
            0.96 * light_intensity,
            0.86 * light_intensity,
            1.0,
        ],
        ambient_color: DEFAULT_AMBIENT_COLOR,
    }
}

/// Builds a bounded directional-light matrix around the current camera target.
fn shadow_view_projection(camera: CameraSnapshot, light_dir: [f32; 3]) -> [f32; 16] {
    let center = snap_shadow_center(camera.target, light_dir);
    let eye = sub3(center, mul3(light_dir, SHADOW_VIEW_DISTANCE));
    let view = look_at_rh(eye, center, [0.0, 1.0, 0.0]);
    mat4_mul(
        orthographic_vulkan(
            SHADOW_WORLD_SIZE,
            SHADOW_WORLD_SIZE,
            SHADOW_NEAR_PLANE,
            SHADOW_FAR_PLANE,
        ),
        view,
    )
}

fn snap_shadow_center(center: [f32; 3], light_dir: [f32; 3]) -> [f32; 3] {
    let forward = normalize_or(light_dir, [0.0, -1.0, 0.0]);
    let right = normalize_or(cross3(forward, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);
    let units_per_texel = SHADOW_WORLD_SIZE / SHADOW_MAP_SIZE as f32;
    let snapped_x = (dot3(center, right) / units_per_texel).floor() * units_per_texel;
    let snapped_y = (dot3(center, up) / units_per_texel).floor() * units_per_texel;
    let depth = dot3(center, forward);

    add3(
        add3(mul3(right, snapped_x), mul3(up, snapped_y)),
        mul3(forward, depth),
    )
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

/// Returns the first debug triangle tint in the frame, or the neutral tint when absent.
fn debug_triangle_tint(draws: &[DrawPacket]) -> [f32; 4] {
    draws
        .iter()
        .find_map(|draw| match draw {
            DrawPacket::DebugTriangle(debug) => Some(debug.tint()),
            DrawPacket::Mesh(_) => None,
        })
        .unwrap_or([1.0, 1.0, 1.0, 1.0])
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

/// Converts a graph pass output into color and depth clears for the current render pass.
fn clear_values_for_output(output: &PassOutput) -> [vk::ClearValue; 2] {
    match output.load() {
        LoadOp::Clear => [color_clear_value(output.clear_color()), depth_clear_value()],
        LoadOp::Load => [color_clear_value([0.0, 0.0, 0.0, 1.0]), depth_clear_value()],
    }
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
