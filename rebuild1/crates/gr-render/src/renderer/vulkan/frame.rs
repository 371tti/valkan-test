use std::collections::HashMap;

use ash::{Device, khr, vk};

use crate::{
    math::{dot3, normalize_or, sub3},
    protocol::{
        CameraSnapshot, FrameSnapshot, LightPacket, LocalLightPacket, RenderItemPacket,
        RenderQualitySettings, SceneBounds,
    },
    renderer::graph::{
        BarrierLocation, FrameGraphPlan, GOD_RAY_MASK_PASS, GOD_RAY_PREFILTER_PASS,
        GOD_RAY_RADIAL_PASS, GOD_RAY_TEMPORAL_PASS, GraphPass, PassOutput, ResourceBarrier,
        ResourceState, SHADOW_CASCADE_COUNT, TAA_RESOLVE_PASS, bloom_downsample_pass_index,
        bloom_upsample_pass_index, shadow_pass_index, translucent_shadow_pass_index,
    },
    renderer::visibility::MeshLodLevel,
};

use super::{
    VulkanDevice, VulkanError,
    god_rays::{GodRayPushConstants, frame_god_ray_sources},
    gpu_timing::GpuFrameTimer,
    material::{MaterialDrawInfo, VulkanMaterialStore},
    mesh::{
        EmissiveLightUniforms, LOCAL_SHADOW_FACE_COUNT, MAX_LOCAL_LIGHTS, MeshDrawOptions,
        MeshDrawState, MeshPassResources, MeshPipelineKey, MeshPipelineSet, ShadowCascadeCull,
        VulkanMeshStore,
    },
    readback::{FramebufferReadbackCopy, FramebufferReadbackSample, record_image_to_buffer},
    shadow::{
        LocalShadowFrameData, LocalShadowLightData, has_local_shadow_light,
        local_shadow_face_contains_bounds_cached, local_shadow_face_culls, local_shadow_frame_data,
        local_shadow_frame_signature, mesh_frame_uniform_for_light, shadow_cascade_cull,
        stable_csm_frame_data_for_resolution,
    },
    swapchain::{ShadowResources, VulkanSwapchain},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.015, 0.018, 0.026, 1.0];
const LOCAL_SHADOW_CASCADE_INDEX: usize = SHADOW_CASCADE_COUNT;

pub(super) struct VulkanFrames {
    command_pool: vk::CommandPool,
    slots: Vec<VulkanFrameSlot>,
    image_render_finished: Vec<vk::Semaphore>,
    gpu_timer: Option<GpuFrameTimer>,
    cursor: usize,
    next_submission_serial: u64,
    last_submitted_serial: u64,
    completed_submission_serial: u64,
}

#[derive(Clone, Copy)]
struct VulkanFrameSlot {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    submitted: bool,
    submitted_frame_id: Option<u64>,
    submitted_serial: Option<u64>,
}

#[derive(Clone, Copy)]
pub(super) struct ActiveFrame {
    slot_index: usize,
    image_index: u32,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    submission_serial: u64,
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
    pub(super) fn create(
        device: &Device,
        queue_family_index: u32,
        timestamp_period_ns: f32,
        timestamp_valid_bits: u32,
    ) -> Result<Self, VulkanError> {
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

        let gpu_timer = GpuFrameTimer::create_if_enabled(
            device,
            slots.len(),
            timestamp_period_ns,
            timestamp_valid_bits,
        );

        tracing::info!(
            frames_in_flight = slots.len(),
            queue_family_index,
            "created Vulkan frame resources"
        );

        Ok(Self {
            command_pool,
            slots,
            image_render_finished: Vec::new(),
            gpu_timer,
            cursor: 0,
            next_submission_serial: 1,
            last_submitted_serial: 0,
            completed_submission_serial: 0,
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

    /// Returns the newest frame serial successfully queued to the graphics queue.
    pub(super) fn last_submitted_serial(&self) -> u64 {
        self.last_submitted_serial
    }

    /// Returns the newest serial proven complete by a reused frame-slot fence.
    pub(super) fn completed_submission_serial(&self) -> u64 {
        self.completed_submission_serial
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
            if let Some(gpu_timer) = &mut self.gpu_timer {
                gpu_timer.trace_completed(device, slot_index, slot.submitted_frame_id);
            }
            if let Some(serial) = slot.submitted_serial {
                self.completed_submission_serial = self.completed_submission_serial.max(serial);
            }
            self.slots[slot_index].submitted = false;
            self.slots[slot_index].submitted_frame_id = None;
            self.slots[slot_index].submitted_serial = None;
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
        let submission_serial = self.next_submission_serial;
        self.next_submission_serial = self.next_submission_serial.saturating_add(1);

        Ok(FrameAcquire::Ready(ActiveFrame {
            slot_index,
            image_index: image,
            command_buffer: slot.command_buffer,
            image_available: slot.image_available,
            render_finished,
            in_flight: slot.in_flight,
            submission_serial,
        }))
    }

    /// Advances the reusable frame slot cursor after a submit/present attempt.
    fn advance(&mut self) {
        self.cursor = (self.cursor + 1) % self.slots.len();
    }

    /// Marks the frame slot as queued so future reuse waits for its fence.
    fn mark_submitted(&mut self, frame: ActiveFrame, frame_id: u64) {
        self.slots[frame.slot_index].submitted = true;
        self.slots[frame.slot_index].submitted_frame_id = Some(frame_id);
        self.slots[frame.slot_index].submitted_serial = Some(frame.submission_serial);
        self.last_submitted_serial = self.last_submitted_serial.max(frame.submission_serial);
    }

    /// Returns the optional timestamp recorder used while this frame's command buffer is open.
    fn gpu_timer_mut(&mut self) -> Option<&mut GpuFrameTimer> {
        self.gpu_timer.as_mut()
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

        if let Some(mut gpu_timer) = self.gpu_timer {
            for (slot_index, slot) in self.slots.iter().enumerate() {
                if slot.submitted {
                    gpu_timer.trace_completed(device, slot_index, slot.submitted_frame_id);
                }
            }
            gpu_timer.destroy(device);
        }
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
        let emissive_lights = local_light_uniforms(snapshot, camera);
        // TAA reprojection requires a fresh normal target every frame.
        let use_full_scene_pass = true;
        let wants_local_shadow =
            features.has_opaque_shadow_casters && has_local_shadow_light(&emissive_lights);
        if features.has_shadow_casters || wants_local_shadow {
            self.ensure_shadow_resources()?;
        }
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
        let taa_frame = swapchain.prepare_taa_frame(
            &self.device,
            frame.slot_index,
            snapshot,
            camera,
            self.quality,
            use_full_scene_pass,
        )?;
        let volumetric_god_rays_active =
            self.quality.bloom().volumetric_god_rays() || self.quality.fog().enabled();
        if volumetric_god_rays_active && taa_frame.reset_reprojection_history {
            // Volumetric God Rays and Fog are accumulated on the low-resolution screen grid. A
            // camera cut, scene discontinuity, or TAA reset must not retain the old ray field.
            swapchain.invalidate_god_ray_history();
        }
        let directional_light = frame_global_light(snapshot);
        let stable_quality = self.quality.stable_csm_pcss();
        let refresh_shadows = features.has_shadow_casters;
        tracing::trace!(
            refresh_shadows,
            blocker_search_samples = stable_quality.blocker_search_samples(),
            filter_samples = stable_quality.filter_samples(),
            "selected stable CSM + PCSS directional shadow frame"
        );
        let shadow_data = features.has_shadow_casters.then(|| {
            stable_csm_frame_data_for_resolution(
                camera,
                swapchain.extent_2d(),
                directional_light,
                self.shadow_map_resolution(),
            )
        });
        let local_shadow_signature = wants_local_shadow
            .then(|| local_shadow_frame_signature(&snapshot.render_items, &emissive_lights));
        let refresh_local_shadows = self
            .shadow_cache
            .local_needs_refresh(local_shadow_signature);
        let cached_local_shadow_data = self.shadow_cache.local_frame_data();
        let current_local_shadow_data = if refresh_local_shadows {
            self.shadows.as_ref().and_then(|shadows| {
                local_shadow_frame_data(&emissive_lights, shadows.local_extent_2d())
            })
        } else {
            None
        };
        let local_shadow_data = if refresh_local_shadows {
            current_local_shadow_data
        } else {
            cached_local_shadow_data.or(current_local_shadow_data)
        };
        let local_shadow_render_data = refresh_local_shadows.then_some(local_shadow_data).flatten();
        let readback = self
            .readback
            .prepare_frame(&self.device, frame.image_index)?;
        let shadows = self.shadows.as_ref();
        let initial_states = swapchain.graph_initial_states(frame.image_index, shadows)?;
        let draw_lists = FrameDrawLists::new(
            &self.materials,
            &self.meshes,
            snapshot,
            swapchain.extent_2d(),
            features,
            refresh_shadows,
            shadow_data,
            local_shadow_render_data,
        );
        let translucent_shadow_cascades = if features.has_translucent_shadow_casters {
            draw_lists.translucent_shadow_cascades()
        } else {
            [false; SHADOW_CASCADE_COUNT]
        };
        tracing::trace!(
            translucent_shadow_cascades = ?translucent_shadow_cascades,
            "selected cascade-local translucent shadow work"
        );
        let bloom_enabled = self.quality.bloom().intensity() > 0.0;
        let god_ray_sources = frame_god_ray_sources(
            camera,
            swapchain.god_ray_extent_2d(),
            self.quality,
            directional_light,
            emissive_lights,
        );
        let volumetric_god_rays = volumetric_god_rays_active;
        // The legacy path is source-gated, while the volumetric path represents the directional
        // sun field and must run even when the source projects outside the camera frustum.
        let god_rays_enabled =
            volumetric_god_rays || god_ray_sources.iter().any(|source| source.source[3] > 0.0);
        let god_ray_history_write_index = swapchain.god_ray_history_write_index();
        let graph =
            FrameGraphPlan::standard_frame_with_shadow_refresh_scene_metadata_bloom_god_rays_and_taa_mode(
                DEFAULT_CLEAR_COLOR,
                initial_states,
                readback.copy.is_some(),
                features.has_shadow_casters,
                translucent_shadow_cascades,
                refresh_shadows,
                use_full_scene_pass,
                bloom_enabled,
                god_rays_enabled,
                god_ray_history_write_index,
                true,
                taa_frame.write_history_index,
                volumetric_god_rays,
            )
            .map_err(|error| VulkanError::GraphCompile(error.to_string()))?;
        trace_compiled_graph("standard_frame_executor", &graph);

        let mut mesh_frame_uniform = mesh_frame_uniform_for_light(
            camera,
            directional_light,
            swapchain.extent_2d(),
            features.has_shadow_casters,
            translucent_shadow_cascades,
            shadow_data,
            stable_quality,
            local_shadow_data,
            emissive_lights,
            snapshot.debug_draw.view_mode,
        );
        mesh_frame_uniform.view_proj = taa_frame.jittered_view_projection;
        self.meshes
            .write_frame_uniform(&self.device, frame.slot_index, mesh_frame_uniform)?;
        let scene_pass_resources = shadows.map_or_else(
            || self.shadow_fallback.mesh_pass_resources(),
            ShadowResources::mesh_pass_resources,
        );
        let directional_shadow_view = scene_pass_resources.directional_shadow_view();
        if swapchain.god_ray_shadow_view() != directional_shadow_view {
            // Descriptor updates are not allowed while an older command buffer may still be
            // reading the set. This transition happens only when lazy CSM resources are created
            // or resized, so the one-time device wait keeps all in-flight frames safe.
            self.wait_idle()?;
            swapchain.update_god_ray_shadow_view(&self.device, directional_shadow_view);
        }
        let gpu_timer = self.frames.gpu_timer_mut();
        record_graph_command_buffer(
            &self.device,
            gpu_timer,
            frame,
            swapchain,
            shadows,
            scene_pass_resources,
            &graph,
            &mut self.meshes,
            snapshot,
            draw_lists,
            readback.copy,
            features,
            self.quality,
            use_full_scene_pass,
            emissive_lights,
            local_shadow_render_data,
        )?;
        submit_frame(&self.device, self.graphics_queue, frame)?;
        swapchain.apply_graph_final_states(frame.image_index, &graph)?;
        if let Some(shadows) = self.shadows.as_mut() {
            shadows.apply_graph_final_states(&graph);
        }
        if refresh_local_shadows {
            self.shadow_cache
                .mark_local_refreshed(local_shadow_signature, current_local_shadow_data);
        }
        if readback.copy.is_some() {
            self.readback
                .mark_copy_recorded(frame.image_index, snapshot.frame_id);
        }
        self.frames.mark_submitted(frame, snapshot.frame_id.raw());
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
        self.invalidate_shadow_state();
        Ok(())
    }

    /// Invalidates cached local-light shadow state after the fixed shadow resources are recreated.
    pub(super) fn invalidate_shadow_state(&mut self) {
        self.shadow_cache.invalidate();
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
        submitted_frame_id: None,
        submitted_serial: None,
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
    mut gpu_timer: Option<&mut GpuFrameTimer>,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    shadows: Option<&ShadowResources>,
    scene_pass_resources: &MeshPassResources,
    graph: &FrameGraphPlan,
    meshes: &mut VulkanMeshStore,
    snapshot: &FrameSnapshot,
    draw_lists: FrameDrawLists<'_>,
    readback: Option<FramebufferReadbackCopy>,
    features: FrameFeatureFlags,
    quality: RenderQualitySettings,
    use_full_scene_pass: bool,
    emissive_lights: EmissiveLightUniforms,
    local_shadow_data: Option<LocalShadowFrameData>,
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
        if let Some(gpu_timer) = gpu_timer.as_deref_mut() {
            gpu_timer.record_start(device, frame.command_buffer, frame.slot_index);
        }
        let mut state = FrameRecordState::new(
            features,
            quality,
            use_full_scene_pass,
            emissive_lights,
            draw_lists,
        );
        if let (Some(shadows), Some(local_shadow)) = (shadows, local_shadow_data) {
            record_local_shadow_passes(
                device,
                frame,
                shadows,
                meshes,
                &state.draw_lists.local_shadow,
                local_shadow,
            )?;
            if let Some(gpu_timer) = gpu_timer.as_deref_mut() {
                gpu_timer.record_checkpoint(
                    device,
                    frame.command_buffer,
                    frame.slot_index,
                    "local_shadow",
                );
            }
        }
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
                meshes,
                snapshot,
                readback,
                &mut state,
            )?;
            if let Some(gpu_timer) = gpu_timer.as_deref_mut() {
                gpu_timer.record_checkpoint(
                    device,
                    frame.command_buffer,
                    frame.slot_index,
                    pass.name(),
                );
            }
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
        if let Some(gpu_timer) = gpu_timer.as_deref_mut() {
            gpu_timer.record_end(device, frame.command_buffer, frame.slot_index);
        }
        device.end_command_buffer(frame.command_buffer)?;
    }

    Ok(())
}

#[derive(Clone, Copy, Default)]
struct FrameFeatureFlags {
    has_shadow_casters: bool,
    has_opaque_shadow_casters: bool,
    has_translucent_shadow_casters: bool,
    has_transparent_scene_items: bool,
}

struct FrameRecordState<'a> {
    features: FrameFeatureFlags,
    quality: RenderQualitySettings,
    use_full_scene_pass: bool,
    emissive_lights: EmissiveLightUniforms,
    draw_lists: FrameDrawLists<'a>,
}

impl<'a> FrameRecordState<'a> {
    /// Carries frame-wide feature flags plus stats discovered while recording graph passes.
    fn new(
        features: FrameFeatureFlags,
        quality: RenderQualitySettings,
        use_full_scene_pass: bool,
        emissive_lights: EmissiveLightUniforms,
        draw_lists: FrameDrawLists<'a>,
    ) -> Self {
        Self {
            features,
            quality,
            use_full_scene_pass,
            emissive_lights,
            draw_lists,
        }
    }
}

struct SceneDrawItem<'a> {
    item: &'a RenderItemPacket,
    options: MeshDrawOptions,
    pipeline_key: MeshPipelineKey,
    material_descriptor_set: vk::DescriptorSet,
    transparent_sort_depth: f32,
}

struct ShadowDrawItem<'a> {
    item: &'a RenderItemPacket,
    pipeline_key: MeshPipelineKey,
    material_descriptor_set: vk::DescriptorSet,
    lod: MeshLodLevel,
}

/// Hot draw facts resolved once before scene and shadow list filtering begins.
///
/// Bounds intentionally remain optional: scene and directional-shadow rendering fail open for
/// incomplete mesh metadata, while local shadows retain their existing fail-closed behavior.
#[derive(Clone, Copy)]
struct ResolvedDrawItem<'a> {
    item: &'a RenderItemPacket,
    material: MaterialDrawInfo,
    bounds: Option<SceneBounds>,
    scene_options: Option<MeshDrawOptions>,
}

struct FrameDrawLists<'a> {
    opaque_scene: Vec<SceneDrawItem<'a>>,
    transparent_scene: Vec<SceneDrawItem<'a>>,
    opaque_shadow: [Vec<ShadowDrawItem<'a>>; SHADOW_CASCADE_COUNT],
    translucent_shadow: [Vec<ShadowDrawItem<'a>>; SHADOW_CASCADE_COUNT],
    local_shadow: [[Vec<ShadowDrawItem<'a>>; LOCAL_SHADOW_FACE_COUNT]; MAX_LOCAL_LIGHTS],
}

impl<'a> FrameDrawLists<'a> {
    /// Prepares per-frame draw lists once so graph passes do not repeat filter/sort/cull work.
    fn new(
        materials: &VulkanMaterialStore,
        meshes: &VulkanMeshStore,
        snapshot: &'a FrameSnapshot,
        extent: vk::Extent2D,
        features: FrameFeatureFlags,
        record_directional_shadow_draws: bool,
        shadow_data: Option<crate::renderer::vulkan::shadow::ShadowFrameData>,
        local_shadow_data: Option<LocalShadowFrameData>,
    ) -> Self {
        let camera = active_camera(snapshot);
        let camera_forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
        let resolved_items = resolve_draw_items(materials, meshes, snapshot, extent, camera);
        let opaque_scene = scene_draw_items(&resolved_items, camera, camera_forward, false);
        let transparent_scene = scene_draw_items(&resolved_items, camera, camera_forward, true);

        // Directional shadow preparation used to scan the resolved items twice per cascade and
        // rebuild the same light-space cull for opaque and translucent passes.  Keep the two pass
        // lists separate, but classify/cull/choose the LOD in one pass so the expensive projection
        // test and mesh metadata lookup are shared.  This is CPU-only and leaves the submitted
        // geometry, sort keys, sample counts, and map resolution unchanged.
        let mut opaque_shadow = std::array::from_fn(|_| Vec::new());
        let mut translucent_shadow = std::array::from_fn(|_| Vec::new());
        if record_directional_shadow_draws
            && (features.has_opaque_shadow_casters || features.has_translucent_shadow_casters)
        {
            if let Some(shadow_data) = shadow_data.as_ref() {
                // ShadowCascadeCull is Copy, but it contains the full projection payload. Build it
                // once per cascade instead of once for each material class.
                let cascade_culls: [ShadowCascadeCull; SHADOW_CASCADE_COUNT] =
                    std::array::from_fn(|cascade_index| {
                        shadow_cascade_cull(camera, cascade_index, shadow_data)
                    });
                for cascade_index in 0..SHADOW_CASCADE_COUNT {
                    let (opaque, translucent) = shadow_draw_items_for_cascade(
                        &resolved_items,
                        meshes,
                        cascade_index,
                        shadow_data.texel_world[cascade_index],
                        shadow_data.cascade_resolution[cascade_index] as u32,
                        cascade_culls[cascade_index],
                        features.has_opaque_shadow_casters,
                        features.has_translucent_shadow_casters,
                    );
                    opaque_shadow[cascade_index] = opaque;
                    translucent_shadow[cascade_index] = translucent;
                }
            }
        }
        let local_shadow = if local_shadow_data.is_some() {
            local_shadow_draw_items(&resolved_items, local_shadow_data)
        } else {
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()))
        };

        if record_directional_shadow_draws && tracing::enabled!(tracing::Level::TRACE) {
            let opaque_indices: [usize; SHADOW_CASCADE_COUNT] =
                std::array::from_fn(|cascade_index| {
                    opaque_shadow[cascade_index]
                        .iter()
                        .map(|draw| meshes.shadow_index_count(draw.item.mesh, draw.lod))
                        .sum::<usize>()
                });
            let translucent_indices: [usize; SHADOW_CASCADE_COUNT] =
                std::array::from_fn(|cascade_index| {
                    translucent_shadow[cascade_index]
                        .iter()
                        .map(|draw| meshes.shadow_index_count(draw.item.mesh, draw.lod))
                        .sum::<usize>()
                });
            let opaque_lods: [[usize; 4]; SHADOW_CASCADE_COUNT] =
                std::array::from_fn(|cascade_index| {
                    let mut counts = [0; 4];
                    for draw in &opaque_shadow[cascade_index] {
                        counts[draw.lod.index()] += 1;
                    }
                    counts
                });
            let translucent_lods: [[usize; 4]; SHADOW_CASCADE_COUNT] =
                std::array::from_fn(|cascade_index| {
                    let mut counts = [0; 4];
                    for draw in &translucent_shadow[cascade_index] {
                        counts[draw.lod.index()] += 1;
                    }
                    counts
                });
            let opaque_fast_counts: [usize; SHADOW_CASCADE_COUNT] =
                std::array::from_fn(|cascade_index| {
                    opaque_shadow[cascade_index]
                        .iter()
                        .filter(|draw| draw.pipeline_key.opaque_shadow)
                        .count()
                });
            tracing::trace!(
                opaque_counts = ?opaque_shadow.each_ref().map(Vec::len),
                translucent_counts = ?translucent_shadow.each_ref().map(Vec::len),
                opaque_indices = ?opaque_indices,
                translucent_indices = ?translucent_indices,
                opaque_lods = ?opaque_lods,
                translucent_lods = ?translucent_lods,
                opaque_fast_counts = ?opaque_fast_counts,
                "built light-space culled directional shadow draw lists"
            );
        }

        Self {
            opaque_scene,
            transparent_scene,
            opaque_shadow,
            translucent_shadow,
            local_shadow,
        }
    }

    /// Returns exactly which persistent transmittance maps receive draws on this refresh.
    fn translucent_shadow_cascades(&self) -> [bool; SHADOW_CASCADE_COUNT] {
        self.translucent_shadow
            .each_ref()
            .map(|draws| !draws.is_empty())
    }
}

/// Resolves the material, bounds, and camera LOD exactly once for list preparation.
fn resolve_draw_items<'a>(
    materials: &VulkanMaterialStore,
    meshes: &VulkanMeshStore,
    snapshot: &'a FrameSnapshot,
    extent: vk::Extent2D,
    camera: CameraSnapshot,
) -> Vec<ResolvedDrawItem<'a>> {
    snapshot
        .render_items
        .iter()
        .filter_map(|item| {
            if !item.flags.visible {
                return None;
            }
            let material = materials.draw_info(item.material)?;
            let mesh = meshes.scene_draw_info(item.mesh, extent, camera, snapshot.optimization);
            Some(ResolvedDrawItem {
                item,
                material,
                bounds: mesh.bounds,
                scene_options: mesh.options,
            })
        })
        .collect()
}

fn scene_draw_items<'a>(
    items: &[ResolvedDrawItem<'a>],
    camera: CameraSnapshot,
    camera_forward: [f32; 3],
    transparent: bool,
) -> Vec<SceneDrawItem<'a>> {
    if transparent {
        let mut draws = items
            .iter()
            .filter_map(|resolved| {
                let item = resolved.item;
                let material = resolved.material;
                if !material.transparent {
                    return None;
                }
                let options = resolved.scene_options?;
                Some(SceneDrawItem {
                    item,
                    options,
                    pipeline_key: MeshPipelineKey {
                        uses_textures: material.uses_any_texture,
                        double_sided: material.double_sided,
                        opaque_scene: material.fully_opaque,
                        opaque_shadow: false,
                    },
                    material_descriptor_set: material.descriptor_set,
                    transparent_sort_depth: transparent_scene_depth(
                        resolved.bounds,
                        camera,
                        camera_forward,
                    ),
                })
            })
            .collect::<Vec<_>>();
        draws.sort_by(|left, right| {
            right
                .transparent_sort_depth
                .total_cmp(&left.transparent_sort_depth)
                .then_with(|| left.item.material.raw().cmp(&right.item.material.raw()))
                .then_with(|| left.item.mesh.raw().cmp(&right.item.mesh.raw()))
        });
        return draws;
    }

    let mut draws = items
        .iter()
        .filter_map(|resolved| {
            let item = resolved.item;
            let material = resolved.material;
            if material.transparent != transparent {
                return None;
            }
            let options = resolved.scene_options?;
            let pipeline_key = MeshPipelineKey {
                uses_textures: material.uses_any_texture,
                double_sided: material.double_sided,
                opaque_scene: material.fully_opaque,
                opaque_shadow: false,
            };
            let draw = SceneDrawItem {
                item,
                options,
                pipeline_key,
                material_descriptor_set: material.descriptor_set,
                transparent_sort_depth: f32::NEG_INFINITY,
            };
            Some((
                (
                    opaque_scene_depth_bucket(resolved.bounds, camera, camera_forward),
                    pipeline_key.opaque_scene,
                    pipeline_key.uses_textures,
                    pipeline_key.double_sided,
                    item.material.raw(),
                    item.mesh.raw(),
                ),
                draw,
            ))
        })
        .collect::<Vec<_>>();

    draws.sort_unstable_by_key(|(key, _)| *key);
    draws.into_iter().map(|(_, draw)| draw).collect()
}

/// Returns the farthest camera-space point of a transparent mesh.
///
/// Alpha blending is order-dependent, so transparent bounds are submitted back-to-front.
/// Using the farthest point is conservative for intersecting bounds and avoids ordering flips
/// when the camera crosses a large thin or double-sided surface.
fn transparent_scene_depth(
    bounds: Option<SceneBounds>,
    camera: CameraSnapshot,
    camera_forward: [f32; 3],
) -> f32 {
    let Some(bounds) = bounds else {
        return f32::NEG_INFINITY;
    };
    let depth = dot3(sub3(bounds.center(), camera.eye), camera_forward) + bounds.radius();
    if depth.is_finite() {
        depth
    } else {
        f32::NEG_INFINITY
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
        let Some(material) = materials.draw_info(item.material) else {
            continue;
        };
        let casts_shadow = item.flags.casts_shadow;
        let casts_opaque_shadow = casts_shadow && material.casts_opaque_shadow;
        let casts_translucent_shadow = casts_shadow && material.casts_translucent_shadow;
        flags.has_transparent_scene_items |= material.transparent;
        flags.has_opaque_shadow_casters |= casts_opaque_shadow;
        flags.has_translucent_shadow_casters |= casts_translucent_shadow;
        flags.has_shadow_casters |= casts_translucent_shadow || casts_opaque_shadow;
        if flags.has_shadow_casters
            && flags.has_opaque_shadow_casters
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
    meshes: &VulkanMeshStore,
    snapshot: &FrameSnapshot,
    readback: Option<FramebufferReadbackCopy>,
    state: &mut FrameRecordState<'_>,
) -> Result<(), VulkanError> {
    let name = pass.name();
    if let Some(cascade_index) = shadow_pass_index(name) {
        let shadows = required_shadow_resources(shadows, name)?;
        return record_shadow_pass(
            device,
            frame,
            shadows,
            cascade_index,
            meshes,
            state.draw_lists.opaque_shadow[cascade_index].as_slice(),
        );
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
            meshes,
            state.draw_lists.translucent_shadow[cascade_index].as_slice(),
        );
    }
    if let Some(mip_index) = bloom_downsample_pass_index(name) {
        return record_bloom_downsample_pass(device, frame, swapchain, mip_index, state.quality);
    }
    if let Some(target_mip_index) = bloom_upsample_pass_index(name) {
        return record_bloom_upsample_pass(
            device,
            frame,
            swapchain,
            target_mip_index,
            state.quality,
        );
    }

    match name {
        GOD_RAY_MASK_PASS => {
            record_god_ray_mask_pass(device, frame, swapchain, snapshot, meshes, state)
        }
        GOD_RAY_PREFILTER_PASS => {
            record_god_ray_prefilter_pass(device, frame, swapchain, snapshot, state)
        }
        GOD_RAY_RADIAL_PASS => {
            record_god_ray_radial_pass(device, frame, swapchain, snapshot, state)
        }
        GOD_RAY_TEMPORAL_PASS => {
            record_god_ray_temporal_pass(device, frame, swapchain, snapshot, state)
        }
        TAA_RESOLVE_PASS => swapchain.record_taa(device, frame.command_buffer),
        "scene" => record_scene_pass(
            device,
            frame,
            swapchain,
            scene_pass_resources,
            pass,
            meshes,
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
        // The graph models one resource per cascade while Vulkan stores the four cascades in one
        // array image. Directional shadow recording performs precise per-layer transitions itself;
        // applying these logical barriers to the full image would touch unrelated cascades.
        if barrier.resource().shadow_cascade().is_some() {
            continue;
        }
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

fn record_image_state_transition_layers(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    from: ResourceState,
    to: ResourceState,
    layer_count: u32,
) {
    let image_barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(access_for_source_state(from))
        .dst_access_mask(access_for_destination_state(to))
        .old_layout(layout_for_state(from))
        .new_layout(layout_for_state(to))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range(aspect).layer_count(layer_count));
    let image_barriers = [image_barrier];

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            stage_for_source_state(from),
            stage_for_destination_state(to),
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );
    }
}

/// Records scene draws into HDR color, material metadata, and depth targets owned by the frame graph.
fn record_scene_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    pass_resources: &MeshPassResources,
    pass: &GraphPass,
    meshes: &VulkanMeshStore,
    state: &FrameRecordState<'_>,
) -> Result<(), VulkanError> {
    let metadata_required = state.use_full_scene_pass;
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
        let mut opaque_draw_state = MeshDrawState::default();
        for draw in &state.draw_lists.opaque_scene {
            if meshes.bind_and_draw(
                device,
                frame.command_buffer,
                opaque_pipeline,
                Some(pass_resources),
                frame.slot_index,
                draw.item,
                draw.material_descriptor_set,
                draw.options,
                draw.pipeline_key,
                &mut opaque_draw_state,
            )? {
                opaque_count += 1;
            }
        }
        let mut transparent_draw_state = MeshDrawState::default();
        for draw in &state.draw_lists.transparent_scene {
            if meshes.bind_and_draw(
                device,
                frame.command_buffer,
                transparent_pipeline,
                Some(pass_resources),
                frame.slot_index,
                draw.item,
                draw.material_descriptor_set,
                draw.options,
                draw.pipeline_key,
                &mut transparent_draw_state,
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

/// Records directional shadow depth for items that explicitly cast shadows.
fn record_shadow_pass(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    cascade_index: usize,
    meshes: &VulkanMeshStore,
    items: &[ShadowDrawItem<'_>],
) -> Result<(), VulkanError> {
    let shadow_extent = shadows.shadow_extent_2d();
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let clear_values = [depth_clear_value()];
    shadows.transition_shadow_layer_to_attachment(device, frame.command_buffer, cascade_index)?;

    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.shadow_render_pass())
        .framebuffer(shadows.shadow_framebuffer(cascade_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let caster_count = record_mesh_draws(
            device,
            frame,
            meshes,
            shadows.shadow_pipeline(),
            None,
            items,
            shadow_extent,
            cascade_index,
        )?;
        if caster_count == 0 {
            tracing::trace!(
                cascade_index,
                "stable CSM shadow cascade cleared without mesh draws"
            );
        }
        device.cmd_end_render_pass(frame.command_buffer);
    }

    shadows.transition_shadow_layer_to_shader_read(device, frame.command_buffer, cascade_index)?;

    Ok(())
}

/// Records depth cubemaps for shadowed local lights before scene lighting samples them.
fn record_local_shadow_passes(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    meshes: &VulkanMeshStore,
    items: &[[Vec<ShadowDrawItem<'_>>; LOCAL_SHADOW_FACE_COUNT]; MAX_LOCAL_LIGHTS],
    local_shadow: LocalShadowFrameData,
) -> Result<(), VulkanError> {
    let command_buffer = frame.command_buffer;
    for local_light in local_shadow.lights.into_iter().flatten() {
        let light_index = local_light.light_index;
        record_image_state_transition_layers(
            device,
            command_buffer,
            shadows.local_depth_image(light_index)?,
            vk::ImageAspectFlags::DEPTH,
            ResourceState::ShaderRead,
            ResourceState::DepthAttachment,
            LOCAL_SHADOW_FACE_COUNT as u32,
        );

        for face_index in 0..LOCAL_SHADOW_FACE_COUNT {
            record_local_shadow_face_pass(
                device,
                frame,
                shadows,
                meshes,
                &items[light_index][face_index],
                light_index,
                face_index,
            )?;
        }

        record_image_state_transition_layers(
            device,
            command_buffer,
            shadows.local_depth_image(light_index)?,
            vk::ImageAspectFlags::DEPTH,
            ResourceState::DepthAttachment,
            ResourceState::ShaderRead,
            LOCAL_SHADOW_FACE_COUNT as u32,
        );

        tracing::trace!(
            light_index,
            caster_count = items[light_index].iter().map(Vec::len).sum::<usize>(),
            face_counts = ?items[light_index].each_ref().map(Vec::len),
            source_radius = local_light.source_radius,
            "recorded local-light cubemap shadow"
        );
    }
    Ok(())
}

fn record_local_shadow_face_pass(
    device: &Device,
    frame: ActiveFrame,
    shadows: &ShadowResources,
    meshes: &VulkanMeshStore,
    items: &[ShadowDrawItem<'_>],
    light_index: usize,
    face_index: usize,
) -> Result<(), VulkanError> {
    let clear_values = [depth_clear_value()];
    let shadow_extent = shadows.local_extent_2d();
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.local_shadow_render_pass())
        .framebuffer(shadows.local_shadow_framebuffer(light_index, face_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let caster_count = record_mesh_draws(
            device,
            frame,
            meshes,
            shadows.local_shadow_pipeline(),
            None,
            items,
            shadow_extent,
            LOCAL_SHADOW_CASCADE_INDEX + light_index * LOCAL_SHADOW_FACE_COUNT + face_index,
        )?;
        if caster_count == 0 {
            tracing::trace!(
                face_index,
                "local shadow cubemap face cleared without mesh draws"
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
    meshes: &VulkanMeshStore,
    items: &[ShadowDrawItem<'_>],
) -> Result<(), VulkanError> {
    let clear_values = [color_clear_value([1.0, 1.0, 1.0, 1.0]), depth_clear_value()];
    let shadow_extent = shadows.extent_2d(cascade_index)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(shadow_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(shadows.translucent_render_pass())
        .framebuffer(shadows.translucent_framebuffer(cascade_index)?)
        .render_area(render_area)
        .clear_values(&clear_values);

    // Safety: graph barriers place transmittance in color-attachment layout. A dedicated cleared
    // depth target keeps the nearest translucent layer; the fragment shader rejects samples behind
    // the immutable opaque blocker-depth map.
    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        let caster_count = record_mesh_draws(
            device,
            frame,
            meshes,
            shadows.translucent_pipeline(),
            Some(shadows.translucent_pass_resources()),
            items,
            shadow_extent,
            cascade_index,
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

/// Records one bloom downsample pass in the HDR mip chain.
fn record_bloom_downsample_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    mip_index: usize,
    quality: RenderQualitySettings,
) -> Result<(), VulkanError> {
    let target_extent = swapchain.bloom_extent_2d(mip_index)?;
    let source_extent = if mip_index == 0 {
        swapchain.extent_2d()
    } else {
        swapchain.bloom_extent_2d(mip_index - 1)?
    };
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(target_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.bloom_downsample_render_pass())
        .framebuffer(swapchain.bloom_downsample_framebuffer(mip_index)?)
        .render_area(render_area);

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.bloom_pipeline().draw_downsample(
            device,
            frame.command_buffer,
            mip_index,
            source_extent,
            target_extent,
            quality.bloom(),
            swapchain.taa_history_write_index(),
        )?;
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

/// Records one additive bloom upsample pass from a smaller mip into the next larger mip.
fn record_bloom_upsample_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    target_mip_index: usize,
    quality: RenderQualitySettings,
) -> Result<(), VulkanError> {
    let source_extent = swapchain.bloom_extent_2d(target_mip_index + 1)?;
    let target_extent = swapchain.bloom_extent_2d(target_mip_index)?;
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(target_extent);
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.bloom_upsample_render_pass())
        .framebuffer(swapchain.bloom_upsample_framebuffer(target_mip_index)?)
        .render_area(render_area);

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.bloom_pipeline().draw_upsample(
            device,
            frame.command_buffer,
            target_mip_index,
            source_extent,
            target_extent,
            quality.bloom(),
        )?;
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

fn god_ray_push_constants(
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> GodRayPushConstants {
    let extent = swapchain.extent_2d();
    let jitter_pixels = swapchain.taa_jitter_pixels();
    let jitter_ndc = [
        jitter_pixels[0] * 2.0 / extent.width.max(1) as f32,
        jitter_pixels[1] * 2.0 / extent.height.max(1) as f32,
    ];
    GodRayPushConstants::new(
        active_camera(snapshot),
        swapchain.god_ray_extent_2d(),
        state.quality,
        frame_global_light(snapshot),
        state.emissive_lights,
        state.features.has_transparent_scene_items,
        swapchain.god_ray_history_valid(),
        snapshot.frame_id.raw(),
        jitter_ndc,
    )
}

fn god_ray_render_pass_info(
    swapchain: &VulkanSwapchain,
    framebuffer: vk::Framebuffer,
) -> vk::RenderPassBeginInfo<'_> {
    let render_area = vk::Rect2D::default()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain.god_ray_extent_2d());
    vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.god_ray_render_pass())
        .framebuffer(framebuffer)
        .render_area(render_area)
}

fn record_god_ray_mask_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    meshes: &VulkanMeshStore,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let push = god_ray_push_constants(swapchain, snapshot, state);
    let render_pass_info =
        god_ray_render_pass_info(swapchain, swapchain.god_ray_mask_framebuffer());

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.god_rays_pipeline().draw_mask(
            device,
            frame.command_buffer,
            swapchain.god_ray_extent_2d(),
            meshes.frame_descriptor_set(frame.slot_index)?,
            push,
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

fn record_god_ray_prefilter_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let push = god_ray_push_constants(swapchain, snapshot, state);
    let render_pass_info =
        god_ray_render_pass_info(swapchain, swapchain.god_ray_prefilter_framebuffer());

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.god_rays_pipeline().draw_prefilter(
            device,
            frame.command_buffer,
            swapchain.god_ray_extent_2d(),
            push,
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

fn record_god_ray_radial_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let push = god_ray_push_constants(swapchain, snapshot, state);
    let render_pass_info =
        god_ray_render_pass_info(swapchain, swapchain.god_ray_radial_framebuffer());

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.god_rays_pipeline().draw_radial(
            device,
            frame.command_buffer,
            swapchain.god_ray_extent_2d(),
            push,
        );
        device.cmd_end_render_pass(frame.command_buffer);
    }

    Ok(())
}

fn record_god_ray_temporal_pass(
    device: &Device,
    frame: ActiveFrame,
    swapchain: &VulkanSwapchain,
    snapshot: &FrameSnapshot,
    state: &FrameRecordState,
) -> Result<(), VulkanError> {
    let write_history_index = swapchain.god_ray_history_write_index();
    let push = god_ray_push_constants(swapchain, snapshot, state);
    let render_pass_info = god_ray_render_pass_info(
        swapchain,
        swapchain.god_ray_history_framebuffer(write_history_index)?,
    );

    unsafe {
        device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
        swapchain.god_rays_pipeline().draw_temporal(
            device,
            frame.command_buffer,
            swapchain.god_ray_extent_2d(),
            write_history_index,
            push,
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
            swapchain.taa_history_write_index(),
            swapchain.god_ray_history_write_index(),
            frame_global_light(snapshot),
            state.emissive_lights,
            snapshot.debug_draw.view_mode,
            swapchain.taa_jitter_pixels(),
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
    meshes: &VulkanMeshStore,
    pipeline: MeshPipelineSet,
    pass_resources: Option<&MeshPassResources>,
    items: &[ShadowDrawItem<'_>],
    extent: vk::Extent2D,
    cascade_index: usize,
) -> Result<usize, VulkanError> {
    let mut drawn = 0;
    let mut draw_state = MeshDrawState::default();
    for draw in items {
        let draw_options = MeshDrawOptions::shadow_preculled(extent, cascade_index, draw.lod);
        if meshes.bind_and_draw(
            device,
            frame.command_buffer,
            pipeline,
            pass_resources,
            frame.slot_index,
            draw.item,
            draw.material_descriptor_set,
            draw_options,
            draw.pipeline_key,
            &mut draw_state,
        )? {
            drawn += 1;
        }
    }

    Ok(drawn)
}

/// Builds both cascade-local shadow draw lists before Vulkan command recording.
///
/// Opaque and translucent casters are still submitted to their original independent passes.  The
/// shared walk only avoids repeating the bounds projection and directional LOD query for the same
/// resolved item; the opaque list keeps its original state sort and the translucent list keeps its
/// original extraction order.
fn shadow_draw_items_for_cascade<'a>(
    items: &[ResolvedDrawItem<'a>],
    meshes: &VulkanMeshStore,
    cascade_index: usize,
    texel_world: f32,
    shadow_resolution: u32,
    shadow_cull: ShadowCascadeCull,
    include_opaque: bool,
    include_translucent: bool,
) -> (Vec<ShadowDrawItem<'a>>, Vec<ShadowDrawItem<'a>>) {
    let mut opaque = Vec::new();
    let mut translucent = Vec::new();
    // Many render items are instances of the same mesh. LOD selection only depends on the mesh and
    // cascade metrics, so memoize it for this one cascade while retaining the existing per-frame
    // list lifetime (no persistent shadow-map cache is introduced).
    let mut lod_by_mesh = HashMap::new();

    for resolved in items {
        let item = resolved.item;
        let material = resolved.material;
        let accepts_opaque = include_opaque
            && shadow_filter_accepts(MeshDrawFilter::OpaqueShadowCasters, material, item);
        let accepts_translucent = include_translucent
            && shadow_filter_accepts(MeshDrawFilter::TranslucentShadowCasters, material, item);
        if !(accepts_opaque || accepts_translucent) || !shadow_cull.contains_bounds(resolved.bounds)
        {
            continue;
        }

        // A material can only be in one alpha class today, but retain both branches so this
        // helper remains correct if a future material intentionally writes both shadow targets.
        let lod = *lod_by_mesh.entry(item.mesh.raw()).or_insert_with(|| {
            meshes.directional_shadow_lod(item.mesh, cascade_index, texel_world, shadow_resolution)
        });
        if accepts_opaque {
            let pipeline_key = MeshPipelineKey {
                uses_textures: material.uses_shadow_alpha_texture,
                double_sided: material.double_sided,
                opaque_scene: false,
                opaque_shadow: !material.uses_shadow_alpha_test,
            };
            opaque.push((
                (
                    pipeline_key.opaque_shadow,
                    pipeline_key.uses_textures,
                    pipeline_key.double_sided,
                    item.material.raw(),
                    item.mesh.raw(),
                ),
                ShadowDrawItem {
                    item,
                    pipeline_key,
                    material_descriptor_set: material.descriptor_set,
                    lod,
                },
            ));
        }
        if accepts_translucent {
            translucent.push(ShadowDrawItem {
                item,
                pipeline_key: MeshPipelineKey {
                    uses_textures: material.uses_base_color_texture,
                    double_sided: material.double_sided,
                    opaque_scene: false,
                    opaque_shadow: false,
                },
                material_descriptor_set: material.descriptor_set,
                lod,
            });
        }
    }

    opaque.sort_unstable_by_key(|(key, _)| *key);
    (
        opaque.into_iter().map(|(_, draw)| draw).collect(),
        translucent,
    )
}

/// Builds local-light shadow draw lists for every enabled cubemap light.
fn local_shadow_draw_items<'a>(
    items: &[ResolvedDrawItem<'a>],
    local_shadow: Option<LocalShadowFrameData>,
) -> [[Vec<ShadowDrawItem<'a>>; LOCAL_SHADOW_FACE_COUNT]; MAX_LOCAL_LIGHTS] {
    let Some(local_shadow) = local_shadow else {
        return std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    };

    std::array::from_fn(|light_index| {
        let Some(local_light) = local_shadow.lights[light_index] else {
            return std::array::from_fn(|_| Vec::new());
        };

        local_shadow_draw_items_for_light(items, local_light)
    })
}

fn local_shadow_draw_items_for_light<'a>(
    items: &[ResolvedDrawItem<'a>],
    local_shadow: LocalShadowLightData,
) -> [Vec<ShadowDrawItem<'a>>; LOCAL_SHADOW_FACE_COUNT] {
    let light = local_shadow.light_position_radius;
    let light_position = [light[0], light[1], light[2]];
    let range = light[3].max(0.0);
    let source_radius = local_shadow.source_radius.max(0.0);
    if range <= 0.0 {
        return std::array::from_fn(|_| Vec::new());
    }

    let mut face_draws: [Vec<ShadowDrawItem<'a>>; LOCAL_SHADOW_FACE_COUNT] =
        std::array::from_fn(|_| Vec::new());
    let face_culls = local_shadow_face_culls(local_shadow);
    for resolved in items {
        let item = resolved.item;
        let material = resolved.material;
        if !shadow_filter_accepts(MeshDrawFilter::OpaqueShadowCasters, material, item) {
            continue;
        }
        let Some(bounds) = resolved.bounds else {
            continue;
        };
        let center = bounds.center();
        let radius = bounds.radius();
        if local_shadow_self_occluder(center, radius, light_position, source_radius) {
            continue;
        }
        let dx = center[0] - light_position[0];
        let dy = center[1] - light_position[1];
        let dz = center[2] - light_position[2];
        let influence = range + radius;
        if dx * dx + dy * dy + dz * dz > influence * influence {
            continue;
        }

        let pipeline_key = MeshPipelineKey {
            uses_textures: material.uses_shadow_alpha_texture,
            double_sided: material.double_sided,
            opaque_scene: false,
            opaque_shadow: !material.uses_shadow_alpha_test,
        };
        for (face_index, draws) in face_draws.iter_mut().enumerate() {
            if local_shadow_face_contains_bounds_cached(face_culls[face_index], center, radius) {
                draws.push(ShadowDrawItem {
                    item,
                    pipeline_key,
                    material_descriptor_set: material.descriptor_set,
                    lod: MeshLodLevel::Medium,
                });
            }
        }
    }

    for draws in &mut face_draws {
        draws.sort_unstable_by_key(|draw| {
            (
                draw.pipeline_key.opaque_shadow,
                draw.pipeline_key.uses_textures,
                draw.pipeline_key.double_sided,
                draw.item.material.raw(),
                draw.item.mesh.raw(),
            )
        });
    }
    face_draws
}

fn local_shadow_self_occluder(
    center: [f32; 3],
    radius: f32,
    light_position: [f32; 3],
    source_radius: f32,
) -> bool {
    let delta = sub3(center, light_position);
    let skip_radius = source_radius.max(0.08).min(radius.max(0.08)) * 0.2;
    dot3(delta, delta) <= skip_radius * skip_radius
}

/// Returns a coarse front-to-back bucket for opaque scene draws.
///
/// The bucket is deliberately coarse so each bucket can still group by pipeline/material. That
/// keeps early depth rejection useful without throwing away the state-bind wins from sorting.
fn opaque_scene_depth_bucket(
    bounds: Option<SceneBounds>,
    camera: CameraSnapshot,
    camera_forward: [f32; 3],
) -> u16 {
    let Some(bounds) = bounds else {
        return u16::MAX;
    };

    let to_center = sub3(bounds.center(), camera.eye);
    let near = camera.near.max(0.001);
    let far = camera.far.max(near + 0.001);
    let depth = (dot3(to_center, camera_forward) - bounds.radius()).max(near);
    if !depth.is_finite() {
        return u16::MAX;
    }

    let far_log = (far / near).log2().max(1.0);
    let depth_log = (depth / near).log2().clamp(0.0, far_log);
    ((depth_log / far_log) * 31.0) as u16
}

/// Returns whether one render item belongs in the selected shadow caster pass.
fn shadow_filter_accepts(
    filter: MeshDrawFilter,
    material: MaterialDrawInfo,
    item: &crate::protocol::RenderItemPacket,
) -> bool {
    item.flags.visible
        && match filter {
            MeshDrawFilter::OpaqueShadowCasters => {
                item.flags.casts_shadow && material.casts_opaque_shadow
            }
            MeshDrawFilter::TranslucentShadowCasters => {
                item.flags.casts_shadow && material.casts_translucent_shadow
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

fn frame_global_light(snapshot: &FrameSnapshot) -> LightPacket {
    snapshot
        .lights
        .first()
        .copied()
        .unwrap_or_else(|| LightPacket::new(1.0))
}

#[derive(Clone, Copy)]
struct LocalLightCandidate {
    score: f32,
    position_radius: [f32; 4],
    color: [f32; 4],
    direction_radius: [f32; 4],
    size_kind: [f32; 4],
}

/// Packs at most four explicit scene lights without deriving extra lights from mesh materials.
///
/// Emissive glTF materials illuminate their own surface; treating every emissive mesh as a
/// nearby light changes asset semantics and costs a local-light branch for every fragment.
fn local_light_uniforms(snapshot: &FrameSnapshot, camera: CameraSnapshot) -> EmissiveLightUniforms {
    const MIN_LOCAL_LIGHT_BRIGHTNESS: f32 = 0.02;

    let mut candidates = Vec::with_capacity(MAX_LOCAL_LIGHTS);
    for light in &snapshot.local_lights {
        let candidate = local_light_candidate(light, camera);
        if candidate.color[3] <= MIN_LOCAL_LIGHT_BRIGHTNESS {
            continue;
        }
        if candidates.len() < MAX_LOCAL_LIGHTS {
            candidates.push(candidate);
            continue;
        }
        let least_important = candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
            .map(|(index, _)| index)
            .expect("full local-light candidate set has an entry");
        if candidate
            .score
            .total_cmp(&candidates[least_important].score)
            .is_gt()
        {
            candidates[least_important] = candidate;
        }
    }

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut uniforms = EmissiveLightUniforms::disabled();
    for (index, candidate) in candidates.iter().take(MAX_LOCAL_LIGHTS).enumerate() {
        uniforms.position_radius[index] = candidate.position_radius;
        uniforms.color[index] = candidate.color;
        uniforms.direction_radius[index] = candidate.direction_radius;
        uniforms.size_kind[index] = candidate.size_kind;
    }
    uniforms.count[0] = candidates.len().min(MAX_LOCAL_LIGHTS) as f32;

    uniforms
}

fn local_light_candidate(light: &LocalLightPacket, camera: CameraSnapshot) -> LocalLightCandidate {
    let brightness = max3(light.color) * light.intensity;
    let camera_delta = sub3(light.position, camera.eye);
    let camera_distance_sq = dot3(camera_delta, camera_delta).max(1.0);
    let score = brightness * light.range * light.range / (1.0 + camera_distance_sq * 0.0025);
    let kind = light.kind.shader_code();
    let casts_shadow = if light.casts_shadow { 1.0 } else { 0.0 };
    let source_radius = light.source_radius.max(0.0).min(light.range);

    LocalLightCandidate {
        score,
        position_radius: [
            light.position[0],
            light.position[1],
            light.position[2],
            light.range,
        ],
        color: [
            light.color[0] * light.intensity,
            light.color[1] * light.intensity,
            light.color[2] * light.intensity,
            brightness,
        ],
        direction_radius: [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            source_radius,
        ],
        size_kind: [light.half_size[0], light.half_size[1], kind, casts_shadow],
    }
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
        ResourceState::ShaderRead => {
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        }
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
        ResourceState::ShaderRead => {
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        }
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
