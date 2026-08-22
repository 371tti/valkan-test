use std::{ffi::CStr, mem::size_of};

use ash::{Device, vk};

use crate::{
    protocol::{CameraSnapshot, FrameSnapshot, NonZeroExtent, RenderQualitySettings},
    renderer::{
        graph::{
            FrameGraphPlan, GraphResource, ResourceState, TAA_DEPTH_HISTORY_RESOURCES,
            TAA_HISTORY_COUNT, TAA_HISTORY_RESOURCES, TAA_MOTION_RESOURCE,
            TAA_NORMAL_HISTORY_RESOURCES, TAA_RESOLVE_PASS,
        },
        pipeline::shader_interface,
    },
};

use super::{
    VulkanError,
    buffer::{GpuBuffer, create_host_buffer, write_buffer_value},
    shader::{self, assets},
    swapchain_target::{ColorTarget, create_color_target, destroy_color_target},
};

pub(super) const TAA_HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub(super) const TAA_MOTION_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
pub(super) const TAA_LINEAR_DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub(super) const TAA_NORMAL_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

const HISTORY_SAMPLE_LIMIT: f32 = 16.0;
const DEPTH_REJECT_ABSOLUTE: f32 = 0.01;
const DEPTH_REJECT_RELATIVE: f32 = 0.005;
const NORMAL_REJECT_COSINE: f32 = 0.90;
const JITTER_SAMPLE_COUNT: u64 = 16;
const JITTER_SEQUENCE_CENTROID: [f32; 2] = [-0.029_296_875, -0.037_037_037];
const CURRENT_COLOR_INPUT_COUNT: usize = 2;
const SCENE_COLOR_INPUT_INDEX: usize = 0;
const CORRECTED_SCENE_COLOR_INPUT_INDEX: usize = 1;

const CURRENT_COLOR_BINDING: u32 = 0;
const CURRENT_DEPTH_BINDING: u32 = 1;
const CURRENT_NORMAL_BINDING: u32 = 2;
const PREVIOUS_COLOR_BINDING: u32 = 3;
const PREVIOUS_DEPTH_BINDING: u32 = 4;
const PREVIOUS_NORMAL_BINDING: u32 = 5;
const PARAMS_BINDING: u32 = 6;
const CURRENT_TRANSPARENT_NORMAL_BINDING: u32 = 7;

const VERTEX_SHADER: &[u8] = assets::POST_VERT;
const FRAGMENT_SHADER: &[u8] = assets::POST_TAA_RESOLVE_FRAG;
const SHADER_ENTRY: &CStr = shader::ENTRY;

#[repr(C)]
#[derive(Clone, Copy)]
struct TemporalResolveUniform {
    current_view_projection: [f32; 16],
    inverse_current_view_projection: [f32; 16],
    previous_view_projection: [f32; 16],
    inverse_current_view: [f32; 16],
    inverse_previous_view: [f32; 16],
    texel_feedback_reset: [f32; 4],
    rejection: [f32; 4],
    jitter_pixels: [f32; 4],
}

struct TemporalHistoryTarget {
    color: ColorTarget,
    linear_depth: ColorTarget,
    normal: ColorTarget,
    color_state: ResourceState,
    depth_state: ResourceState,
    normal_state: ResourceState,
}

struct TemporalMotionTarget {
    color: ColorTarget,
    state: ResourceState,
}

#[derive(Clone, Copy)]
struct PreviousFrame {
    frame_id: u64,
    scene: u64,
    surface_generation: u64,
    camera: CameraSnapshot,
    aspect: f32,
    view_projection: [f32; 16],
    jitter_pixels: [f32; 2],
    aa_blend: f32,
}

#[derive(Clone, Copy)]
struct PendingFrame {
    previous: PreviousFrame,
    slot_index: usize,
    write_history_index: usize,
    current_color_input_index: usize,
}

#[derive(Clone, Copy)]
pub(super) struct TaaFrameInfo {
    pub(super) jittered_view_projection: [f32; 16],
    pub(super) inverse_current_view_projection: [f32; 16],
    pub(super) previous_view_projection: [f32; 16],
    pub(super) inverse_current_view: [f32; 16],
    pub(super) inverse_previous_view: [f32; 16],
    pub(super) current_jitter_pixels: [f32; 2],
    pub(super) previous_jitter_pixels: [f32; 2],
    pub(super) write_history_index: usize,
    /// Resets only consumers of the shared camera/depth/normal history (camera cut, scene change,
    /// resize, or stale frame). This deliberately ignores whether color TAA itself is enabled.
    pub(super) reset_reprojection_history: bool,
}

/// Owns the HDR TAA resolve plus the motion/depth/normal history shared by temporal effects.
pub(super) struct TemporalAntiAliasing {
    histories: Vec<TemporalHistoryTarget>,
    motion: TemporalMotionTarget,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
    uniform_buffers: Vec<GpuBuffer>,
    history_write_index: usize,
    history_valid: bool,
    frame_index: u64,
    previous: Option<PreviousFrame>,
    pending: Option<PendingFrame>,
}

struct TemporalBuild<'a> {
    device: &'a Device,
    histories: Vec<TemporalHistoryTarget>,
    motion: Option<TemporalMotionTarget>,
    render_pass: Option<vk::RenderPass>,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    color_sampler: Option<vk::Sampler>,
    data_sampler: Option<vk::Sampler>,
    uniform_buffers: Vec<GpuBuffer>,
    finished: bool,
}

impl<'a> TemporalBuild<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            histories: Vec::new(),
            motion: None,
            render_pass: None,
            framebuffers: Vec::new(),
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            descriptor_sets: Vec::new(),
            color_sampler: None,
            data_sampler: None,
            uniform_buffers: Vec::new(),
            finished: false,
        }
    }

    fn finish(mut self) -> TemporalAntiAliasing {
        let taa = TemporalAntiAliasing {
            histories: std::mem::take(&mut self.histories),
            motion: self
                .motion
                .take()
                .expect("TAA motion target was not created"),
            render_pass: self
                .render_pass
                .take()
                .expect("TAA render pass was not created"),
            framebuffers: std::mem::take(&mut self.framebuffers),
            pipeline: self.pipeline.take().expect("TAA pipeline was not created"),
            pipeline_layout: self
                .pipeline_layout
                .take()
                .expect("TAA pipeline layout was not created"),
            descriptor_set_layout: self
                .descriptor_set_layout
                .take()
                .expect("TAA descriptor set layout was not created"),
            descriptor_pool: self
                .descriptor_pool
                .take()
                .expect("TAA descriptor pool was not created"),
            descriptor_sets: std::mem::take(&mut self.descriptor_sets),
            color_sampler: self
                .color_sampler
                .take()
                .expect("TAA color sampler was not created"),
            data_sampler: self
                .data_sampler
                .take()
                .expect("TAA data sampler was not created"),
            uniform_buffers: std::mem::take(&mut self.uniform_buffers),
            history_write_index: 0,
            history_valid: false,
            frame_index: 0,
            previous: None,
            pending: None,
        };
        self.finished = true;
        taa
    }
}

impl Drop for TemporalBuild<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Some(pipeline) = self.pipeline.take() {
            destroy_pipeline(self.device, pipeline);
        }
        if let Some(pool) = self.descriptor_pool.take() {
            destroy_descriptor_pool(self.device, pool);
        }
        if let Some(sampler) = self.data_sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(sampler) = self.color_sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(layout) = self.pipeline_layout.take() {
            destroy_pipeline_layout(self.device, layout);
        }
        if let Some(layout) = self.descriptor_set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
        for framebuffer in self.framebuffers.drain(..) {
            destroy_framebuffer(self.device, framebuffer);
        }
        if let Some(render_pass) = self.render_pass.take() {
            destroy_render_pass(self.device, render_pass);
        }
        for buffer in self.uniform_buffers.drain(..) {
            buffer.destroy(self.device);
        }
        if let Some(motion) = self.motion.take() {
            destroy_color_target(self.device, motion.color);
        }
        for history in self.histories.drain(..) {
            destroy_temporal_history(self.device, history);
        }
    }
}

impl TemporalAntiAliasing {
    pub(super) fn create(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        extent: NonZeroExtent,
        frame_slot_count: usize,
        current_color_view: vk::ImageView,
        current_depth_view: vk::ImageView,
        current_normal_view: vk::ImageView,
        current_transparent_normal_view: vk::ImageView,
    ) -> Result<Self, VulkanError> {
        let mut build = TemporalBuild::new(device);
        for _ in 0..TAA_HISTORY_COUNT {
            build
                .histories
                .push(create_temporal_history(device, memory_properties, extent)?);
        }
        build.motion = Some(TemporalMotionTarget {
            color: create_color_target(
                device,
                memory_properties,
                extent,
                TAA_MOTION_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )?,
            state: ResourceState::Undefined,
        });
        build.render_pass = Some(create_render_pass(device)?);
        let render_pass = build.render_pass.expect("TAA render pass was just created");
        let motion_view = build
            .motion
            .as_ref()
            .expect("TAA motion target was just created")
            .color
            .view;
        for history in &build.histories {
            build.framebuffers.push(create_framebuffer(
                device,
                render_pass,
                extent,
                history,
                motion_view,
            )?);
        }

        build.descriptor_set_layout = Some(create_descriptor_set_layout(device)?);
        let descriptor_set_layout = build
            .descriptor_set_layout
            .expect("TAA descriptor layout was just created");
        build.pipeline_layout = Some(create_pipeline_layout(device, descriptor_set_layout)?);
        build.color_sampler = Some(create_sampler(device, vk::Filter::LINEAR)?);
        build.data_sampler = Some(create_sampler(device, vk::Filter::NEAREST)?);

        let frame_slot_count = frame_slot_count.max(1);
        for _ in 0..frame_slot_count {
            build.uniform_buffers.push(create_host_buffer(
                device,
                memory_properties,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                size_of::<TemporalResolveUniform>() as vk::DeviceSize,
            )?);
        }
        let descriptor_set_count = CURRENT_COLOR_INPUT_COUNT * TAA_HISTORY_COUNT * frame_slot_count;
        build.descriptor_pool = Some(create_descriptor_pool(device, descriptor_set_count as u32)?);
        build.descriptor_sets = allocate_descriptor_sets(
            device,
            build
                .descriptor_pool
                .expect("TAA descriptor pool was just created"),
            descriptor_set_layout,
            descriptor_set_count,
        )?;
        update_descriptor_sets(
            device,
            &build.descriptor_sets,
            &build.uniform_buffers,
            &build.histories,
            frame_slot_count,
            current_color_view,
            current_depth_view,
            current_normal_view,
            current_transparent_normal_view,
            build
                .color_sampler
                .expect("TAA color sampler was just created"),
            build
                .data_sampler
                .expect("TAA data sampler was just created"),
        );
        build.pipeline = Some(create_pipeline(
            device,
            build
                .pipeline_layout
                .expect("TAA pipeline layout was just created"),
            render_pass,
        )?);

        tracing::info!(
            width = extent.width(),
            height = extent.height(),
            frame_slot_count,
            history_format = ?TAA_HISTORY_FORMAT,
            motion_format = ?TAA_MOTION_FORMAT,
            "created HDR temporal anti-aliasing resources"
        );
        Ok(build.finish())
    }

    pub(super) fn prepare_frame(
        &mut self,
        device: &Device,
        slot_index: usize,
        snapshot: &FrameSnapshot,
        camera: CameraSnapshot,
        quality: RenderQualitySettings,
        extent: vk::Extent2D,
        use_corrected_scene_color: bool,
    ) -> Result<TaaFrameInfo, VulkanError> {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let aa_blend = quality.anti_aliasing().blend();
        let temporal_enabled = aa_blend > 0.0;
        let jitter_pixels = if temporal_enabled {
            halton_jitter(self.frame_index % JITTER_SAMPLE_COUNT)
        } else {
            [0.0, 0.0]
        };
        let current_view_projection =
            jitter_view_projection(camera.view_projection(aspect), jitter_pixels, extent);
        let inverse_current_view_projection =
            invert_mat4(current_view_projection).unwrap_or_else(identity_mat4);
        let reset_reprojection_history = !self.history_valid
            || self.previous.is_none_or(|previous| {
                temporal_reprojection_history_discontinuous(previous, snapshot, camera, aspect)
            });
        let reset_history = reset_reprojection_history
            || self.previous.is_none_or(|previous| {
                (aa_blend - previous.aa_blend).abs() > 0.0001
                    || (previous.aa_blend > 0.0) != temporal_enabled
            });
        let previous_view_projection = self
            .previous
            .map_or(current_view_projection, |previous| previous.view_projection);
        let previous_camera = self.previous.map_or(camera, |previous| previous.camera);
        let inverse_current_view =
            invert_mat4(camera_view_matrix(camera)).unwrap_or_else(identity_mat4);
        let inverse_previous_view =
            invert_mat4(camera_view_matrix(previous_camera)).unwrap_or_else(identity_mat4);
        let previous_jitter = self
            .previous
            .map_or(jitter_pixels, |previous| previous.jitter_pixels);
        let feedback = if temporal_enabled {
            0.80 + 0.16 * aa_blend.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let uniform = TemporalResolveUniform {
            current_view_projection,
            inverse_current_view_projection,
            previous_view_projection,
            inverse_current_view,
            inverse_previous_view,
            texel_feedback_reset: [
                1.0 / extent.width.max(1) as f32,
                1.0 / extent.height.max(1) as f32,
                feedback,
                if reset_history || !temporal_enabled {
                    1.0
                } else {
                    0.0
                },
            ],
            rejection: [
                DEPTH_REJECT_ABSOLUTE,
                DEPTH_REJECT_RELATIVE,
                NORMAL_REJECT_COSINE,
                HISTORY_SAMPLE_LIMIT,
            ],
            jitter_pixels: [
                jitter_pixels[0],
                jitter_pixels[1],
                previous_jitter[0],
                previous_jitter[1],
            ],
        };
        let uniform_buffer = self.uniform_buffers.get(slot_index).ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: slot_index,
                count: self.uniform_buffers.len(),
            },
        )?;
        write_buffer_value(device, uniform_buffer, &uniform)?;

        let pending_previous = PreviousFrame {
            frame_id: snapshot.frame_id.raw(),
            scene: snapshot.scene.raw(),
            surface_generation: snapshot.surface_generation.raw(),
            camera,
            aspect,
            view_projection: current_view_projection,
            jitter_pixels,
            aa_blend,
        };
        self.pending = Some(PendingFrame {
            previous: pending_previous,
            slot_index,
            write_history_index: self.history_write_index,
            current_color_input_index: if use_corrected_scene_color {
                CORRECTED_SCENE_COLOR_INPUT_INDEX
            } else {
                SCENE_COLOR_INPUT_INDEX
            },
        });

        Ok(TaaFrameInfo {
            jittered_view_projection: current_view_projection,
            inverse_current_view_projection,
            previous_view_projection,
            inverse_current_view,
            inverse_previous_view,
            current_jitter_pixels: jitter_pixels,
            previous_jitter_pixels: previous_jitter,
            write_history_index: self.history_write_index,
            reset_reprojection_history,
        })
    }

    pub(super) fn record(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
    ) -> Result<(), VulkanError> {
        let pending = self.pending.ok_or_else(|| {
            VulkanError::GraphCompile("TAA pass recorded before frame preparation".to_string())
        })?;
        let framebuffer = self
            .framebuffers
            .get(pending.write_history_index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: pending.write_history_index,
                count: self.framebuffers.len(),
            })?;
        let descriptor_index = taa_descriptor_index(
            pending.current_color_input_index,
            pending.write_history_index,
            pending.slot_index,
            self.uniform_buffers.len(),
        );
        let descriptor_set = self.descriptor_sets.get(descriptor_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: descriptor_index,
                count: self.descriptor_sets.len(),
            },
        )?;
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area);
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = [render_area];
        let descriptor_sets = [descriptor_set];

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                shader_interface::FRAME_SET,
                &descriptor_sets,
                &[],
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
        }
        Ok(())
    }

    /// Binds the corrected-HDR descriptor variants after the shadow resolver creates its target.
    /// The SceneColor variants remain intact for frame graphs that omit scene metadata.
    pub(super) fn update_current_color_input(
        &self,
        device: &Device,
        current_color_view: vk::ImageView,
    ) {
        update_current_color_descriptor_bindings(
            device,
            &self.descriptor_sets,
            self.uniform_buffers.len(),
            current_color_view,
            self.color_sampler,
        );
    }

    pub(super) fn history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        std::array::from_fn(|index| self.histories[index].color.view)
    }

    pub(super) fn depth_history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        std::array::from_fn(|index| self.histories[index].linear_depth.view)
    }

    pub(super) fn normal_history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        std::array::from_fn(|index| self.histories[index].normal.view)
    }

    pub(super) fn history_write_index(&self) -> usize {
        self.history_write_index
    }

    /// Returns the jitter used by the frame currently being recorded. Post effects reconstructing
    /// positions from the jittered scene depth must use this exact sample offset.
    pub(super) fn pending_jitter_pixels(&self) -> [f32; 2] {
        self.pending
            .map(|pending| pending.previous.jitter_pixels)
            .unwrap_or([0.0, 0.0])
    }

    pub(super) fn graph_states(
        &self,
    ) -> (
        [ResourceState; TAA_HISTORY_COUNT],
        [ResourceState; TAA_HISTORY_COUNT],
        [ResourceState; TAA_HISTORY_COUNT],
        ResourceState,
    ) {
        (
            std::array::from_fn(|index| self.histories[index].color_state),
            std::array::from_fn(|index| self.histories[index].depth_state),
            std::array::from_fn(|index| self.histories[index].normal_state),
            self.motion.state,
        )
    }

    pub(super) fn apply_graph_final_states(&mut self, plan: &FrameGraphPlan) {
        for (index, history) in self.histories.iter_mut().enumerate() {
            if let Some(state) = plan.final_state_for(TAA_HISTORY_RESOURCES[index]) {
                history.color_state = state;
            }
            if let Some(state) = plan.final_state_for(TAA_DEPTH_HISTORY_RESOURCES[index]) {
                history.depth_state = state;
            }
            if let Some(state) = plan.final_state_for(TAA_NORMAL_HISTORY_RESOURCES[index]) {
                history.normal_state = state;
            }
        }
        if let Some(state) = plan.final_state_for(TAA_MOTION_RESOURCE) {
            self.motion.state = state;
        }
        if plan
            .passes()
            .iter()
            .any(|pass| pass.name() == TAA_RESOLVE_PASS)
        {
            if let Some(pending) = self.pending.take() {
                self.previous = Some(pending.previous);
                self.history_valid = true;
                self.history_write_index = 1 - pending.write_history_index;
                self.frame_index = self.frame_index.saturating_add(1);
            }
        } else {
            self.pending = None;
            self.history_valid = false;
        }
    }

    pub(super) fn graph_image(
        &self,
        resource: GraphResource,
    ) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        let image = if let Some(index) = resource.taa_history() {
            self.histories.get(index).map(|target| target.color.image)
        } else if let Some(index) = resource.taa_depth_history() {
            self.histories
                .get(index)
                .map(|target| target.linear_depth.image)
        } else if let Some(index) = resource.taa_normal_history() {
            self.histories.get(index).map(|target| target.normal.image)
        } else if resource == TAA_MOTION_RESOURCE {
            Some(self.motion.color.image)
        } else {
            None
        }?;
        Some((image, vk::ImageAspectFlags::COLOR))
    }

    pub(super) fn destroy(self, device: &Device) {
        destroy_pipeline(device, self.pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.data_sampler);
        destroy_sampler(device, self.color_sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.descriptor_set_layout);
        for framebuffer in self.framebuffers {
            destroy_framebuffer(device, framebuffer);
        }
        destroy_render_pass(device, self.render_pass);
        for buffer in self.uniform_buffers {
            buffer.destroy(device);
        }
        destroy_color_target(device, self.motion.color);
        for history in self.histories {
            destroy_temporal_history(device, history);
        }
    }
}

fn create_temporal_history(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
) -> Result<TemporalHistoryTarget, VulkanError> {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let color = create_color_target(device, memory_properties, extent, TAA_HISTORY_FORMAT, usage)?;
    let linear_depth = match create_color_target(
        device,
        memory_properties,
        extent,
        TAA_LINEAR_DEPTH_FORMAT,
        usage,
    ) {
        Ok(target) => target,
        Err(error) => {
            destroy_color_target(device, color);
            return Err(error);
        }
    };
    let normal =
        match create_color_target(device, memory_properties, extent, TAA_NORMAL_FORMAT, usage) {
            Ok(target) => target,
            Err(error) => {
                destroy_color_target(device, linear_depth);
                destroy_color_target(device, color);
                return Err(error);
            }
        };
    Ok(TemporalHistoryTarget {
        color,
        linear_depth,
        normal,
        color_state: ResourceState::Undefined,
        depth_state: ResourceState::Undefined,
        normal_state: ResourceState::Undefined,
    })
}

fn destroy_temporal_history(device: &Device, history: TemporalHistoryTarget) {
    destroy_color_target(device, history.normal);
    destroy_color_target(device, history.linear_depth);
    destroy_color_target(device, history.color);
}

fn create_render_pass(device: &Device) -> Result<vk::RenderPass, VulkanError> {
    let formats = [
        TAA_HISTORY_FORMAT,
        TAA_MOTION_FORMAT,
        TAA_LINEAR_DEPTH_FORMAT,
        TAA_NORMAL_FORMAT,
    ];
    let attachments = formats.map(|format| {
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    });
    let color_references = std::array::from_fn::<_, 4, _>(|index| {
        vk::AttachmentReference::default()
            .attachment(index as u32)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    });
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references)];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    unsafe { device.create_render_pass(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: NonZeroExtent,
    history: &TemporalHistoryTarget,
    motion_view: vk::ImageView,
) -> Result<vk::Framebuffer, VulkanError> {
    let attachments = [
        history.color.view,
        motion_view,
        history.linear_depth.view,
        history.normal.view,
    ];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let sampler_binding = |binding| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    };
    let bindings = [
        sampler_binding(CURRENT_COLOR_BINDING),
        sampler_binding(CURRENT_DEPTH_BINDING),
        sampler_binding(CURRENT_NORMAL_BINDING),
        sampler_binding(PREVIOUS_COLOR_BINDING),
        sampler_binding(PREVIOUS_DEPTH_BINDING),
        sampler_binding(PREVIOUS_NORMAL_BINDING),
        sampler_binding(CURRENT_TRANSPARENT_NORMAL_BINDING),
        vk::DescriptorSetLayoutBinding::default()
            .binding(PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [descriptor_set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_sampler(device: &Device, filter: vk::Filter) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_pool(
    device: &Device,
    set_count: u32,
) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(set_count * 7),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(set_count),
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(set_count)
        .pool_sizes(&pool_sizes);
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    let layouts = vec![layout; count];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

#[allow(clippy::too_many_arguments)]
fn update_descriptor_sets(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    uniform_buffers: &[GpuBuffer],
    histories: &[TemporalHistoryTarget],
    frame_slot_count: usize,
    current_color_view: vk::ImageView,
    current_depth_view: vk::ImageView,
    current_normal_view: vk::ImageView,
    current_transparent_normal_view: vk::ImageView,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
) {
    for current_color_input_index in 0..CURRENT_COLOR_INPUT_COUNT {
        for write_index in 0..TAA_HISTORY_COUNT {
            let read_index = 1 - write_index;
            for (slot_index, uniform_buffer) in uniform_buffers.iter().enumerate() {
                let descriptor_index = taa_descriptor_index(
                    current_color_input_index,
                    write_index,
                    slot_index,
                    frame_slot_count,
                );
                let descriptor_set = descriptor_sets[descriptor_index];
                let current_color = [image_info(color_sampler, current_color_view)];
                let current_depth = [image_info(data_sampler, current_depth_view)];
                let current_normal = [image_info(data_sampler, current_normal_view)];
                let previous_color = [image_info(color_sampler, histories[read_index].color.view)];
                let previous_depth = [image_info(
                    data_sampler,
                    histories[read_index].linear_depth.view,
                )];
                let previous_normal = [image_info(data_sampler, histories[read_index].normal.view)];
                let current_transparent_normal =
                    [image_info(data_sampler, current_transparent_normal_view)];
                let buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.handle())
                    .offset(0)
                    .range(size_of::<TemporalResolveUniform>() as vk::DeviceSize)];
                let writes = [
                    image_write(descriptor_set, CURRENT_COLOR_BINDING, &current_color),
                    image_write(descriptor_set, CURRENT_DEPTH_BINDING, &current_depth),
                    image_write(descriptor_set, CURRENT_NORMAL_BINDING, &current_normal),
                    image_write(descriptor_set, PREVIOUS_COLOR_BINDING, &previous_color),
                    image_write(descriptor_set, PREVIOUS_DEPTH_BINDING, &previous_depth),
                    image_write(descriptor_set, PREVIOUS_NORMAL_BINDING, &previous_normal),
                    image_write(
                        descriptor_set,
                        CURRENT_TRANSPARENT_NORMAL_BINDING,
                        &current_transparent_normal,
                    ),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(PARAMS_BINDING)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_info),
                ];
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }
            }
        }
    }
}

fn update_current_color_descriptor_bindings(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    frame_slot_count: usize,
    current_color_view: vk::ImageView,
    color_sampler: vk::Sampler,
) {
    for taa_write_index in 0..TAA_HISTORY_COUNT {
        for slot_index in 0..frame_slot_count {
            let descriptor_set = descriptor_sets[taa_descriptor_index(
                CORRECTED_SCENE_COLOR_INPUT_INDEX,
                taa_write_index,
                slot_index,
                frame_slot_count,
            )];
            let current_color = [image_info(color_sampler, current_color_view)];
            let writes = [image_write(
                descriptor_set,
                CURRENT_COLOR_BINDING,
                &current_color,
            )];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }
}

fn taa_descriptor_index(
    current_color_input_index: usize,
    taa_write_index: usize,
    slot_index: usize,
    frame_slot_count: usize,
) -> usize {
    ((current_color_input_index * TAA_HISTORY_COUNT + taa_write_index) * frame_slot_count)
        + slot_index
}

fn image_info(sampler: vk::Sampler, view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

fn image_write<'a>(
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    image_info: &'a [vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(image_info)
}

fn create_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline, VulkanError> {
    let vertex_shader = shader::create_shader_module(device, VERTEX_SHADER)?;
    let fragment_shader = match shader::create_shader_module(device, FRAGMENT_SHADER) {
        Ok(module) => module,
        Err(error) => {
            shader::destroy_shader_module(device, vertex_shader);
            return Err(error);
        }
    };
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(SHADER_ENTRY),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(SHADER_ENTRY),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_write = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
        vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    );
    let color_attachments = [color_write; 4];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_infos = [pipeline_info];
    let result = match unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
    } {
        Ok(mut pipelines) => Ok(pipelines.remove(0)),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                destroy_pipeline(device, pipeline);
            }
            Err(VulkanError::Vk(error))
        }
    };
    shader::destroy_shader_module(device, fragment_shader);
    shader::destroy_shader_module(device, vertex_shader);
    result
}

fn temporal_reprojection_history_discontinuous(
    previous: PreviousFrame,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
    aspect: f32,
) -> bool {
    // A skipped application frame is still valid: `previous` is the last frame actually rendered,
    // and its exact matrices remain the right reprojection source. Only stale/non-monotonic input
    // invalidates that relationship.
    if snapshot.frame_id.raw() <= previous.frame_id
        || snapshot.scene.raw() != previous.scene
        || snapshot.surface_generation.raw() != previous.surface_generation
        || (aspect - previous.aspect).abs() > 0.0005
    {
        return true;
    }

    let eye_delta = distance3(camera.eye, previous.camera.eye);
    let view_distance = distance3(previous.camera.eye, previous.camera.target).max(1.0);
    let forward = normalize3(sub3(camera.target, camera.eye));
    let previous_forward = normalize3(sub3(previous.camera.target, previous.camera.eye));
    let direction_cosine = dot3(forward, previous_forward);
    let fov_delta = (camera.fov_y_radians - previous.camera.fov_y_radians).abs();
    let near_delta = (camera.near - previous.camera.near).abs() / previous.camera.near.max(0.0001);
    let far_delta = (camera.far - previous.camera.far).abs() / previous.camera.far.max(0.0001);

    eye_delta > (view_distance * 2.0).max(5.0)
        || direction_cosine < 35.0_f32.to_radians().cos()
        || fov_delta > 8.0_f32.to_radians()
        || near_delta > 0.10
        || far_delta > 0.10
}

fn halton_jitter(index: u64) -> [f32; 2] {
    // Start at sample one: Halton sample zero is a corner. Center the complete 16-phase cycle so
    // a converged history has no persistent sub-pixel bias in either axis.
    [
        halton(index + 1, 2) - 0.5 - JITTER_SEQUENCE_CENTROID[0],
        halton(index + 1, 3) - 0.5 - JITTER_SEQUENCE_CENTROID[1],
    ]
}

fn halton(mut index: u64, base: u64) -> f32 {
    let mut fraction = 1.0_f32;
    let mut result = 0.0_f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

fn jitter_view_projection(
    mut view_projection: [f32; 16],
    jitter_pixels: [f32; 2],
    extent: vk::Extent2D,
) -> [f32; 16] {
    let jitter_ndc = [
        jitter_pixels[0] * 2.0 / extent.width.max(1) as f32,
        jitter_pixels[1] * 2.0 / extent.height.max(1) as f32,
    ];
    for column in 0..4 {
        let base = column * 4;
        let homogeneous_row = view_projection[base + 3];
        view_projection[base] += jitter_ndc[0] * homogeneous_row;
        view_projection[base + 1] += jitter_ndc[1] * homogeneous_row;
    }
    view_projection
}

fn invert_mat4(matrix: [f32; 16]) -> Option<[f32; 16]> {
    let mut augmented = [[0.0_f32; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[column * 4 + row];
        }
        augmented[row][row + 4] = 1.0;
    }
    for pivot_column in 0..4 {
        let mut pivot_row = pivot_column;
        for row in pivot_column + 1..4 {
            if augmented[row][pivot_column].abs() > augmented[pivot_row][pivot_column].abs() {
                pivot_row = row;
            }
        }
        if augmented[pivot_row][pivot_column].abs() <= 1e-8 {
            return None;
        }
        if pivot_row != pivot_column {
            augmented.swap(pivot_row, pivot_column);
        }
        let inverse_pivot = 1.0 / augmented[pivot_column][pivot_column];
        for column in 0..8 {
            augmented[pivot_column][column] *= inverse_pivot;
        }
        for row in 0..4 {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[row][pivot_column];
            for column in 0..8 {
                augmented[row][column] -= factor * augmented[pivot_column][column];
            }
        }
    }
    let mut inverse = [0.0_f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            inverse[column * 4 + row] = augmented[row][column + 4];
        }
    }
    Some(inverse)
}

fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn camera_view_matrix(camera: CameraSnapshot) -> [f32; 16] {
    let forward = normalize3(sub3(camera.target, camera.eye));
    let right = normalize_or_axis(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
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
        -dot3(right, camera.eye),
        -dot3(up, camera.eye),
        dot3(forward, camera.eye),
        1.0,
    ]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_or_axis(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = dot3(value, value).sqrt();
    if length <= 1e-6 {
        fallback
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = dot3(value, value).sqrt();
    if length <= 1e-6 {
        [0.0, 0.0, -1.0]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let delta = sub3(a, b);
    dot3(delta, delta).sqrt()
}

fn destroy_framebuffer(device: &Device, framebuffer: vk::Framebuffer) {
    if framebuffer != vk::Framebuffer::null() {
        unsafe { device.destroy_framebuffer(framebuffer, None) };
    }
}

fn destroy_render_pass(device: &Device, render_pass: vk::RenderPass) {
    if render_pass != vk::RenderPass::null() {
        unsafe { device.destroy_render_pass(render_pass, None) };
    }
}

fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline != vk::Pipeline::null() {
        unsafe { device.destroy_pipeline(pipeline, None) };
    }
}

fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout != vk::PipelineLayout::null() {
        unsafe { device.destroy_pipeline_layout(layout, None) };
    }
}

fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    }
}

fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool != vk::DescriptorPool::null() {
        unsafe { device.destroy_descriptor_pool(pool, None) };
    }
}

fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler != vk::Sampler::null() {
        unsafe { device.destroy_sampler(sampler, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        FrameId, FrameSnapshotBuilder, SceneHandle, SurfaceGeneration, SurfaceId, ViewId,
        ViewPacket,
    };

    fn transform_homogeneous(matrix: [f32; 16], value: [f32; 4]) -> [f32; 4] {
        std::array::from_fn(|row| {
            matrix[row] * value[0]
                + matrix[4 + row] * value[1]
                + matrix[8 + row] * value[2]
                + matrix[12 + row] * value[3]
        })
    }

    fn projected_uv_and_depth(matrix: [f32; 16], world: [f32; 3]) -> ([f32; 2], f32) {
        let clip = transform_homogeneous(matrix, [world[0], world[1], world[2], 1.0]);
        assert!(
            clip[3] > 0.0,
            "test point must remain in front of the camera"
        );
        let inverse_w = 1.0 / clip[3];
        (
            [
                clip[0] * inverse_w * 0.5 + 0.5,
                clip[1] * inverse_w * 0.5 + 0.5,
            ],
            clip[2] * inverse_w,
        )
    }

    fn snapshot_with_frame_id(frame_id: u64, camera: CameraSnapshot) -> FrameSnapshot {
        let frame_id = FrameId::from_raw(frame_id).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene handle is non-zero");
        let surface = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation =
            SurfaceGeneration::from_raw(1).expect("test surface generation is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let extent = NonZeroExtent::new(1600, 900).expect("test extent is non-zero");
        let mut builder = FrameSnapshotBuilder::new(frame_id, scene, surface, generation);
        builder.add_view(ViewPacket::new(view, extent).with_camera(camera));
        builder.build().expect("test snapshot has one view")
    }

    #[test]
    fn halton_jitter_stays_inside_one_pixel() {
        let mut centroid = [0.0_f32; 2];
        for index in 0..JITTER_SAMPLE_COUNT {
            let jitter = halton_jitter(index);
            assert!((-0.5..=0.5).contains(&jitter[0]));
            assert!((-0.5..=0.5).contains(&jitter[1]));
            centroid[0] += jitter[0];
            centroid[1] += jitter[1];
        }
        assert!(centroid[0].abs() < 1.0e-6);
        assert!(centroid[1].abs() < 1.0e-6);
    }

    #[test]
    fn jitter_changes_only_clip_xy_rows() {
        let original = identity_mat4();
        let jittered = jitter_view_projection(
            original,
            [0.25, -0.25],
            vk::Extent2D {
                width: 100,
                height: 50,
            },
        );
        assert_eq!(jittered[3], original[3]);
        assert_eq!(jittered[7], original[7]);
        assert_eq!(jittered[11], original[11]);
        assert_eq!(jittered[15], original[15]);
        assert_ne!(jittered[12], original[12]);
        assert_ne!(jittered[13], original[13]);
    }

    #[test]
    fn static_reprojection_removes_current_and_previous_jitter_phase_delta() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let aspect = extent.width as f32 / extent.height as f32;
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let current_jitter = halton_jitter(5);
        let previous_jitter = halton_jitter(11);
        let current_view_projection =
            jitter_view_projection(camera.view_projection(aspect), current_jitter, extent);
        let previous_view_projection =
            jitter_view_projection(camera.view_projection(aspect), previous_jitter, extent);
        let inverse_current =
            invert_mat4(current_view_projection).expect("test projection is invertible");
        let world = [0.35, -0.2, 0.0];
        let (current_uv, current_depth) = projected_uv_and_depth(current_view_projection, world);

        // Mirror temporal_reproject_world: reconstruct from the current jittered device sample,
        // then project through the exact previous jittered matrix.
        let current_clip = [
            current_uv[0] * 2.0 - 1.0,
            current_uv[1] * 2.0 - 1.0,
            current_depth,
            1.0,
        ];
        let reconstructed_h = transform_homogeneous(inverse_current, current_clip);
        let inverse_reconstructed_w = 1.0 / reconstructed_h[3];
        let reconstructed_world = [
            reconstructed_h[0] * inverse_reconstructed_w,
            reconstructed_h[1] * inverse_reconstructed_w,
            reconstructed_h[2] * inverse_reconstructed_w,
        ];
        let (matrix_previous_uv, _) =
            projected_uv_and_depth(previous_view_projection, reconstructed_world);
        let texel_size = [1.0 / extent.width as f32, 1.0 / extent.height as f32];
        let stable_previous_uv = [
            matrix_previous_uv[0] + (current_jitter[0] - previous_jitter[0]) * texel_size[0],
            matrix_previous_uv[1] + (current_jitter[1] - previous_jitter[1]) * texel_size[1],
        ];

        assert!(
            (matrix_previous_uv[0] - current_uv[0]).abs() > 1.0e-5
                || (matrix_previous_uv[1] - current_uv[1]).abs() > 1.0e-5,
            "different jitter phases must produce a measurable raw reprojection delta"
        );
        assert!((stable_previous_uv[0] - current_uv[0]).abs() < 1.0e-5);
        assert!((stable_previous_uv[1] - current_uv[1]).abs() < 1.0e-5);
    }

    #[test]
    fn skipped_frame_ids_continue_history_but_stale_ids_reset_it() {
        let camera = CameraSnapshot::default();
        let aspect = 16.0 / 9.0;
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera,
            aspect,
            view_projection: camera.view_projection(aspect),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
        };
        let skipped_forward = snapshot_with_frame_id(12, camera);
        let duplicate = snapshot_with_frame_id(10, camera);
        let stale = snapshot_with_frame_id(9, camera);

        assert!(!temporal_reprojection_history_discontinuous(
            previous,
            &skipped_forward,
            camera,
            aspect,
        ));
        assert!(temporal_reprojection_history_discontinuous(
            previous, &duplicate, camera, aspect,
        ));
        assert!(temporal_reprojection_history_discontinuous(
            previous, &stale, camera, aspect,
        ));
    }

    #[test]
    fn temporal_uniform_layout_matches_slang_constant_buffer() {
        assert_eq!(size_of::<TemporalResolveUniform>(), 368);
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, inverse_current_view),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, inverse_previous_view),
            256
        );
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, texel_feedback_reset),
            320
        );
    }

    #[test]
    fn descriptors_cover_current_color_taa_and_frame_ping_pong() {
        let frame_slots = 2;
        let mut indices = std::collections::BTreeSet::new();
        for current_color_input in 0..CURRENT_COLOR_INPUT_COUNT {
            for taa_write in 0..TAA_HISTORY_COUNT {
                for slot in 0..frame_slots {
                    indices.insert(taa_descriptor_index(
                        current_color_input,
                        taa_write,
                        slot,
                        frame_slots,
                    ));
                }
            }
        }
        assert_eq!(
            indices,
            (0..CURRENT_COLOR_INPUT_COUNT * TAA_HISTORY_COUNT * frame_slots).collect()
        );
    }
}
