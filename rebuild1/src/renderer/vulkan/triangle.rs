use std::{
    ffi::CStr,
    io::Cursor,
    mem::{offset_of, size_of},
};

use ash::{Device, Instance, util, vk};

use crate::renderer::pipeline::shader_interface;

use super::{
    VulkanError,
    buffer::{
        GpuBuffer, create_buffer_with_data, destroy_buffers, memory_properties, write_buffer_value,
    },
};

const SHADER_ENTRY: &CStr = c"main";
const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug_triangle.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug_triangle.frag.spv"));
const DEBUG_TRIANGLE_VERTICES: [DebugVertex; 3] = [
    DebugVertex {
        position: [0.0, -0.55],
        color: [1.0, 0.24, 0.18],
    },
    DebugVertex {
        position: [0.58, 0.48],
        color: [0.18, 0.85, 0.45],
    },
    DebugVertex {
        position: [-0.58, 0.48],
        color: [0.23, 0.45, 1.0],
    },
];

pub(super) struct DebugTriangleResources {
    vertex_buffer: GpuBuffer,
    frame_uniforms: Vec<GpuBuffer>,
    frame_descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    frame_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
}

#[derive(Clone, Copy)]
pub(super) struct DebugTrianglePipeline {
    handle: vk::Pipeline,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugVertex {
    position: [f32; 2],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugFrameUniform {
    tint: [f32; 4],
}

struct DebugTriangleCreateGuard<'a> {
    device: &'a Device,
    frame_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: Option<GpuBuffer>,
    frame_uniforms: Vec<GpuBuffer>,
    descriptor_pool: vk::DescriptorPool,
    armed: bool,
}

impl DebugTriangleResources {
    /// Creates the temporary triangle resources used until real draw packets exist.
    pub(super) fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frame_count: usize,
    ) -> Result<Self, VulkanError> {
        let memory_properties = memory_properties(instance, physical_device);
        let mut guard = DebugTriangleCreateGuard::new(device);
        guard.frame_set_layout = create_frame_set_layout(device)?;
        guard.material_set_layout = create_empty_set_layout(device)?;
        guard.pass_set_layout = create_empty_set_layout(device)?;
        guard.pipeline_layout = create_pipeline_layout(
            device,
            guard.frame_set_layout,
            guard.material_set_layout,
            guard.pass_set_layout,
        )?;
        guard.vertex_buffer = Some(create_buffer_with_data(
            device,
            &memory_properties,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &DEBUG_TRIANGLE_VERTICES,
        )?);
        guard.frame_uniforms = create_frame_uniforms(device, &memory_properties, frame_count)?;
        guard.descriptor_pool = create_descriptor_pool(device, frame_count)?;
        let frame_descriptor_sets = allocate_frame_descriptor_sets(
            device,
            guard.descriptor_pool,
            guard.frame_set_layout,
            frame_count,
        )?;
        update_frame_descriptor_sets(device, &frame_descriptor_sets, &guard.frame_uniforms);

        tracing::info!(
            vertices = DEBUG_TRIANGLE_VERTICES.len(),
            frame_uniforms = guard.frame_uniforms.len(),
            "created debug triangle resources"
        );

        Ok(guard.finish(frame_descriptor_sets))
    }

    /// Creates the graphics pipeline compatible with one swapchain render pass.
    pub(super) fn create_pipeline(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<DebugTrianglePipeline, VulkanError> {
        let pipeline = create_debug_triangle_pipeline(device, self.pipeline_layout, render_pass)?;
        tracing::info!("created debug triangle pipeline");
        Ok(DebugTrianglePipeline { handle: pipeline })
    }

    /// Writes the small frame descriptor used by the debug triangle shader.
    pub(super) fn write_frame_uniform(
        &self,
        device: &Device,
        frame_slot: usize,
        tint: [f32; 4],
    ) -> Result<(), VulkanError> {
        let uniform =
            self.frame_uniforms
                .get(frame_slot)
                .ok_or(VulkanError::FrameSlotIndexOutOfRange {
                    index: frame_slot,
                    count: self.frame_uniforms.len(),
                })?;
        let data = DebugFrameUniform { tint };

        write_buffer_value(device, uniform, &data)
    }

    /// Binds the temporary triangle pipeline, frame descriptor, and vertex buffer, then draws it.
    pub(super) fn bind_and_draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        pipeline: DebugTrianglePipeline,
        frame_slot: usize,
        extent: vk::Extent2D,
    ) -> Result<(), VulkanError> {
        let descriptor_set = self.frame_descriptor_sets.get(frame_slot).copied().ok_or(
            VulkanError::FrameSlotIndexOutOfRange {
                index: frame_slot,
                count: self.frame_descriptor_sets.len(),
            },
        )?;
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = [vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent)];
        let vertex_buffers = [self.vertex_buffer.handle()];
        let offsets = [0_u64];
        let descriptor_sets = [descriptor_set];

        tracing::trace!(
            vertices = DEBUG_TRIANGLE_VERTICES.len(),
            width = extent.width,
            height = extent.height,
            "recording debug triangle draw"
        );

        // Safety: the command buffer is recording inside a compatible render pass, the pipeline
        // was created for that pass, and all bound resources live until the frame completes.
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
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
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_draw(
                command_buffer,
                DEBUG_TRIANGLE_VERTICES.len() as u32,
                1,
                0,
                0,
            );
        }

        Ok(())
    }

    /// Destroys one swapchain-owned debug triangle pipeline.
    pub(super) fn destroy_pipeline(&self, device: &Device, pipeline: DebugTrianglePipeline) {
        destroy_pipeline(device, pipeline.handle);
    }

    /// Destroys all device-level resources owned by the debug triangle.
    pub(super) fn destroy(self, device: &Device) {
        tracing::trace!(
            frame_uniforms = self.frame_uniforms.len(),
            "destroying debug triangle resources"
        );

        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_buffers(device, self.frame_uniforms);
        self.vertex_buffer.destroy(device);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.pass_set_layout);
        destroy_descriptor_set_layout(device, self.material_set_layout);
        destroy_descriptor_set_layout(device, self.frame_set_layout);
    }
}

impl<'a> DebugTriangleCreateGuard<'a> {
    /// Starts a guarded debug triangle creation sequence that cleans up on early return.
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            frame_set_layout: vk::DescriptorSetLayout::null(),
            material_set_layout: vk::DescriptorSetLayout::null(),
            pass_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vertex_buffer: None,
            frame_uniforms: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            armed: true,
        }
    }

    /// Converts fully-created guarded resources into the long-lived owner.
    fn finish(mut self, frame_descriptor_sets: Vec<vk::DescriptorSet>) -> DebugTriangleResources {
        self.armed = false;

        DebugTriangleResources {
            vertex_buffer: self
                .vertex_buffer
                .take()
                .expect("debug triangle vertex buffer is created before finish"),
            frame_uniforms: std::mem::take(&mut self.frame_uniforms),
            frame_descriptor_sets,
            descriptor_pool: self.descriptor_pool,
            frame_set_layout: self.frame_set_layout,
            material_set_layout: self.material_set_layout,
            pass_set_layout: self.pass_set_layout,
            pipeline_layout: self.pipeline_layout,
        }
    }
}

impl Drop for DebugTriangleCreateGuard<'_> {
    /// Destroys partially-created debug triangle resources when creation fails.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        destroy_descriptor_pool(self.device, self.descriptor_pool);
        destroy_buffers(self.device, std::mem::take(&mut self.frame_uniforms));

        if let Some(vertex_buffer) = self.vertex_buffer.take() {
            vertex_buffer.destroy(self.device);
        }

        destroy_pipeline_layout(self.device, self.pipeline_layout);
        destroy_descriptor_set_layout(self.device, self.pass_set_layout);
        destroy_descriptor_set_layout(self.device, self.material_set_layout);
        destroy_descriptor_set_layout(self.device, self.frame_set_layout);
    }
}

/// Creates the frame descriptor set layout shared with `shaders/debug_triangle.*`.
fn create_frame_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(shader_interface::FRAME_TINT_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates an intentionally empty descriptor set layout for future material/pass sets.
fn create_empty_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let create_info = vk::DescriptorSetLayoutCreateInfo::default();

    // Safety: no pointers are stored beyond the call and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the pipeline layout with the fixed frame/material/pass set order.
fn create_pipeline_layout(
    device: &Device,
    frame_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let mut set_layouts = [vk::DescriptorSetLayout::null(); 3];
    set_layouts[shader_interface::FRAME_SET as usize] = frame_set_layout;
    set_layouts[shader_interface::MATERIAL_SET as usize] = material_set_layout;
    set_layouts[shader_interface::PASS_SET as usize] = pass_set_layout;
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

    // Safety: descriptor set layouts are alive for the duration of this pipeline layout.
    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates one host-visible uniform buffer for each frame slot.
fn create_frame_uniforms(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>, VulkanError> {
    let initial = [DebugFrameUniform {
        tint: [1.0, 1.0, 1.0, 1.0],
    }];
    let mut buffers = Vec::with_capacity(frame_count);

    for _ in 0..frame_count {
        match create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &initial,
        ) {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => {
                destroy_buffers(device, buffers);
                return Err(error);
            }
        }
    }

    Ok(buffers)
}

/// Creates the descriptor pool used for per-frame uniform descriptors.
fn create_descriptor_pool(
    device: &Device,
    frame_count: usize,
) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(frame_count as u32);
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frame_count as u32)
        .pool_sizes(&pool_sizes);

    // Safety: pool sizes are local values and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates one frame descriptor set per frame slot.
fn allocate_frame_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    frame_set_layout: vk::DescriptorSetLayout,
    frame_count: usize,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    let layouts = vec![frame_set_layout; frame_count];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and layouts are alive for the allocation call.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

/// Writes uniform buffer descriptors for every frame descriptor set.
fn update_frame_descriptor_sets(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    frame_uniforms: &[GpuBuffer],
) {
    for (&descriptor_set, uniform) in descriptor_sets.iter().zip(frame_uniforms) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform.handle())
            .offset(0)
            .range(size_of::<DebugFrameUniform>() as vk::DeviceSize)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(shader_interface::FRAME_TINT_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];

        // Safety: descriptor sets were allocated from the pool and the buffer infos are valid.
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

/// Creates the graphics pipeline that draws the temporary debug triangle.
fn create_debug_triangle_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline, VulkanError> {
    let vertex_shader = create_shader_module(device, VERTEX_SHADER)?;
    let fragment_shader = match create_shader_module(device, FRAGMENT_SHADER) {
        Ok(shader) => shader,
        Err(error) => {
            destroy_shader_module(device, vertex_shader);
            return Err(error);
        }
    };
    let pipeline = create_graphics_pipeline(
        device,
        pipeline_layout,
        render_pass,
        vertex_shader,
        fragment_shader,
    );

    destroy_shader_module(device, fragment_shader);
    destroy_shader_module(device, vertex_shader);
    pipeline
}

/// Creates one shader module from build-script compiled SPIR-V bytes.
fn create_shader_module(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule, VulkanError> {
    let code = util::read_spv(&mut Cursor::new(bytes)).map_err(VulkanError::ShaderCodeRead)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    // Safety: SPIR-V bytes are generated by `build.rs` and copied into a local word vector.
    unsafe { device.create_shader_module(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the Vulkan graphics pipeline object after shader modules exist.
fn create_graphics_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
) -> Result<vk::Pipeline, VulkanError> {
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
    let vertex_bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<DebugVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let vertex_attributes = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(DebugVertex, position) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(DebugVertex, color) as u32),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attributes);
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
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
        vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    );
    let color_blend_attachments = [color_blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
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
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_infos = [pipeline_info];

    // Safety: all pipeline state references live for the duration of the call, and the render pass
    // is compatible with the framebuffer pass used during command recording.
    match unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
    } {
        Ok(mut pipelines) => Ok(pipelines.remove(0)),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                destroy_pipeline(device, pipeline);
            }
            Err(VulkanError::Vk(error))
        }
    }
}

/// Destroys one descriptor set layout.
fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout == vk::DescriptorSetLayout::null() {
        return;
    }

    // Safety: descriptor set layouts are destroyed after dependent pools and pipeline layouts.
    unsafe {
        device.destroy_descriptor_set_layout(layout, None);
    }
}

/// Destroys one descriptor pool and all descriptor sets allocated from it.
fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool == vk::DescriptorPool::null() {
        return;
    }

    // Safety: descriptor sets from this pool are no longer used by in-flight command buffers.
    unsafe {
        device.destroy_descriptor_pool(pool, None);
    }
}

/// Destroys one pipeline layout.
fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout == vk::PipelineLayout::null() {
        return;
    }

    // Safety: all pipelines that reference this layout are destroyed before the layout.
    unsafe {
        device.destroy_pipeline_layout(layout, None);
    }
}

/// Destroys one graphics pipeline.
fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline == vk::Pipeline::null() {
        return;
    }

    // Safety: the pipeline was created by this device and is no longer referenced by commands.
    unsafe {
        device.destroy_pipeline(pipeline, None);
    }
}

/// Destroys one temporary shader module.
fn destroy_shader_module(device: &Device, shader: vk::ShaderModule) {
    if shader == vk::ShaderModule::null() {
        return;
    }

    // Safety: pipeline creation has finished before temporary shader modules are destroyed.
    unsafe {
        device.destroy_shader_module(shader, None);
    }
}
