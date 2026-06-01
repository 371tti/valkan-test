use std::{ffi::CStr, io::Cursor, mem::size_of};

use ash::{Device, util, vk};

use crate::{protocol::CameraEffects, renderer::pipeline::shader_interface};

use super::VulkanError;

const SHADER_ENTRY: &CStr = c"main";
const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv"));

pub(super) struct PostPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    pass_set_layout: vk::DescriptorSetLayout,
    empty_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PostPushConstants {
    white_balance: [f32; 4],
    exposure: f32,
    contrast: f32,
    saturation: f32,
    enabled: f32,
}

impl PostPipeline {
    /// Creates the post pipeline that samples scene color and writes to the swapchain pass.
    pub(super) fn create(
        device: &Device,
        render_pass: vk::RenderPass,
        scene_color_view: vk::ImageView,
    ) -> Result<Self, VulkanError> {
        let empty_set_layout = create_empty_set_layout(device)?;
        let pass_set_layout = match create_pass_set_layout(device) {
            Ok(layout) => layout,
            Err(error) => {
                destroy_descriptor_set_layout(device, empty_set_layout);
                return Err(error);
            }
        };
        let pipeline_layout =
            match create_pipeline_layout(device, empty_set_layout, pass_set_layout) {
                Ok(layout) => layout,
                Err(error) => {
                    destroy_descriptor_set_layout(device, pass_set_layout);
                    destroy_descriptor_set_layout(device, empty_set_layout);
                    return Err(error);
                }
            };
        let sampler = match create_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, empty_set_layout);
                return Err(error);
            }
        };
        let descriptor_pool = match create_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_sampler(device, sampler);
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, empty_set_layout);
                return Err(error);
            }
        };
        let descriptor_set = match allocate_descriptor_set(device, descriptor_pool, pass_set_layout)
        {
            Ok(set) => set,
            Err(error) => {
                destroy_descriptor_pool(device, descriptor_pool);
                destroy_sampler(device, sampler);
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, empty_set_layout);
                return Err(error);
            }
        };
        update_descriptor_set(device, descriptor_set, scene_color_view, sampler);
        let pipeline = match create_post_pipeline(device, pipeline_layout, render_pass) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                destroy_descriptor_pool(device, descriptor_pool);
                destroy_sampler(device, sampler);
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, empty_set_layout);
                return Err(error);
            }
        };

        tracing::info!("created Vulkan post pipeline");
        Ok(Self {
            pipeline,
            pipeline_layout,
            pass_set_layout,
            empty_set_layout,
            descriptor_pool,
            descriptor_set,
            sampler,
        })
    }

    /// Records the full-screen post pass that copies scene color into the swapchain image.
    pub(super) fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        camera_effects: CameraEffects,
    ) {
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
        let descriptor_sets = [self.descriptor_set];
        let white_balance = camera_effects.white_balance();
        let push = PostPushConstants {
            white_balance: [white_balance[0], white_balance[1], white_balance[2], 1.0],
            exposure: camera_effects.exposure().value(),
            contrast: camera_effects.contrast(),
            saturation: camera_effects.saturation(),
            enabled: if camera_effects.enabled() { 1.0 } else { 0.0 },
        };
        let push_bytes = push_constant_bytes(&push);

        tracing::trace!(
            width = extent.width,
            height = extent.height,
            exposure = push.exposure,
            white_balance = ?white_balance,
            contrast = push.contrast,
            saturation = push.saturation,
            enabled = camera_effects.enabled(),
            "recording Vulkan post pass"
        );

        // Safety: the command buffer is inside the post render pass, the pipeline is compatible
        // with that pass, and the descriptor set points at the live scene color image view.
        unsafe {
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
                shader_interface::PASS_SET,
                &descriptor_sets,
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }

    /// Destroys all swapchain-dependent post resources.
    pub(super) fn destroy(self, device: &Device) {
        destroy_pipeline(device, self.pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.pass_set_layout);
        destroy_descriptor_set_layout(device, self.empty_set_layout);
    }
}

/// Creates the descriptor set layout used by the post shader's scene texture.
fn create_pass_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: descriptor binding data is local and lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates an empty descriptor set layout for unused frame/material slots.
fn create_empty_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let create_info = vk::DescriptorSetLayoutCreateInfo::default();

    // Safety: no pointers are stored after the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates a pipeline layout that keeps frame/material/pass set order stable.
fn create_pipeline_layout(
    device: &Device,
    empty_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let mut set_layouts = [empty_set_layout; 3];
    set_layouts[shader_interface::PASS_SET as usize] = pass_set_layout;
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<PostPushConstants>() as u32);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);

    // Safety: descriptor set layouts and push constant ranges live for this call.
    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the sampler used to read the scene color target.
fn create_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);

    // Safety: sampler create info contains only local values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the descriptor pool for one scene color sampler binding.
fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);

    // Safety: pool sizes are local values.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates the descriptor set bound by the post pass.
fn allocate_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    pass_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VulkanError> {
    let layouts = [pass_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and set layout are alive for allocation.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map(|mut sets| sets.remove(0))
        .map_err(VulkanError::Vk)
}

/// Writes the sampled scene color image into the post descriptor set.
fn update_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    scene_color_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_info = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(scene_color_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info)];

    // Safety: descriptor set, sampler, and image view belong to this device and remain alive.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Creates the full-screen post graphics pipeline.
fn create_post_pipeline(
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

    // Safety: SPIR-V bytes are copied into an owned word vector for this call.
    unsafe { device.create_shader_module(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates fixed-function state for a full-screen triangle post pass.
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
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_infos = [pipeline_info];

    // Safety: pipeline state references are local and render pass compatibility is fixed.
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

/// Views a push-constant struct as bytes for Vulkan command recording.
fn push_constant_bytes<T>(value: &T) -> &[u8] {
    // Safety: push constants are plain repr(C) data and are copied during command recording.
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

/// Destroys one descriptor set layout.
fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        // Safety: descriptor set layouts are destroyed after dependent objects.
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    }
}

/// Destroys one pipeline layout.
fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout != vk::PipelineLayout::null() {
        // Safety: all pipelines using this layout are destroyed first.
        unsafe { device.destroy_pipeline_layout(layout, None) };
    }
}

/// Destroys one descriptor pool and descriptor sets allocated from it.
fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool != vk::DescriptorPool::null() {
        // Safety: descriptor sets from this pool are no longer in use.
        unsafe { device.destroy_descriptor_pool(pool, None) };
    }
}

/// Destroys one sampler.
fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler != vk::Sampler::null() {
        // Safety: the sampler was created by this device and is no longer referenced.
        unsafe { device.destroy_sampler(sampler, None) };
    }
}

/// Destroys one graphics pipeline.
fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline != vk::Pipeline::null() {
        // Safety: the pipeline was created by this device and is no longer referenced.
        unsafe { device.destroy_pipeline(pipeline, None) };
    }
}

/// Destroys one temporary shader module.
fn destroy_shader_module(device: &Device, shader: vk::ShaderModule) {
    if shader != vk::ShaderModule::null() {
        // Safety: shader modules are destroyed after pipeline creation returns.
        unsafe { device.destroy_shader_module(shader, None) };
    }
}
