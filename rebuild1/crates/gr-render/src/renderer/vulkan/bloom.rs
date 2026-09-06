use std::{ffi::CStr, mem::size_of};

use ash::{Device, vk};

use crate::{protocol::BloomQualitySettings, renderer::pipeline::shader_interface};

use super::{
    VulkanError,
    shader::{self, assets},
};

const SHADER_ENTRY: &CStr = shader::ENTRY;
const VERTEX_SHADER: &[u8] = assets::POST_VERT;
const DOWNSAMPLE_SHADER: &[u8] = assets::POST_BLOOM_DOWNSAMPLE_FRAG;
const UPSAMPLE_SHADER: &[u8] = assets::POST_BLOOM_UPSAMPLE_FRAG;
const BLOOM_SOURCE_BINDING: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct BloomPushConstants {
    source_texel: [f32; 4],
    params: [f32; 4],
}

impl BloomPushConstants {
    fn downsample(source_extent: vk::Extent2D, bloom: BloomQualitySettings, first: bool) -> Self {
        Self {
            source_texel: [
                1.0 / source_extent.width.max(1) as f32,
                1.0 / source_extent.height.max(1) as f32,
                0.0,
                0.0,
            ],
            params: [
                bloom.threshold(),
                if first { 1.0 } else { 0.0 },
                bloom.radius_pixels(),
                0.0,
            ],
        }
    }

    fn upsample(source_extent: vk::Extent2D, bloom: BloomQualitySettings) -> Self {
        let radius = (bloom.radius_pixels() / 18.0).clamp(0.65, 2.20);
        Self {
            source_texel: [
                1.0 / source_extent.width.max(1) as f32,
                1.0 / source_extent.height.max(1) as f32,
                0.0,
                0.0,
            ],
            params: [bloom.threshold(), 0.0, radius, 0.0],
        }
    }
}

pub(super) struct BloomPipeline {
    downsample_pipeline: vk::Pipeline,
    upsample_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    source_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    downsample_sets: Vec<vk::DescriptorSet>,
    upsample_sets: Vec<vk::DescriptorSet>,
    sampler: vk::Sampler,
}

struct BloomBuild<'a> {
    device: &'a Device,
    downsample_pipeline: Option<vk::Pipeline>,
    upsample_pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    source_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    downsample_sets: Vec<vk::DescriptorSet>,
    upsample_sets: Vec<vk::DescriptorSet>,
    sampler: Option<vk::Sampler>,
    finished: bool,
}

impl<'a> BloomBuild<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            downsample_pipeline: None,
            upsample_pipeline: None,
            pipeline_layout: None,
            source_set_layout: None,
            descriptor_pool: None,
            downsample_sets: Vec::new(),
            upsample_sets: Vec::new(),
            sampler: None,
            finished: false,
        }
    }

    fn finish(mut self) -> BloomPipeline {
        let pipeline = BloomPipeline {
            downsample_pipeline: take_created(
                &mut self.downsample_pipeline,
                "bloom downsample pipeline",
            ),
            upsample_pipeline: take_created(&mut self.upsample_pipeline, "bloom upsample pipeline"),
            pipeline_layout: take_created(&mut self.pipeline_layout, "bloom pipeline layout"),
            source_set_layout: take_created(&mut self.source_set_layout, "bloom set layout"),
            descriptor_pool: take_created(&mut self.descriptor_pool, "bloom descriptor pool"),
            downsample_sets: std::mem::take(&mut self.downsample_sets),
            upsample_sets: std::mem::take(&mut self.upsample_sets),
            sampler: take_created(&mut self.sampler, "bloom sampler"),
        };
        self.finished = true;
        pipeline
    }
}

impl Drop for BloomBuild<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        if let Some(pipeline) = self.upsample_pipeline.take() {
            destroy_pipeline(self.device, pipeline);
        }
        if let Some(pipeline) = self.downsample_pipeline.take() {
            destroy_pipeline(self.device, pipeline);
        }
        if let Some(pool) = self.descriptor_pool.take() {
            destroy_descriptor_pool(self.device, pool);
        }
        if let Some(sampler) = self.sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(layout) = self.pipeline_layout.take() {
            destroy_pipeline_layout(self.device, layout);
        }
        if let Some(layout) = self.source_set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
    }
}

impl BloomPipeline {
    /// Creates the bloom chain from the resolved scene color and its mip targets.
    pub(super) fn create(
        device: &Device,
        downsample_render_pass: vk::RenderPass,
        upsample_render_pass: vk::RenderPass,
        source_view: vk::ImageView,
        bloom_views: &[vk::ImageView],
    ) -> Result<Self, VulkanError> {
        assert!(
            !bloom_views.is_empty(),
            "bloom pipeline needs at least one mip target"
        );

        let mut build = BloomBuild::new(device);
        build.source_set_layout = Some(create_source_set_layout(device)?);
        build.pipeline_layout = Some(create_pipeline_layout(
            device,
            take_created_ref(build.source_set_layout)?,
        )?);
        build.sampler = Some(create_sampler(device)?);
        let set_count = bloom_views.len() + bloom_views.len().saturating_sub(1);
        build.descriptor_pool = Some(create_descriptor_pool(device, set_count as u32)?);
        build.downsample_sets = allocate_descriptor_sets(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.source_set_layout)?,
            bloom_views.len() as u32,
        )?;
        build.upsample_sets = allocate_descriptor_sets(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.source_set_layout)?,
            bloom_views.len().saturating_sub(1) as u32,
        )?;
        update_downsample_descriptors(
            device,
            &build.downsample_sets,
            take_created_ref(build.sampler)?,
            source_view,
            bloom_views,
        );
        update_upsample_descriptors(
            device,
            &build.upsample_sets,
            take_created_ref(build.sampler)?,
            bloom_views,
        );
        build.downsample_pipeline = Some(create_bloom_pipeline(
            device,
            take_created_ref(build.pipeline_layout)?,
            downsample_render_pass,
            DOWNSAMPLE_SHADER,
            false,
        )?);
        build.upsample_pipeline = Some(create_bloom_pipeline(
            device,
            take_created_ref(build.pipeline_layout)?,
            upsample_render_pass,
            UPSAMPLE_SHADER,
            true,
        )?);

        tracing::info!(
            bloom_mips = bloom_views.len(),
            "created Vulkan bloom mip-chain pipeline"
        );
        Ok(build.finish())
    }

    pub(super) fn draw_downsample(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        mip_index: usize,
        source_extent: vk::Extent2D,
        target_extent: vk::Extent2D,
        bloom: BloomQualitySettings,
    ) -> Result<(), VulkanError> {
        let descriptor_index = mip_index;
        let descriptor_set = self.downsample_sets.get(descriptor_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: descriptor_index,
                count: self.downsample_sets.len(),
            },
        )?;
        self.draw(
            device,
            command_buffer,
            self.downsample_pipeline,
            descriptor_set,
            target_extent,
            BloomPushConstants::downsample(source_extent, bloom, mip_index == 0),
        );
        Ok(())
    }

    pub(super) fn draw_upsample(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        target_mip_index: usize,
        source_extent: vk::Extent2D,
        target_extent: vk::Extent2D,
        bloom: BloomQualitySettings,
    ) -> Result<(), VulkanError> {
        let descriptor_set = self.upsample_sets.get(target_mip_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: target_mip_index,
                count: self.upsample_sets.len(),
            },
        )?;
        self.draw(
            device,
            command_buffer,
            self.upsample_pipeline,
            descriptor_set,
            target_extent,
            BloomPushConstants::upsample(source_extent, bloom),
        );
        Ok(())
    }

    fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        descriptor_set: vk::DescriptorSet,
        target_extent: vk::Extent2D,
        push: BloomPushConstants,
    ) {
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(target_extent.width as f32)
            .height(target_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = [vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(target_extent)];
        let descriptor_sets = [descriptor_set];
        let push_bytes = push_constant_bytes(&push);

        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
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

    pub(super) fn destroy(self, device: &Device) {
        destroy_pipeline(device, self.upsample_pipeline);
        destroy_pipeline(device, self.downsample_pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.source_set_layout);
    }
}

fn create_source_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(BLOOM_SOURCE_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_pipeline_layout(
    device: &Device,
    source_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [source_set_layout];
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<BloomPushConstants>() as u32);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);

    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

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

    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_pool(
    device: &Device,
    descriptor_count: u32,
) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(descriptor_count);
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(descriptor_count)
        .pool_sizes(&pool_sizes);

    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    source_set_layout: vk::DescriptorSetLayout,
    count: u32,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let layouts = vec![source_set_layout; count as usize];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

fn update_downsample_descriptors(
    device: &Device,
    sets: &[vk::DescriptorSet],
    sampler: vk::Sampler,
    source_view: vk::ImageView,
    bloom_views: &[vk::ImageView],
) {
    for (index, &set) in sets.iter().enumerate() {
        let source_view = if index == 0 {
            source_view
        } else {
            bloom_views[index - 1]
        };
        update_source_descriptor(device, set, sampler, source_view);
    }
}

fn update_upsample_descriptors(
    device: &Device,
    sets: &[vk::DescriptorSet],
    sampler: vk::Sampler,
    bloom_views: &[vk::ImageView],
) {
    for (target_index, &set) in sets.iter().enumerate() {
        update_source_descriptor(device, set, sampler, bloom_views[target_index + 1]);
    }
}

fn update_source_descriptor(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    image_view: vk::ImageView,
) {
    let image_info = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(BLOOM_SOURCE_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info)];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn create_bloom_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    fragment_shader_bytes: &[u8],
    additive_blend: bool,
) -> Result<vk::Pipeline, VulkanError> {
    let vertex_shader = shader::create_shader_module(device, VERTEX_SHADER)?;
    let fragment_shader = match shader::create_shader_module(device, fragment_shader_bytes) {
        Ok(shader) => shader,
        Err(error) => {
            shader::destroy_shader_module(device, vertex_shader);
            return Err(error);
        }
    };
    let pipeline = create_graphics_pipeline(
        device,
        pipeline_layout,
        render_pass,
        vertex_shader,
        fragment_shader,
        additive_blend,
    );

    shader::destroy_shader_module(device, fragment_shader);
    shader::destroy_shader_module(device, vertex_shader);
    pipeline
}

fn create_graphics_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    additive_blend: bool,
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
    let color_blend_attachment = if additive_blend {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
    } else {
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
    };
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

fn take_created_ref<T: Copy>(value: Option<T>) -> Result<T, VulkanError> {
    value.ok_or_else(|| VulkanError::GraphCompile("bloom build order is invalid".to_string()))
}

fn take_created<T>(value: &mut Option<T>, label: &'static str) -> T {
    value
        .take()
        .unwrap_or_else(|| panic!("{label} was not created"))
}

fn push_constant_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    }
}

fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout != vk::PipelineLayout::null() {
        unsafe { device.destroy_pipeline_layout(layout, None) };
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

fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline != vk::Pipeline::null() {
        unsafe { device.destroy_pipeline(pipeline, None) };
    }
}
