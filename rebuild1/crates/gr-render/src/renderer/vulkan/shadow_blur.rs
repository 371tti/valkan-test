use std::{ffi::CStr, io::Cursor, mem::size_of};

use ash::{Device, util, vk};

use crate::renderer::graph::SHADOW_CASCADE_COUNT;

use super::VulkanError;

const SHADER_ENTRY: &CStr = c"main";
const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
const FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_moment_blur.frag.spv"));
const SOURCE_MOMENTS_BINDING: u32 = 0;

pub(super) struct ShadowMomentBlurPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    horizontal_sets: [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
    vertical_sets: [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
    sampler: vk::Sampler,
}

struct ShadowMomentBlurBuild<'a> {
    device: &'a Device,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    horizontal_sets: Option<[vk::DescriptorSet; SHADOW_CASCADE_COUNT]>,
    vertical_sets: Option<[vk::DescriptorSet; SHADOW_CASCADE_COUNT]>,
    sampler: Option<vk::Sampler>,
    finished: bool,
}

impl<'a> ShadowMomentBlurBuild<'a> {
    /// Tracks partially-created shadow blur objects so setup can fail cleanly.
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            pipeline: None,
            pipeline_layout: None,
            set_layout: None,
            descriptor_pool: None,
            horizontal_sets: None,
            vertical_sets: None,
            sampler: None,
            finished: false,
        }
    }

    /// Moves all created Vulkan handles into the runtime blur pipeline owner.
    fn finish(mut self) -> ShadowMomentBlurPipeline {
        let pipeline = take_created(&mut self.pipeline, "shadow blur pipeline");
        let pipeline_layout =
            take_created(&mut self.pipeline_layout, "shadow blur pipeline layout");
        let set_layout = take_created(&mut self.set_layout, "shadow blur descriptor set layout");
        let descriptor_pool =
            take_created(&mut self.descriptor_pool, "shadow blur descriptor pool");
        let horizontal_sets =
            take_created(&mut self.horizontal_sets, "shadow blur horizontal sets");
        let vertical_sets = take_created(&mut self.vertical_sets, "shadow blur vertical sets");
        let sampler = take_created(&mut self.sampler, "shadow blur sampler");
        self.finished = true;

        ShadowMomentBlurPipeline {
            pipeline,
            pipeline_layout,
            set_layout,
            descriptor_pool,
            horizontal_sets,
            vertical_sets,
            sampler,
        }
    }
}

impl Drop for ShadowMomentBlurBuild<'_> {
    /// Releases successfully-created setup handles when shadow blur creation aborts.
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
        if let Some(sampler) = self.sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(layout) = self.pipeline_layout.take() {
            destroy_pipeline_layout(self.device, layout);
        }
        if let Some(layout) = self.set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShadowMomentBlurPush {
    texel_step: [f32; 2],
    radius_scale: f32,
    _pad: f32,
}

impl ShadowMomentBlurPipeline {
    /// Creates the separable moment blur pipeline used between shadow render and lighting.
    pub(super) fn create(
        device: &Device,
        render_pass: vk::RenderPass,
        horizontal_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        vertical_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    ) -> Result<Self, VulkanError> {
        let mut build = ShadowMomentBlurBuild::new(device);
        build.set_layout = Some(create_set_layout(device)?);
        build.pipeline_layout = Some(create_pipeline_layout(
            device,
            expect_created(build.set_layout, "shadow blur descriptor set layout"),
        )?);
        build.sampler = Some(create_sampler(device)?);
        build.descriptor_pool = Some(create_descriptor_pool(device)?);
        let sets = allocate_descriptor_sets(
            device,
            expect_created(build.descriptor_pool, "shadow blur descriptor pool"),
            expect_created(build.set_layout, "shadow blur descriptor set layout"),
        )?;
        build.horizontal_sets = Some(sets.0);
        build.vertical_sets = Some(sets.1);
        update_descriptor_sets(
            device,
            build
                .horizontal_sets
                .expect("horizontal descriptor sets were created"),
            build
                .vertical_sets
                .expect("vertical descriptor sets were created"),
            horizontal_source_views,
            vertical_source_views,
            expect_created(build.sampler, "shadow blur sampler"),
        );
        build.pipeline = Some(create_pipeline(
            device,
            expect_created(build.pipeline_layout, "shadow blur pipeline layout"),
            render_pass,
        )?);

        tracing::info!("created Vulkan shadow moment blur pipeline");
        Ok(build.finish())
    }

    /// Records the horizontal pass that blurs raw moment output into the scratch target.
    pub(super) fn draw_horizontal(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        cascade_index: usize,
        extent: vk::Extent2D,
    ) {
        self.draw(
            device,
            command_buffer,
            self.horizontal_sets[cascade_index],
            extent,
            [1.0 / extent.width.max(1) as f32, 0.0],
            cascade_blur_radius(cascade_index),
        );
    }

    /// Records the vertical pass that writes the filtered moment map sampled by lighting.
    pub(super) fn draw_vertical(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        cascade_index: usize,
        extent: vk::Extent2D,
    ) {
        self.draw(
            device,
            command_buffer,
            self.vertical_sets[cascade_index],
            extent,
            [0.0, 1.0 / extent.height.max(1) as f32],
            cascade_blur_radius(cascade_index),
        );
    }

    /// Destroys all blur descriptors, samplers, layout, and pipeline objects.
    pub(super) fn destroy(self, device: &Device) {
        destroy_pipeline(device, self.pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.set_layout);
    }

    /// Records one fullscreen blur draw with a direction-specific texel step.
    fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        extent: vk::Extent2D,
        texel_step: [f32; 2],
        radius_scale: f32,
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
        let descriptor_sets = [descriptor_set];
        let push = ShadowMomentBlurPush {
            texel_step,
            radius_scale,
            _pad: 0.0,
        };

        // Safety: command recording is inside the blur render pass and the descriptor source image
        // was transitioned to shader-read by the frame graph before this pass.
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
                0,
                &descriptor_sets,
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constant_bytes(&push),
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }
}

/// Uses a mild cascade-scaled blur so far cascades avoid blocky moment edges.
fn cascade_blur_radius(cascade_index: usize) -> f32 {
    [1.0, 1.15, 1.35, 1.6][cascade_index.min(SHADOW_CASCADE_COUNT - 1)]
}

/// Creates the sampled-image descriptor layout for one blur source texture.
fn create_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(SOURCE_MOMENTS_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates a pipeline layout with one source texture set and one small push constant block.
fn create_pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [set_layout];
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<ShadowMomentBlurPush>() as u32);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);

    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the linear sampler used by both separable blur passes.
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

/// Allocates descriptor storage for horizontal and vertical blur source images.
fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count((SHADOW_CASCADE_COUNT * 2) as u32);
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets((SHADOW_CASCADE_COUNT * 2) as u32)
        .pool_sizes(&pool_sizes);

    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates one descriptor set per cascade and blur direction.
fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
) -> Result<
    (
        [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
        [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
    ),
    VulkanError,
> {
    let layouts = vec![set_layout; SHADOW_CASCADE_COUNT * 2];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let sets =
        unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)?;
    let horizontal_sets = descriptor_array(&sets[..SHADOW_CASCADE_COUNT]);
    let vertical_sets = descriptor_array(&sets[SHADOW_CASCADE_COUNT..]);

    Ok((horizontal_sets, vertical_sets))
}

/// Copies a fixed-size descriptor slice into the cascade array type.
fn descriptor_array(slice: &[vk::DescriptorSet]) -> [vk::DescriptorSet; SHADOW_CASCADE_COUNT] {
    std::array::from_fn(|index| slice[index])
}

/// Writes source moment maps into every blur descriptor set.
fn update_descriptor_sets(
    device: &Device,
    horizontal_sets: [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
    vertical_sets: [vk::DescriptorSet; SHADOW_CASCADE_COUNT],
    horizontal_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    vertical_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    sampler: vk::Sampler,
) {
    let mut image_infos = Vec::with_capacity(SHADOW_CASCADE_COUNT * 2);
    for view in horizontal_source_views
        .into_iter()
        .chain(vertical_source_views)
    {
        image_infos.push(descriptor_image_info(sampler, view));
    }

    let mut writes = Vec::with_capacity(SHADOW_CASCADE_COUNT * 2);
    for (index, set) in horizontal_sets.into_iter().chain(vertical_sets).enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(SOURCE_MOMENTS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos[index..index + 1]),
        );
    }

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Builds a shader-read descriptor for one moment texture.
fn descriptor_image_info(sampler: vk::Sampler, view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

/// Creates the fullscreen graphics pipeline for separable moment blur.
fn create_pipeline(
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

    unsafe { device.create_shader_module(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates fixed-function state for a fullscreen blur pass.
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
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

/// Returns a created handle from a build slot with a resource-specific panic message.
fn expect_created<T: Copy>(value: Option<T>, label: &'static str) -> T {
    value.unwrap_or_else(|| panic!("{label} was not created"))
}

/// Moves a created handle out of a build slot with a resource-specific panic message.
fn take_created<T>(value: &mut Option<T>, label: &'static str) -> T {
    value
        .take()
        .unwrap_or_else(|| panic!("{label} was not created"))
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

fn destroy_shader_module(device: &Device, shader: vk::ShaderModule) {
    if shader != vk::ShaderModule::null() {
        unsafe { device.destroy_shader_module(shader, None) };
    }
}
