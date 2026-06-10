use std::{ffi::CStr, io::Cursor, mem::size_of};

use ash::{Device, util, vk};

use crate::{
    protocol::{AntiAliasingQualitySettings, CameraEffects, CameraSnapshot, RenderQualitySettings},
    renderer::pipeline::shader_interface,
};

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
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
}

struct PostBuild<'a> {
    device: &'a Device,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    pass_set_layout: Option<vk::DescriptorSetLayout>,
    empty_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set: Option<vk::DescriptorSet>,
    color_sampler: Option<vk::Sampler>,
    data_sampler: Option<vk::Sampler>,
    finished: bool,
}

impl<'a> PostBuild<'a> {
    /// Starts a guarded post-pipeline build that cleans up unless `finish` consumes it.
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            pipeline: None,
            pipeline_layout: None,
            pass_set_layout: None,
            empty_set_layout: None,
            descriptor_pool: None,
            descriptor_set: None,
            color_sampler: None,
            data_sampler: None,
            finished: false,
        }
    }

    /// Moves all successfully-created objects into the runtime pipeline owner.
    fn finish(mut self) -> PostPipeline {
        let pipeline = take_created(&mut self.pipeline, "post pipeline");
        let pipeline_layout = take_created(&mut self.pipeline_layout, "post pipeline layout");
        let pass_set_layout = take_created(&mut self.pass_set_layout, "post pass set layout");
        let empty_set_layout = take_created(&mut self.empty_set_layout, "post empty set layout");
        let descriptor_pool = take_created(&mut self.descriptor_pool, "post descriptor pool");
        let descriptor_set = take_created(&mut self.descriptor_set, "post descriptor set");
        let color_sampler = take_created(&mut self.color_sampler, "post color sampler");
        let data_sampler = take_created(&mut self.data_sampler, "post data sampler");
        self.finished = true;

        PostPipeline {
            pipeline,
            pipeline_layout,
            pass_set_layout,
            empty_set_layout,
            descriptor_pool,
            descriptor_set,
            color_sampler,
            data_sampler,
        }
    }

    /// Returns the empty set layout after it has been created.
    fn empty_set_layout(&self) -> vk::DescriptorSetLayout {
        expect_created(self.empty_set_layout, "post empty set layout")
    }

    /// Returns the pass set layout after it has been created.
    fn pass_set_layout(&self) -> vk::DescriptorSetLayout {
        expect_created(self.pass_set_layout, "post pass set layout")
    }

    /// Returns the pipeline layout after it has been created.
    fn pipeline_layout(&self) -> vk::PipelineLayout {
        expect_created(self.pipeline_layout, "post pipeline layout")
    }

    /// Returns the descriptor pool after it has been created.
    fn descriptor_pool(&self) -> vk::DescriptorPool {
        expect_created(self.descriptor_pool, "post descriptor pool")
    }

    /// Returns the descriptor set after it has been allocated.
    fn descriptor_set(&self) -> vk::DescriptorSet {
        expect_created(self.descriptor_set, "post descriptor set")
    }

    /// Returns the color sampler after it has been created.
    fn color_sampler(&self) -> vk::Sampler {
        expect_created(self.color_sampler, "post color sampler")
    }

    /// Returns the metadata sampler after it has been created.
    fn data_sampler(&self) -> vk::Sampler {
        expect_created(self.data_sampler, "post data sampler")
    }
}

impl Drop for PostBuild<'_> {
    /// Releases partially-created post resources when creation fails.
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
        if let Some(layout) = self.pass_set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
        if let Some(layout) = self.empty_set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PostPushConstants {
    white_balance: [f32; 4],
    camera: [f32; 4],
    depth: [f32; 4],
    ssao: [f32; 4],
    aa: [f32; 4],
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
        scene_depth_view: vk::ImageView,
        scene_normal_roughness_view: vk::ImageView,
    ) -> Result<Self, VulkanError> {
        let mut build = PostBuild::new(device);
        build.empty_set_layout = Some(create_empty_set_layout(device)?);
        build.pass_set_layout = Some(create_pass_set_layout(device)?);
        build.pipeline_layout = Some(create_pipeline_layout(
            device,
            build.empty_set_layout(),
            build.pass_set_layout(),
        )?);
        build.color_sampler = Some(create_sampler(device, vk::Filter::LINEAR)?);
        build.data_sampler = Some(create_sampler(device, vk::Filter::NEAREST)?);
        build.descriptor_pool = Some(create_descriptor_pool(device)?);
        build.descriptor_set = Some(allocate_descriptor_set(
            device,
            build.descriptor_pool(),
            build.pass_set_layout(),
        )?);
        update_descriptor_set(
            device,
            build.descriptor_set(),
            scene_color_view,
            scene_depth_view,
            scene_normal_roughness_view,
            build.color_sampler(),
            build.data_sampler(),
        );
        build.pipeline = Some(create_post_pipeline(
            device,
            build.pipeline_layout(),
            render_pass,
        )?);

        tracing::info!("created Vulkan post pipeline");
        Ok(build.finish())
    }

    /// Records the full-screen post pass that copies scene color into the swapchain image.
    pub(super) fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        camera_effects: CameraEffects,
        camera: CameraSnapshot,
        quality: RenderQualitySettings,
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
        let ssao = quality.ssao();
        let anti_aliasing = quality.anti_aliasing();
        let post = quality.post();
        let push = PostPushConstants {
            white_balance: [white_balance[0], white_balance[1], white_balance[2], 1.0],
            camera: post_camera_params(camera, extent),
            depth: post_depth_params(camera),
            ssao: [
                ssao.intensity(),
                ssao.radius(),
                ssao.bias(),
                ssao.sample_count() as f32,
            ],
            aa: post_aa_params(extent, anti_aliasing),
            exposure: camera_effects.exposure().value(),
            contrast: (camera_effects.contrast() * post.contrast()).clamp(0.25, 4.0),
            saturation: (camera_effects.saturation() * post.saturation()).clamp(0.0, 4.0),
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
            ssao_intensity = push.ssao[0],
            aa_threshold = push.aa[2],
            aa_blend = push.aa[3],
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
        destroy_sampler(device, self.data_sampler);
        destroy_sampler(device, self.color_sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.pass_set_layout);
        destroy_descriptor_set_layout(device, self.empty_set_layout);
    }
}

/// Creates the descriptor set layout used by the post shader's scene textures.
fn create_pass_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        post_sampler_binding(0),
        post_sampler_binding(1),
        post_sampler_binding(2),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: descriptor binding data is local and lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates one sampled-image binding for the post pass descriptor set.
fn post_sampler_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
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

/// Creates one post-pass sampler with explicit filtering for color or metadata images.
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

    // Safety: sampler create info contains only local values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the descriptor pool for scene color, depth, and normal/roughness sampler bindings.
fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(4);
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

/// Writes sampled scene color, depth, and normal/roughness images into the post descriptor set.
fn update_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    scene_color_view: vk::ImageView,
    scene_depth_view: vk::ImageView,
    scene_normal_roughness_view: vk::ImageView,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
) {
    let color_info = [post_image_info(color_sampler, scene_color_view)];
    let depth_info = [post_image_info(data_sampler, scene_depth_view)];
    let normal_roughness_info = [post_image_info(data_sampler, scene_normal_roughness_view)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&color_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&normal_roughness_info),
    ];

    // Safety: descriptor set, sampler, and image view belong to this device and remain alive.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Builds one post-pass sampled-image descriptor with shader-read layout.
fn post_image_info(sampler: vk::Sampler, image_view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
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

/// Packs perspective coefficients used by the post shader's view-space math.
///
/// The shader reconstructs and reprojects SSR rays from these signed coefficients, so the
/// Vulkan Y flip in `CameraSnapshot::view_projection` must be preserved here. The inverse
/// coefficients are precomputed here because view reconstruction runs for every post pixel.
fn post_camera_params(camera: CameraSnapshot, extent: vk::Extent2D) -> [f32; 4] {
    let aspect = (extent.width as f32 / extent.height.max(1) as f32).max(0.0001);
    let tan_half_fov = (camera.fov_y_radians * 0.5).tan();
    let tan_half_fov = if tan_half_fov.is_finite() && tan_half_fov > 0.0001 {
        tan_half_fov
    } else {
        0.57735026
    };
    let f = 1.0 / tan_half_fov;
    let focal_x = f / aspect;
    let focal_y = -f;

    [focal_x, focal_y, 1.0 / focal_x, 1.0 / focal_y]
}

/// Packs depth constants that are reused by every post-process depth reconstruction.
fn post_depth_params(camera: CameraSnapshot) -> [f32; 4] {
    let near = camera.near.max(0.0001);
    let far = camera.far.max(near + 0.001);
    [near, far, near * far, far - near]
}

/// Packs high-quality post-process AA texel size and edge resolve settings.
fn post_aa_params(extent: vk::Extent2D, anti_aliasing: AntiAliasingQualitySettings) -> [f32; 4] {
    [
        1.0 / extent.width.max(1) as f32,
        1.0 / extent.height.max(1) as f32,
        anti_aliasing.edge_threshold(),
        anti_aliasing.blend(),
    ]
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
