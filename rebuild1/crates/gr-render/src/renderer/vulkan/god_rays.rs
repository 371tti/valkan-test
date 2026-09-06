use std::{ffi::CStr, mem::size_of};

use ash::{Device, vk};

use crate::{
    math::{cross3, dot3, normalize_or, sub3},
    protocol::{BloomQualitySettings, CameraSnapshot, LightPacket, RenderQualitySettings},
    renderer::{DEFAULT_DIRECTIONAL_LIGHT_DIR, pipeline::shader_interface},
};

use super::{
    VulkanError,
    mesh::{EmissiveLightUniforms, MAX_LOCAL_LIGHTS},
    shader::{self, assets},
    taa::TemporalEffectFrame,
};

const SHADER_ENTRY: &CStr = shader::ENTRY;
const VERTEX_SHADER: &[u8] = assets::POST_VERT;
const MASK_SHADER: &[u8] = assets::POST_GOD_RAY_MASK_FRAG;
const PREFILTER_SHADER: &[u8] = assets::POST_GOD_RAY_PREFILTER_FRAG;
const RADIAL_SHADER: &[u8] = assets::POST_GOD_RAY_RADIAL_FRAG;
const TEMPORAL_SHADER: &[u8] = assets::POST_GOD_RAY_TEMPORAL_FRAG;

const MASK_SCENE_COLOR_BINDING: u32 = 0;
const MASK_SCENE_DEPTH_BINDING: u32 = 1;
const MASK_TRANSPARENT_METADATA_BINDING: u32 = 2;
const MASK_DIRECTIONAL_SHADOW_BINDING: u32 = 3;
const MASK_TAA_DEPTH_0_BINDING: u32 = 4;
const MASK_TAA_DEPTH_1_BINDING: u32 = 5;
const MASK_TRANSLUCENT_SHADOW_0_BINDING: u32 = 6;
const MASK_TRANSLUCENT_SHADOW_1_BINDING: u32 = 7;
const MASK_TRANSLUCENT_SHADOW_2_BINDING: u32 = 8;
const MASK_TRANSLUCENT_SHADOW_3_BINDING: u32 = 9;
const SOURCE_BINDING: u32 = 0;
const TEMPORAL_CURRENT_BINDING: u32 = 0;
const TEMPORAL_HISTORY_BINDING: u32 = 1;
const GOD_RAY_SOURCE_COUNT: usize = 2;
const VOLUMETRIC_GOD_RAY_MAX_DISTANCE: f32 = 160.0;

#[derive(Clone, Copy, Default)]
pub(super) struct GodRaySource {
    pub(super) source: [f32; 4],
    pub(super) color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GodRayPushConstants {
    depth: [f32; 4],
    target: [f32; 4],
    bloom: [f32; 4],
    features: [f32; 4],
    source0: [f32; 4],
    color0: [f32; 4],
    source1: [f32; 4],
    color1: [f32; 4],
    /// xy = current scene projection jitter in NDC. z = this frame's TAA depth-history index; w =
    /// 1 when that stable history is valid. The low-resolution mask is evaluated on a stable
    /// screen lattice, so its full-resolution scene/depth lookups must undo the same offset on the
    /// legacy radial route.
    jitter_ndc: [f32; 4],
    /// x = source-history feedback, y = directional-light motion, z = shared sample phase,
    /// w = temporal reset marker. The dedicated volumetric ray march consumes these values
    /// directly; the legacy radial path ignores them.
    temporal: [f32; 4],
}

impl GodRayPushConstants {
    pub(super) fn new(
        camera: CameraSnapshot,
        target_extent: vk::Extent2D,
        quality: RenderQualitySettings,
        directional_light: LightPacket,
        emissive_lights: EmissiveLightUniforms,
        has_transparent_scene_items: bool,
        history_valid: bool,
        frame_id: u64,
        jitter_ndc: [f32; 2],
        taa_history_write_index: usize,
        taa_stable_metadata_valid: bool,
        temporal: TemporalEffectFrame,
    ) -> Self {
        let bloom = quality.bloom();
        let sources = frame_god_ray_sources(
            camera,
            target_extent,
            quality,
            directional_light,
            emissive_lights,
        );
        // The high-quality path integrates a camera-ray volumetric medium. Fog and directional
        // scattering intentionally share this low-resolution history in the previous renderer
        // path, which keeps the temporal resolve stable while the scene is static.
        let fog = quality.fog();
        let volumetric = quality.features().volumetric_fog_enabled();
        let volumetric_max_distance = fog
            .max_distance()
            .min(VOLUMETRIC_GOD_RAY_MAX_DISTANCE)
            .max(camera.far)
            .max(camera.near + 0.05);
        let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
        let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
        let up = cross3(right, forward);
        // Keep the phase bounded so converting the protocol frame id to f32 never loses the
        // low bits that drive the spatiotemporal noise sequence after a long-running session.
        // Keep the frame counter bounded for precision, but fold the shared light-aware phase
        // into the actual source jitter. A post-resolve blend alone cannot remove a deterministic
        // march layer; the source must evaluate different points when the sun moves.
        let noise_frame = (frame_id % 4096) as f32
            + temporal.sample_phase * 0.9375
            + temporal.light_motion * 3.171875;

        Self {
            depth: depth_params(camera),
            target: target_params(target_extent, camera),
            bloom: bloom_params(bloom),
            features: [
                if has_transparent_scene_items {
                    1.0
                } else {
                    0.0
                },
                if history_valid { 1.0 } else { 0.0 },
                if volumetric { 1.0 } else { 0.0 },
                if fog.enabled() { 1.0 } else { 0.0 },
            ],
            // The volumetric path needs the camera basis to reconstruct each world-space camera
            // ray. Reuse the source lanes only for that path; the radial approximation keeps its
            // old payload.
            source0: if volumetric {
                [right[0], right[1], right[2], volumetric_max_distance]
            } else {
                sources[0].source
            },
            color0: if volumetric {
                [up[0], up[1], up[2], 0.0]
            } else {
                sources[0].color
            },
            source1: if volumetric {
                [forward[0], forward[1], forward[2], noise_frame]
            } else {
                sources[1].source
            },
            color1: sources[1].color,
            jitter_ndc: [
                jitter_ndc[0],
                jitter_ndc[1],
                (taa_history_write_index.min(1)) as f32,
                if taa_stable_metadata_valid { 1.0 } else { 0.0 },
            ],
            temporal: [
                temporal.sun_feedback,
                temporal.light_motion,
                temporal.sample_phase,
                if temporal.reset { 1.0 } else { 0.0 },
            ],
        }
    }
}

pub(super) struct GodRaysPipeline {
    mask_pipeline: vk::Pipeline,
    prefilter_pipeline: vk::Pipeline,
    radial_pipeline: vk::Pipeline,
    temporal_pipeline: vk::Pipeline,
    mask_pipeline_layout: vk::PipelineLayout,
    source_pipeline_layout: vk::PipelineLayout,
    temporal_pipeline_layout: vk::PipelineLayout,
    mask_set_layout: vk::DescriptorSetLayout,
    source_set_layout: vk::DescriptorSetLayout,
    temporal_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    mask_set: vk::DescriptorSet,
    prefilter_set: vk::DescriptorSet,
    radial_set: vk::DescriptorSet,
    temporal_sets: [vk::DescriptorSet; 2],
    volumetric_temporal_sets: [vk::DescriptorSet; 2],
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
    shadow_sampler: vk::Sampler,
    directional_shadow_view: vk::ImageView,
    translucent_shadow_views: [vk::ImageView; 4],
}

struct GodRaysBuild<'a> {
    device: &'a Device,
    mask_pipeline: Option<vk::Pipeline>,
    prefilter_pipeline: Option<vk::Pipeline>,
    radial_pipeline: Option<vk::Pipeline>,
    temporal_pipeline: Option<vk::Pipeline>,
    mask_pipeline_layout: Option<vk::PipelineLayout>,
    source_pipeline_layout: Option<vk::PipelineLayout>,
    temporal_pipeline_layout: Option<vk::PipelineLayout>,
    mask_set_layout: Option<vk::DescriptorSetLayout>,
    source_set_layout: Option<vk::DescriptorSetLayout>,
    temporal_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    mask_set: Option<vk::DescriptorSet>,
    prefilter_set: Option<vk::DescriptorSet>,
    radial_set: Option<vk::DescriptorSet>,
    temporal_sets: Vec<vk::DescriptorSet>,
    volumetric_temporal_sets: Vec<vk::DescriptorSet>,
    color_sampler: Option<vk::Sampler>,
    data_sampler: Option<vk::Sampler>,
    shadow_sampler: Option<vk::Sampler>,
    directional_shadow_view: Option<vk::ImageView>,
    translucent_shadow_views: Option<[vk::ImageView; 4]>,
    finished: bool,
}

impl<'a> GodRaysBuild<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            mask_pipeline: None,
            prefilter_pipeline: None,
            radial_pipeline: None,
            temporal_pipeline: None,
            mask_pipeline_layout: None,
            source_pipeline_layout: None,
            temporal_pipeline_layout: None,
            mask_set_layout: None,
            source_set_layout: None,
            temporal_set_layout: None,
            descriptor_pool: None,
            mask_set: None,
            prefilter_set: None,
            radial_set: None,
            temporal_sets: Vec::new(),
            volumetric_temporal_sets: Vec::new(),
            color_sampler: None,
            data_sampler: None,
            shadow_sampler: None,
            directional_shadow_view: None,
            translucent_shadow_views: None,
            finished: false,
        }
    }

    fn finish(mut self) -> GodRaysPipeline {
        let temporal_sets = [
            self.temporal_sets
                .first()
                .copied()
                .expect("god-ray temporal set 0 was not allocated"),
            self.temporal_sets
                .get(1)
                .copied()
                .expect("god-ray temporal set 1 was not allocated"),
        ];
        let volumetric_temporal_sets = [
            self.volumetric_temporal_sets
                .first()
                .copied()
                .expect("god-ray volumetric temporal set 0 was not allocated"),
            self.volumetric_temporal_sets
                .get(1)
                .copied()
                .expect("god-ray volumetric temporal set 1 was not allocated"),
        ];
        let pipeline = GodRaysPipeline {
            mask_pipeline: take_created(&mut self.mask_pipeline, "god-ray mask pipeline"),
            prefilter_pipeline: take_created(
                &mut self.prefilter_pipeline,
                "god-ray prefilter pipeline",
            ),
            radial_pipeline: take_created(&mut self.radial_pipeline, "god-ray radial pipeline"),
            temporal_pipeline: take_created(
                &mut self.temporal_pipeline,
                "god-ray temporal pipeline",
            ),
            mask_pipeline_layout: take_created(
                &mut self.mask_pipeline_layout,
                "god-ray mask pipeline layout",
            ),
            source_pipeline_layout: take_created(
                &mut self.source_pipeline_layout,
                "god-ray source pipeline layout",
            ),
            temporal_pipeline_layout: take_created(
                &mut self.temporal_pipeline_layout,
                "god-ray temporal pipeline layout",
            ),
            mask_set_layout: take_created(&mut self.mask_set_layout, "god-ray mask set layout"),
            source_set_layout: take_created(
                &mut self.source_set_layout,
                "god-ray source set layout",
            ),
            temporal_set_layout: take_created(
                &mut self.temporal_set_layout,
                "god-ray temporal set layout",
            ),
            descriptor_pool: take_created(&mut self.descriptor_pool, "god-ray descriptor pool"),
            mask_set: take_created(&mut self.mask_set, "god-ray mask set"),
            prefilter_set: take_created(&mut self.prefilter_set, "god-ray prefilter set"),
            radial_set: take_created(&mut self.radial_set, "god-ray radial set"),
            temporal_sets,
            volumetric_temporal_sets,
            color_sampler: take_created(&mut self.color_sampler, "god-ray color sampler"),
            data_sampler: take_created(&mut self.data_sampler, "god-ray data sampler"),
            shadow_sampler: take_created(&mut self.shadow_sampler, "god-ray shadow sampler"),
            directional_shadow_view: take_created(
                &mut self.directional_shadow_view,
                "god-ray directional shadow view",
            ),
            translucent_shadow_views: take_created(
                &mut self.translucent_shadow_views,
                "god-ray translucent shadow views",
            ),
        };
        self.finished = true;
        pipeline
    }
}

impl Drop for GodRaysBuild<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        for pipeline in [
            self.temporal_pipeline.take(),
            self.radial_pipeline.take(),
            self.prefilter_pipeline.take(),
            self.mask_pipeline.take(),
        ]
        .into_iter()
        .flatten()
        {
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
        if let Some(sampler) = self.shadow_sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        for layout in [
            self.temporal_pipeline_layout.take(),
            self.source_pipeline_layout.take(),
            self.mask_pipeline_layout.take(),
        ]
        .into_iter()
        .flatten()
        {
            destroy_pipeline_layout(self.device, layout);
        }
        for layout in [
            self.temporal_set_layout.take(),
            self.source_set_layout.take(),
            self.mask_set_layout.take(),
        ]
        .into_iter()
        .flatten()
        {
            destroy_descriptor_set_layout(self.device, layout);
        }
    }
}

impl GodRaysPipeline {
    pub(super) fn create(
        device: &Device,
        render_pass: vk::RenderPass,
        frame_set_layout: vk::DescriptorSetLayout,
        scene_color_view: vk::ImageView,
        scene_depth_view: vk::ImageView,
        scene_transparent_normal_roughness_view: vk::ImageView,
        directional_shadow_view: vk::ImageView,
        translucent_shadow_views: [vk::ImageView; 4],
        taa_depth_history_views: [vk::ImageView; 2],
        mask_view: vk::ImageView,
        prefilter_view: vk::ImageView,
        blur_view: vk::ImageView,
        history_views: [vk::ImageView; 2],
    ) -> Result<Self, VulkanError> {
        let mut build = GodRaysBuild::new(device);
        build.mask_set_layout = Some(create_mask_set_layout(device)?);
        build.source_set_layout = Some(create_source_set_layout(device)?);
        build.temporal_set_layout = Some(create_temporal_set_layout(device)?);
        build.mask_pipeline_layout = Some(create_mask_pipeline_layout(
            device,
            take_created_ref(build.mask_set_layout)?,
            frame_set_layout,
        )?);
        build.source_pipeline_layout = Some(create_pipeline_layout(
            device,
            take_created_ref(build.source_set_layout)?,
        )?);
        build.temporal_pipeline_layout = Some(create_pipeline_layout(
            device,
            take_created_ref(build.temporal_set_layout)?,
        )?);
        build.color_sampler = Some(create_sampler(device, vk::Filter::LINEAR)?);
        build.data_sampler = Some(create_sampler(device, vk::Filter::NEAREST)?);
        build.shadow_sampler = Some(create_shadow_sampler(device)?);
        build.directional_shadow_view = Some(directional_shadow_view);
        build.translucent_shadow_views = Some(translucent_shadow_views);
        build.descriptor_pool = Some(create_descriptor_pool(device)?);
        build.mask_set = Some(allocate_descriptor_set(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.mask_set_layout)?,
        )?);
        build.prefilter_set = Some(allocate_descriptor_set(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.source_set_layout)?,
        )?);
        build.radial_set = Some(allocate_descriptor_set(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.source_set_layout)?,
        )?);
        build.temporal_sets = allocate_descriptor_sets(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.temporal_set_layout)?,
            2,
        )?;
        build.volumetric_temporal_sets = allocate_descriptor_sets(
            device,
            take_created_ref(build.descriptor_pool)?,
            take_created_ref(build.temporal_set_layout)?,
            2,
        )?;

        update_mask_descriptor(
            device,
            take_created_ref(build.mask_set)?,
            take_created_ref(build.color_sampler)?,
            take_created_ref(build.data_sampler)?,
            take_created_ref(build.shadow_sampler)?,
            scene_color_view,
            scene_depth_view,
            scene_transparent_normal_roughness_view,
            directional_shadow_view,
            translucent_shadow_views,
            taa_depth_history_views,
        );
        update_source_descriptor(
            device,
            take_created_ref(build.prefilter_set)?,
            take_created_ref(build.color_sampler)?,
            mask_view,
        );
        update_source_descriptor(
            device,
            take_created_ref(build.radial_set)?,
            take_created_ref(build.color_sampler)?,
            prefilter_view,
        );
        update_temporal_descriptors(
            device,
            &build.temporal_sets,
            take_created_ref(build.color_sampler)?,
            blur_view,
            history_views,
        );
        update_temporal_descriptors(
            device,
            &build.volumetric_temporal_sets,
            take_created_ref(build.color_sampler)?,
            mask_view,
            history_views,
        );

        build.mask_pipeline = Some(create_god_ray_pipeline(
            device,
            take_created_ref(build.mask_pipeline_layout)?,
            render_pass,
            MASK_SHADER,
        )?);
        build.prefilter_pipeline = Some(create_god_ray_pipeline(
            device,
            take_created_ref(build.source_pipeline_layout)?,
            render_pass,
            PREFILTER_SHADER,
        )?);
        build.radial_pipeline = Some(create_god_ray_pipeline(
            device,
            take_created_ref(build.source_pipeline_layout)?,
            render_pass,
            RADIAL_SHADER,
        )?);
        build.temporal_pipeline = Some(create_god_ray_pipeline(
            device,
            take_created_ref(build.temporal_pipeline_layout)?,
            render_pass,
            TEMPORAL_SHADER,
        )?);

        tracing::info!("created Vulkan god-ray low-resolution pipeline");
        Ok(build.finish())
    }

    /// Returns whether the dedicated volume descriptors need to be rebound to current shadows.
    pub(super) fn shadow_views_changed(
        &self,
        directional_shadow_view: vk::ImageView,
        translucent_shadow_views: &[vk::ImageView; 4],
    ) -> bool {
        self.directional_shadow_view != directional_shadow_view
            || self.translucent_shadow_views != *translucent_shadow_views
    }

    /// Rebinds the CSM and deep translucent shadow views after shadow resources become available
    /// or are resized.
    pub(super) fn update_directional_shadow_view(
        &mut self,
        device: &Device,
        image_view: vk::ImageView,
        translucent_shadow_views: [vk::ImageView; 4],
    ) {
        if self.directional_shadow_view == image_view
            && self.translucent_shadow_views == translucent_shadow_views
        {
            return;
        }

        let info = [depth_image_info(self.shadow_sampler, image_view)];
        let translucent_info =
            translucent_shadow_views.map(|view| image_info(self.data_sampler, view));
        let writes = [
            descriptor_write(self.mask_set, MASK_DIRECTIONAL_SHADOW_BINDING, &info),
            descriptor_write(
                self.mask_set,
                MASK_TRANSLUCENT_SHADOW_0_BINDING,
                std::slice::from_ref(&translucent_info[0]),
            ),
            descriptor_write(
                self.mask_set,
                MASK_TRANSLUCENT_SHADOW_1_BINDING,
                std::slice::from_ref(&translucent_info[1]),
            ),
            descriptor_write(
                self.mask_set,
                MASK_TRANSLUCENT_SHADOW_2_BINDING,
                std::slice::from_ref(&translucent_info[2]),
            ),
            descriptor_write(
                self.mask_set,
                MASK_TRANSLUCENT_SHADOW_3_BINDING,
                std::slice::from_ref(&translucent_info[3]),
            ),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        self.directional_shadow_view = image_view;
        self.translucent_shadow_views = translucent_shadow_views;
    }

    pub(super) fn draw_mask(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        target_extent: vk::Extent2D,
        frame_descriptor_set: vk::DescriptorSet,
        push: GodRayPushConstants,
    ) {
        self.draw(
            device,
            command_buffer,
            self.mask_pipeline,
            self.mask_pipeline_layout,
            &[self.mask_set, frame_descriptor_set],
            target_extent,
            push,
        );
    }

    pub(super) fn draw_prefilter(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        target_extent: vk::Extent2D,
        push: GodRayPushConstants,
    ) {
        self.draw(
            device,
            command_buffer,
            self.prefilter_pipeline,
            self.source_pipeline_layout,
            &[self.prefilter_set],
            target_extent,
            push,
        );
    }

    pub(super) fn draw_radial(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        target_extent: vk::Extent2D,
        push: GodRayPushConstants,
    ) {
        self.draw(
            device,
            command_buffer,
            self.radial_pipeline,
            self.source_pipeline_layout,
            &[self.radial_set],
            target_extent,
            push,
        );
    }

    pub(super) fn draw_temporal(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        target_extent: vk::Extent2D,
        write_history_index: usize,
        volumetric: bool,
        push: GodRayPushConstants,
    ) -> Result<(), VulkanError> {
        let descriptor_sets = if volumetric {
            &self.volumetric_temporal_sets
        } else {
            &self.temporal_sets
        };
        let descriptor_set = descriptor_sets.get(write_history_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: write_history_index,
                count: self.temporal_sets.len(),
            },
        )?;
        self.draw(
            device,
            command_buffer,
            self.temporal_pipeline,
            self.temporal_pipeline_layout,
            &[descriptor_set],
            target_extent,
            push,
        );
        Ok(())
    }

    fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        target_extent: vk::Extent2D,
        push: GodRayPushConstants,
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
        let push_bytes = push_constant_bytes(&push);

        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                shader_interface::FRAME_SET,
                descriptor_sets,
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }

    pub(super) fn destroy(self, device: &Device) {
        destroy_pipeline(device, self.temporal_pipeline);
        destroy_pipeline(device, self.radial_pipeline);
        destroy_pipeline(device, self.prefilter_pipeline);
        destroy_pipeline(device, self.mask_pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.shadow_sampler);
        destroy_sampler(device, self.data_sampler);
        destroy_sampler(device, self.color_sampler);
        destroy_pipeline_layout(device, self.temporal_pipeline_layout);
        destroy_pipeline_layout(device, self.source_pipeline_layout);
        destroy_pipeline_layout(device, self.mask_pipeline_layout);
        destroy_descriptor_set_layout(device, self.temporal_set_layout);
        destroy_descriptor_set_layout(device, self.source_set_layout);
        destroy_descriptor_set_layout(device, self.mask_set_layout);
    }
}

fn camera_params(camera: CameraSnapshot, extent: vk::Extent2D) -> [f32; 4] {
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

fn depth_params(camera: CameraSnapshot) -> [f32; 4] {
    let near = camera.near.max(0.0001);
    let far = camera.far.max(near + 0.001);
    [near, far, near * far, far - near]
}

fn target_params(extent: vk::Extent2D, camera: CameraSnapshot) -> [f32; 4] {
    let width = extent.width.max(1) as f32;
    let height = extent.height.max(1) as f32;
    let aspect = height / width;
    let tan_half_fov = (camera.fov_y_radians * 0.5).tan();
    let focal = if tan_half_fov.is_finite() && tan_half_fov > 0.0001 {
        1.0 / tan_half_fov
    } else {
        1.7320508
    };
    [1.0 / width, 1.0 / height, aspect, focal]
}

fn bloom_params(bloom: BloomQualitySettings) -> [f32; 4] {
    [
        bloom.intensity(),
        bloom.threshold(),
        bloom.radius_pixels(),
        bloom.god_rays_intensity(),
    ]
}

pub(super) fn frame_god_ray_sources(
    camera: CameraSnapshot,
    target_extent: vk::Extent2D,
    quality: RenderQualitySettings,
    directional_light: LightPacket,
    emissive_lights: EmissiveLightUniforms,
) -> [GodRaySource; GOD_RAY_SOURCE_COUNT] {
    let camera_params = camera_params(camera, target_extent);
    let bloom = quality.bloom();
    god_ray_sources(
        camera,
        camera_params,
        bloom,
        directional_light,
        emissive_lights,
    )
}

fn god_ray_sources(
    camera: CameraSnapshot,
    camera_params: [f32; 4],
    bloom: BloomQualitySettings,
    directional_light: LightPacket,
    emissive_lights: EmissiveLightUniforms,
) -> [GodRaySource; GOD_RAY_SOURCE_COUNT] {
    let mut sources = [GodRaySource::default(); GOD_RAY_SOURCE_COUNT];
    if bloom.god_rays_intensity() <= 0.0 {
        return sources;
    }

    let forward = normalize_or(sub3(camera.target, camera.eye), [0.0, 0.0, -1.0]);
    let right = normalize_or(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);

    if let Some(source) =
        directional_god_ray_source(camera_params, forward, right, up, directional_light)
    {
        push_god_ray_source(&mut sources, source);
    }

    let light_count = (emissive_lights.count[0] as usize).min(MAX_LOCAL_LIGHTS);
    let focal = camera_params[0]
        .abs()
        .max(camera_params[1].abs())
        .max(0.0001);
    let near = camera.near.max(0.001);
    for index in 0..light_count {
        if let Some(source) = local_god_ray_source(
            camera,
            camera_params,
            forward,
            right,
            up,
            focal,
            near,
            emissive_lights,
            index,
        ) {
            push_god_ray_source(&mut sources, source);
        }
    }

    sources
}

fn directional_god_ray_source(
    camera_params: [f32; 4],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    directional_light: LightPacket,
) -> Option<GodRaySource> {
    let light_dir = normalize_or(directional_light.direction, DEFAULT_DIRECTIONAL_LIGHT_DIR);
    let sun_dir = [-light_dir[0], -light_dir[1], -light_dir[2]];
    let forward_alignment = dot3(forward, sun_dir);
    let visibility = smoothstep(-0.08, 0.24, forward_alignment);
    if visibility <= 0.001 {
        return None;
    }

    let inv_depth = 1.0 / forward_alignment.max(0.0001);
    let ndc_x = dot3(right, sun_dir) * camera_params[0] * inv_depth;
    let ndc_y = dot3(up, sun_dir) * camera_params[1] * inv_depth;
    let uv = [ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5];
    let screen_fade = screen_presence(uv, 1.15);
    if screen_fade <= 0.001 {
        return None;
    }

    let color = [
        directional_light.color[0] * directional_light.intensity,
        directional_light.color[1] * directional_light.intensity,
        directional_light.color[2] * directional_light.intensity,
    ];
    let brightness = max3(color).sqrt().clamp(0.0, 1.35);
    let chroma = chroma3(color, [1.0, 0.88, 0.72]);

    Some(GodRaySource {
        source: [
            uv[0],
            uv[1],
            0.018,
            visibility * screen_fade * brightness * 0.55,
        ],
        color: [chroma[0], chroma[1], chroma[2], 1.0],
    })
}

fn local_god_ray_source(
    camera: CameraSnapshot,
    camera_params: [f32; 4],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    focal: f32,
    near: f32,
    emissive_lights: EmissiveLightUniforms,
    index: usize,
) -> Option<GodRaySource> {
    let position_radius = emissive_lights.position_radius[index];
    let color = emissive_lights.color[index];
    let position = [position_radius[0], position_radius[1], position_radius[2]];
    let delta = sub3(position, camera.eye);
    let depth = dot3(forward, delta);
    if !depth.is_finite() || depth <= near {
        return None;
    }

    let inv_depth = 1.0 / depth;
    let ndc_x = dot3(right, delta) * camera_params[0] * inv_depth;
    let ndc_y = dot3(up, delta) * camera_params[1] * inv_depth;
    let uv = [ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5];
    let screen_fade = screen_presence(uv, 0.18);
    if screen_fade <= 0.001 {
        return None;
    }

    let range = position_radius[3].max(0.001);
    let distance = dot3(delta, delta).sqrt();
    let distance_fade = 1.0 - smoothstep(range * 0.72, range * 1.20, distance);
    let light_color = [color[0], color[1], color[2]];
    let brightness = color[3].max(max3(light_color)).max(0.0);
    let strength = (brightness * 0.20).sqrt().clamp(0.0, 1.4) * distance_fade * screen_fade;
    if strength <= 0.001 || !strength.is_finite() {
        return None;
    }

    let source_radius = emissive_lights.direction_radius[index][3]
        .max(emissive_lights.size_kind[index][0].max(emissive_lights.size_kind[index][1]))
        .max(range * 0.010)
        .min(range * 0.28);
    let screen_radius = (source_radius * focal * 0.5 * inv_depth).clamp(0.0045, 0.070);
    let chroma = chroma3(light_color, [1.0, 0.82, 0.55]);

    Some(GodRaySource {
        source: [uv[0], uv[1], screen_radius, strength],
        color: [chroma[0], chroma[1], chroma[2], 0.0],
    })
}

fn push_god_ray_source(sources: &mut [GodRaySource; GOD_RAY_SOURCE_COUNT], source: GodRaySource) {
    if source.source[3] <= 0.0 {
        return;
    }
    if source.source[3] > sources[0].source[3] {
        sources[1] = sources[0];
        sources[0] = source;
    } else if source.source[3] > sources[1].source[3] {
        sources[1] = source;
    }
}

fn screen_presence(uv: [f32; 2], margin: f32) -> f32 {
    let offscreen_x = ((uv[0] - 0.5).abs() - (0.5 + margin)).max(0.0);
    let offscreen_y = ((uv[1] - 0.5).abs() - (0.5 + margin)).max(0.0);

    1.0 - smoothstep(0.02, 0.42 + margin, offscreen_x.hypot(offscreen_y))
}

fn chroma3(color: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let peak = max3(color);
    if !peak.is_finite() || peak <= 0.0001 {
        return fallback;
    }

    [color[0] / peak, color[1] / peak, color[2] / peak]
}

fn max3(value: [f32; 3]) -> f32 {
    value[0].max(value[1]).max(value[2])
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn create_mask_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        sampler_binding(MASK_SCENE_COLOR_BINDING),
        sampler_binding(MASK_SCENE_DEPTH_BINDING),
        sampler_binding(MASK_TRANSPARENT_METADATA_BINDING),
        sampler_binding(MASK_DIRECTIONAL_SHADOW_BINDING),
        sampler_binding(MASK_TAA_DEPTH_0_BINDING),
        sampler_binding(MASK_TAA_DEPTH_1_BINDING),
        sampler_binding(MASK_TRANSLUCENT_SHADOW_0_BINDING),
        sampler_binding(MASK_TRANSLUCENT_SHADOW_1_BINDING),
        sampler_binding(MASK_TRANSLUCENT_SHADOW_2_BINDING),
        sampler_binding(MASK_TRANSLUCENT_SHADOW_3_BINDING),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_mask_pipeline_layout(
    device: &Device,
    mask_set_layout: vk::DescriptorSetLayout,
    frame_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [mask_set_layout, frame_set_layout];
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<GodRayPushConstants>() as u32);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);

    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_source_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [sampler_binding(SOURCE_BINDING)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_temporal_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        sampler_binding(TEMPORAL_CURRENT_BINDING),
        sampler_binding(TEMPORAL_HISTORY_BINDING),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn sampler_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
}

fn create_pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [set_layout];
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<GodRayPushConstants>() as u32);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);

    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_sampler(device: &Device, filter: vk::Filter) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);

    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_shadow_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .min_lod(0.0)
        .max_lod(0.0);

    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(20);
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(7)
        .pool_sizes(&pool_sizes);

    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

fn allocate_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VulkanError> {
    let mut sets = allocate_descriptor_sets(device, descriptor_pool, set_layout, 1)?;
    Ok(sets.remove(0))
}

fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    count: u32,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    let layouts = vec![set_layout; count as usize];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

fn update_mask_descriptor(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
    shadow_sampler: vk::Sampler,
    scene_color_view: vk::ImageView,
    scene_depth_view: vk::ImageView,
    transparent_view: vk::ImageView,
    directional_shadow_view: vk::ImageView,
    translucent_shadow_views: [vk::ImageView; 4],
    taa_depth_history_views: [vk::ImageView; 2],
) {
    let color_info = [image_info(color_sampler, scene_color_view)];
    let depth_info = [image_info(data_sampler, scene_depth_view)];
    let transparent_info = [image_info(data_sampler, transparent_view)];
    let shadow_info = [depth_image_info(shadow_sampler, directional_shadow_view)];
    let taa_depth_0_info = [image_info(data_sampler, taa_depth_history_views[0])];
    let taa_depth_1_info = [image_info(data_sampler, taa_depth_history_views[1])];
    let translucent_infos = translucent_shadow_views.map(|view| image_info(data_sampler, view));
    let writes = [
        descriptor_write(descriptor_set, MASK_SCENE_COLOR_BINDING, &color_info),
        descriptor_write(descriptor_set, MASK_SCENE_DEPTH_BINDING, &depth_info),
        descriptor_write(
            descriptor_set,
            MASK_TRANSPARENT_METADATA_BINDING,
            &transparent_info,
        ),
        descriptor_write(
            descriptor_set,
            MASK_DIRECTIONAL_SHADOW_BINDING,
            &shadow_info,
        ),
        descriptor_write(descriptor_set, MASK_TAA_DEPTH_0_BINDING, &taa_depth_0_info),
        descriptor_write(descriptor_set, MASK_TAA_DEPTH_1_BINDING, &taa_depth_1_info),
        descriptor_write(
            descriptor_set,
            MASK_TRANSLUCENT_SHADOW_0_BINDING,
            std::slice::from_ref(&translucent_infos[0]),
        ),
        descriptor_write(
            descriptor_set,
            MASK_TRANSLUCENT_SHADOW_1_BINDING,
            std::slice::from_ref(&translucent_infos[1]),
        ),
        descriptor_write(
            descriptor_set,
            MASK_TRANSLUCENT_SHADOW_2_BINDING,
            std::slice::from_ref(&translucent_infos[2]),
        ),
        descriptor_write(
            descriptor_set,
            MASK_TRANSLUCENT_SHADOW_3_BINDING,
            std::slice::from_ref(&translucent_infos[3]),
        ),
    ];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn update_source_descriptor(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    image_view: vk::ImageView,
) {
    let info = [image_info(sampler, image_view)];
    let writes = [descriptor_write(descriptor_set, SOURCE_BINDING, &info)];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn update_temporal_descriptors(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    sampler: vk::Sampler,
    blur_view: vk::ImageView,
    history_views: [vk::ImageView; 2],
) {
    for (write_index, &set) in descriptor_sets.iter().enumerate() {
        let read_index = 1 - write_index;
        let current_info = [image_info(sampler, blur_view)];
        let history_info = [image_info(sampler, history_views[read_index])];
        let writes = [
            descriptor_write(set, TEMPORAL_CURRENT_BINDING, &current_info),
            descriptor_write(set, TEMPORAL_HISTORY_BINDING, &history_info),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

fn descriptor_write<'a>(
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

fn image_info(sampler: vk::Sampler, image_view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(image_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

fn depth_image_info(sampler: vk::Sampler, image_view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(image_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
}

fn create_god_ray_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    fragment_shader_bytes: &[u8],
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

fn take_created_ref<T: Copy>(value: Option<T>) -> Result<T, VulkanError> {
    value.ok_or_else(|| VulkanError::GraphCompile("god-ray build order is invalid".to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_god_ray_tracks_global_light_direction_and_color() {
        let centered =
            LightPacket::new(1.0).with_direction_and_color([0.0, 0.0, -1.0], [1.0, 0.2, 0.1]);
        let shifted =
            LightPacket::new(1.0).with_direction_and_color([-0.5, 0.0, -1.0], [0.1, 0.2, 1.0]);
        let camera_params = [1.0, 1.0, 1.0, 1.0];
        let forward = [0.0, 0.0, 1.0];
        let right = [1.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];

        let centered_source =
            directional_god_ray_source(camera_params, forward, right, up, centered)
                .expect("forward global light should produce a god-ray source");
        let shifted_source = directional_god_ray_source(camera_params, forward, right, up, shifted)
            .expect("shifted global light should produce a god-ray source");

        assert!((centered_source.source[0] - 0.5).abs() < 1.0e-6);
        assert!(shifted_source.source[0] > centered_source.source[0]);
        assert!(centered_source.color[0] > centered_source.color[2]);
        assert!(shifted_source.color[2] > shifted_source.color[0]);
    }

    #[test]
    fn volumetric_push_keeps_medium_range_independent_of_raster_far_clip() {
        let camera = CameraSnapshot::perspective(
            [0.0, 1.8, 5.0],
            [0.0, 1.5, 4.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.03,
            32.0,
        )
        .expect("test camera should be valid");
        let quality = RenderQualitySettings::high_quality();
        let base = GodRayPushConstants::new(
            camera,
            vk::Extent2D {
                width: 640,
                height: 360,
            },
            quality,
            LightPacket::new(1.0),
            EmissiveLightUniforms::disabled(),
            false,
            false,
            12,
            [0.0, 0.0],
            0,
            false,
            TemporalEffectFrame::default(),
        );
        assert!(base.features[2] > 0.5);
        assert!((base.source0[3] - 160.0).abs() < 0.001);

        let mut moved = TemporalEffectFrame::default();
        moved.sample_phase = 0.5;
        moved.light_motion = 0.75;
        let moved = GodRayPushConstants::new(
            camera,
            vk::Extent2D {
                width: 640,
                height: 360,
            },
            quality,
            LightPacket::new(1.0),
            EmissiveLightUniforms::disabled(),
            false,
            false,
            12,
            [0.0, 0.0],
            0,
            false,
            moved,
        );
        assert_ne!(base.source1[3], moved.source1[3]);
    }
}
