use ash::{Device, Instance, khr, vk};

use crate::protocol::{CameraSnapshot, FrameSnapshot, NonZeroExtent, RenderQualitySettings};

use super::{
    VulkanDevice, VulkanError,
    bloom::BloomPipeline,
    god_rays::GodRaysPipeline,
    mesh::{MAX_LOCAL_LIGHTS, MeshPassResources, MeshPipelineSet, VulkanMeshStore},
    post::PostPipeline,
    stable_csm_depth::{STABLE_CSM_DEPTH_FORMAT, StableCsmDepthArray},
    swapchain_pass::{
        create_bloom_downsample_render_pass, create_bloom_upsample_render_pass,
        create_god_ray_render_pass, create_local_shadow_framebuffer,
        create_local_shadow_render_pass, create_post_framebuffer, create_post_render_pass,
        create_scene_fast_framebuffer, create_scene_fast_render_pass, create_scene_framebuffer,
        create_scene_render_pass, create_shadow_framebuffer, create_shadow_render_pass,
        create_translucent_shadow_framebuffer, create_translucent_shadow_render_pass,
        destroy_framebuffer, destroy_render_pass,
    },
    swapchain_target::{
        ColorTarget, DepthCubeTarget, DepthTarget, create_color_target, create_depth_cube_target,
        create_depth_target, destroy_color_target, destroy_depth_cube_target, destroy_depth_target,
        destroy_image_view, initialize_depth_cube_shader_read_image,
        initialize_mipped_color_shader_read_image,
    },
    taa::{TaaFrameInfo, TemporalAntiAliasing},
};
use crate::renderer::graph::{
    BLOOM_MIP_COUNT, FrameGraphInitialStates, GOD_RAY_HISTORY_COUNT, GOD_RAY_HISTORY_RESOURCES,
    GOD_RAY_TEMPORAL_PASS, GraphResource, ResourceState, SHADOW_CASCADE_COUNT,
    SHADOW_CASCADE_RESOURCES, TRANSLUCENT_SHADOW_RESOURCES,
};

pub(super) const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// Stable CSM stores one depth layer per cascade. There is no temporal direction/sample axis.
pub(super) const STABLE_CSM_LAYER_COUNT: usize = SHADOW_CASCADE_COUNT;

// Scene color is sampled by the post shader before tonemapping.
// Do not use the swapchain format here: B8G8R8A8_SRGB/RGBA8_UNORM clamps HDR PBR
// lighting to [0, 1], which makes close-camera specular highlights and reflections collapse.
const SCENE_COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Opaque metadata fits in UNORM8: oct-encoded normal.xy, roughness, reflectance.
const SCENE_NORMAL_ROUGHNESS_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

// Transparent metadata stores alpha as 1.0 + gl_FragCoord.z to mark valid pixels.
// UNORM8 would clamp that to 1.0, so post.frag's `transparent.w > 1.0` test can never work.
const SCENE_TRANSPARENT_NORMAL_ROUGHNESS_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// RGB stores the nearest translucent layer's transmittance while alpha stores its shadow depth.
// FP16 avoids the 256 depth levels of UNORM8, which visibly terrace thin or sloped receivers.
// This sampled color-attachment format is already required by SCENE_COLOR_FORMAT.
const TRANSLUCENT_SHADOW_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const FALLBACK_SHADOW_TRANSMITTANCE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Verifies the optimal-tiling features required by the directional shadow pipeline.
///
/// D16 must support linear-filtered sampling because the single comparison lookup intentionally
/// relies on the hardware 2x2 PCF footprint.
pub(super) fn validate_shadow_format_support(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(), VulkanError> {
    let requirements = [
        (
            STABLE_CSM_DEPTH_FORMAT,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            "D16 stable CSM directional shadow array",
        ),
        (
            DEPTH_FORMAT,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::FormatFeatureFlags::SAMPLED_IMAGE,
            "D32 local-light shadow depth",
        ),
        (
            TRANSLUCENT_SHADOW_FORMAT,
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE,
            "FP16 translucent shadow",
        ),
    ];

    for (format, required, label) in requirements {
        let properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        let available = properties.optimal_tiling_features;
        if !available.contains(required) {
            tracing::error!(
                ?format,
                ?required,
                ?available,
                label = %label,
                "required Vulkan shadow format features are unavailable"
            );
            return Err(VulkanError::Vk(vk::Result::ERROR_FORMAT_NOT_SUPPORTED));
        }
    }

    Ok(())
}

pub(super) struct VulkanSwapchain {
    pub(super) handle: vk::SwapchainKHR,
    pub(super) extent: NonZeroExtent,
    pub(super) format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    present_mode: vk::PresentModeKHR,
    transfer_src_supported: bool,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    image_states: Vec<ResourceState>,
    scene: SceneTargets,
    scene_render_pass: vk::RenderPass,
    scene_fast_render_pass: vk::RenderPass,
    bloom_downsample_render_pass: vk::RenderPass,
    bloom_upsample_render_pass: vk::RenderPass,
    god_ray_render_pass: vk::RenderPass,
    post_render_pass: vk::RenderPass,
    mesh_pipeline: MeshPipelineSet,
    mesh_fast_pipeline: MeshPipelineSet,
    transparent_mesh_pipeline: MeshPipelineSet,
    transparent_mesh_fast_pipeline: MeshPipelineSet,
    bloom_pipeline: BloomPipeline,
    god_rays_pipeline: GodRaysPipeline,
    post_pipeline: PostPipeline,
    taa: TemporalAntiAliasing,
    scene_framebuffer: vk::Framebuffer,
    scene_fast_framebuffer: vk::Framebuffer,
    bloom: BloomTargets,
    god_rays: GodRayTargets,
    bloom_downsample_framebuffers: Vec<vk::Framebuffer>,
    bloom_upsample_framebuffers: Vec<vk::Framebuffer>,
    god_ray_framebuffers: GodRayFramebuffers,
    post_framebuffers: Vec<vk::Framebuffer>,
}

pub(super) struct ShadowResources {
    directional_depth: StableCsmDepthArray,
    shadow_framebuffers: [vk::Framebuffer; STABLE_CSM_LAYER_COUNT],
    directional_state: ResourceState,
    shadow_extent: NonZeroExtent,
    translucent_depth: DepthTarget,
    cascades: [ShadowCascade; SHADOW_CASCADE_COUNT],
    local: Vec<LocalShadowCube>,
    shadow_render_pass: vk::RenderPass,
    local_shadow_render_pass: vk::RenderPass,
    translucent_render_pass: vk::RenderPass,
    mesh_pass_resources: MeshPassResources,
    shadow_pipeline: MeshPipelineSet,
    local_shadow_pipeline: MeshPipelineSet,
    translucent_pipeline: MeshPipelineSet,
}

pub(super) struct ShadowSamplerFallback {
    transmittance: ColorTarget,
    directional_depth: StableCsmDepthArray,
    local_depth: DepthCubeTarget,
    mesh_pass_resources: MeshPassResources,
}

struct ShadowCascade {
    transmittance: ColorTarget,
    transmittance_state: ResourceState,
    translucent_framebuffer: vk::Framebuffer,
}

struct LocalShadowCube {
    depth: DepthCubeTarget,
    framebuffers: [vk::Framebuffer; 6],
    extent: NonZeroExtent,
}

struct SceneTargets {
    color: ColorTarget,
    normal_roughness: ColorTarget,
    transparent_normal_roughness: ColorTarget,
    depth: DepthTarget,
    color_state: ResourceState,
    normal_roughness_state: ResourceState,
    transparent_normal_roughness_state: ResourceState,
    depth_state: ResourceState,
}

struct BloomLevelTarget {
    color: ColorTarget,
    extent: NonZeroExtent,
    state: ResourceState,
}

struct BloomTargets {
    levels: Vec<BloomLevelTarget>,
}

struct GodRayTargets {
    mask: GodRayTarget,
    prefilter: GodRayTarget,
    blur: GodRayTarget,
    histories: Vec<GodRayTarget>,
    history_write_index: usize,
    history_valid: bool,
}

struct GodRayTarget {
    color: ColorTarget,
    extent: NonZeroExtent,
    state: ResourceState,
}

struct GodRayTargetSet {
    mask: GodRayTarget,
    prefilter: GodRayTarget,
    blur: GodRayTarget,
    histories: Vec<GodRayTarget>,
}

#[derive(Default)]
struct GodRayFramebuffers {
    mask: vk::Framebuffer,
    prefilter: vk::Framebuffer,
    radial: vk::Framebuffer,
    histories: Vec<vk::Framebuffer>,
}

impl SceneTargets {
    /// Groups scene attachments and their graph states behind one runtime-owned boundary.
    fn new(
        color: ColorTarget,
        normal_roughness: ColorTarget,
        transparent_normal_roughness: ColorTarget,
        depth: DepthTarget,
    ) -> Self {
        Self {
            color,
            normal_roughness,
            transparent_normal_roughness,
            depth,
            color_state: ResourceState::Undefined,
            normal_roughness_state: ResourceState::Undefined,
            transparent_normal_roughness_state: ResourceState::Undefined,
            depth_state: ResourceState::Undefined,
        }
    }

    /// Returns the tracked graph states for the scene metadata attachments and depth.
    fn graph_states(&self) -> (ResourceState, ResourceState, ResourceState, ResourceState) {
        (
            self.color_state,
            self.normal_roughness_state,
            self.transparent_normal_roughness_state,
            self.depth_state,
        )
    }

    /// Applies graph final states back into the tracked scene attachment states.
    fn apply_graph_final_states(&mut self, plan: &crate::renderer::graph::FrameGraphPlan) {
        if let Some(state) = plan.final_state_for(GraphResource::SceneColor) {
            self.color_state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::SceneNormalRoughness) {
            self.normal_roughness_state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::SceneTransparentNormalRoughness) {
            self.transparent_normal_roughness_state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::SceneDepth) {
            self.depth_state = state;
        }
    }

    /// Resolves one scene graph resource to its image and aspect range.
    fn graph_image(&self, resource: GraphResource) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        match resource {
            GraphResource::SceneColor => Some((self.color.image, vk::ImageAspectFlags::COLOR)),
            GraphResource::SceneNormalRoughness => {
                Some((self.normal_roughness.image, vk::ImageAspectFlags::COLOR))
            }
            GraphResource::SceneTransparentNormalRoughness => Some((
                self.transparent_normal_roughness.image,
                vk::ImageAspectFlags::COLOR,
            )),
            GraphResource::SceneDepth => Some((self.depth.image, vk::ImageAspectFlags::DEPTH)),
            _ => None,
        }
    }

    /// Destroys every scene attachment after the swapchain stops referencing them.
    fn destroy(self, device: &Device) {
        destroy_depth_target(device, self.depth);
        destroy_color_target(device, self.transparent_normal_roughness);
        destroy_color_target(device, self.normal_roughness);
        destroy_color_target(device, self.color);
    }
}

impl BloomTargets {
    fn new(levels: Vec<BloomLevelTarget>) -> Self {
        assert_eq!(
            levels.len(),
            BLOOM_MIP_COUNT,
            "bloom target chain must match graph resource count"
        );
        Self { levels }
    }

    fn graph_states(&self) -> [ResourceState; BLOOM_MIP_COUNT] {
        std::array::from_fn(|index| self.levels[index].state)
    }

    fn apply_graph_final_states(&mut self, plan: &crate::renderer::graph::FrameGraphPlan) {
        for (index, level) in self.levels.iter_mut().enumerate() {
            if let Some(state) =
                plan.final_state_for(crate::renderer::graph::BLOOM_MIP_RESOURCES[index])
            {
                level.state = state;
            }
        }
    }

    fn graph_image(&self, resource: GraphResource) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        let index = resource.bloom_mip()?;
        self.levels
            .get(index)
            .map(|level| (level.color.image, vk::ImageAspectFlags::COLOR))
    }

    fn extent_2d(&self, mip_index: usize) -> Result<vk::Extent2D, VulkanError> {
        let level =
            self.levels
                .get(mip_index)
                .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                    index: mip_index,
                    count: self.levels.len(),
                })?;
        Ok(vk::Extent2D {
            width: level.extent.width(),
            height: level.extent.height(),
        })
    }

    fn destroy(self, device: &Device) {
        for level in self.levels.into_iter().rev() {
            destroy_color_target(device, level.color);
        }
    }
}

impl GodRayTargets {
    fn new(
        mask: GodRayTarget,
        prefilter: GodRayTarget,
        blur: GodRayTarget,
        histories: Vec<GodRayTarget>,
    ) -> Self {
        assert_eq!(
            histories.len(),
            GOD_RAY_HISTORY_COUNT,
            "god-ray history target count must match graph resource count"
        );
        Self {
            mask,
            prefilter,
            blur,
            histories,
            history_write_index: 0,
            history_valid: false,
        }
    }

    fn history_states(&self) -> [ResourceState; GOD_RAY_HISTORY_COUNT] {
        std::array::from_fn(|index| self.histories[index].state)
    }

    fn apply_graph_final_states(&mut self, plan: &crate::renderer::graph::FrameGraphPlan) {
        if let Some(state) = plan.final_state_for(GraphResource::GodRayMask) {
            self.mask.state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::GodRayPrefilter) {
            self.prefilter.state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::GodRayBlur) {
            self.blur.state = state;
        }
        for (index, history) in self.histories.iter_mut().enumerate() {
            if let Some(state) = plan.final_state_for(GOD_RAY_HISTORY_RESOURCES[index]) {
                history.state = state;
            }
        }
        if plan
            .passes()
            .iter()
            .any(|pass| pass.name() == GOD_RAY_TEMPORAL_PASS)
        {
            self.history_valid = true;
            self.history_write_index = 1 - self.history_write_index;
        }
    }

    fn graph_image(&self, resource: GraphResource) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        let color = match resource {
            GraphResource::GodRayMask => Some(&self.mask.color),
            GraphResource::GodRayPrefilter => Some(&self.prefilter.color),
            GraphResource::GodRayBlur => Some(&self.blur.color),
            _ => resource
                .god_ray_history()
                .and_then(|index| self.histories.get(index).map(|target| &target.color)),
        }?;

        Some((color.image, vk::ImageAspectFlags::COLOR))
    }

    fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.mask.extent.width(),
            height: self.mask.extent.height(),
        }
    }

    fn history_write_index(&self) -> usize {
        self.history_write_index
    }

    fn history_valid(&self) -> bool {
        self.history_valid
    }

    /// Drops temporal rays when the source model changes (legacy radial versus volumetric).
    fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.history_write_index = 0;
    }

    fn destroy(self, device: &Device) {
        for history in self.histories.into_iter().rev() {
            destroy_color_target(device, history.color);
        }
        destroy_color_target(device, self.blur.color);
        destroy_color_target(device, self.prefilter.color);
        destroy_color_target(device, self.mask.color);
    }
}

impl GodRayFramebuffers {
    fn count(&self) -> usize {
        3 + self.histories.len()
    }

    fn destroy(self, device: &Device) {
        for framebuffer in self.histories {
            destroy_framebuffer(device, framebuffer);
        }
        destroy_framebuffer(device, self.radial);
        destroy_framebuffer(device, self.prefilter);
        destroy_framebuffer(device, self.mask);
    }
}

pub(super) struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SwapchainConfig {
    extent: NonZeroExtent,
    image_count: u32,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    present_mode: vk::PresentModeKHR,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
    transfer_src_supported: bool,
}

impl SwapchainConfig {
    /// Returns the surface-capability-resolved drawable extent.
    pub(super) fn extent(&self) -> NonZeroExtent {
        self.extent
    }
}

struct SwapchainBuild<'a> {
    device: &'a VulkanDevice,
    handle: vk::SwapchainKHR,
    extent: NonZeroExtent,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    present_mode: vk::PresentModeKHR,
    transfer_src_supported: bool,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    scene_color: Option<ColorTarget>,
    scene_normal_roughness: Option<ColorTarget>,
    scene_transparent_normal_roughness: Option<ColorTarget>,
    scene_depth: Option<DepthTarget>,
    bloom_levels: Vec<BloomLevelTarget>,
    god_ray_mask: Option<GodRayTarget>,
    god_ray_prefilter: Option<GodRayTarget>,
    god_ray_blur: Option<GodRayTarget>,
    god_ray_histories: Vec<GodRayTarget>,
    scene_render_pass: Option<vk::RenderPass>,
    scene_fast_render_pass: Option<vk::RenderPass>,
    bloom_downsample_render_pass: Option<vk::RenderPass>,
    bloom_upsample_render_pass: Option<vk::RenderPass>,
    god_ray_render_pass: Option<vk::RenderPass>,
    post_render_pass: Option<vk::RenderPass>,
    mesh_pipeline: Option<MeshPipelineSet>,
    mesh_fast_pipeline: Option<MeshPipelineSet>,
    transparent_mesh_pipeline: Option<MeshPipelineSet>,
    transparent_mesh_fast_pipeline: Option<MeshPipelineSet>,
    bloom_pipeline: Option<BloomPipeline>,
    god_rays_pipeline: Option<GodRaysPipeline>,
    post_pipeline: Option<PostPipeline>,
    taa: Option<TemporalAntiAliasing>,
    scene_framebuffer: Option<vk::Framebuffer>,
    scene_fast_framebuffer: Option<vk::Framebuffer>,
    bloom_downsample_framebuffers: Vec<vk::Framebuffer>,
    bloom_upsample_framebuffers: Vec<vk::Framebuffer>,
    god_ray_framebuffers: GodRayFramebuffers,
    post_framebuffers: Vec<vk::Framebuffer>,
    finished: bool,
}

impl<'a> SwapchainBuild<'a> {
    /// Captures partially-created swapchain resources so failure cleanup stays in one place.
    fn new(
        device: &'a VulkanDevice,
        handle: vk::SwapchainKHR,
        config: SwapchainConfig,
        images: Vec<vk::Image>,
        image_views: Vec<vk::ImageView>,
    ) -> Self {
        Self {
            device,
            handle,
            extent: config.extent,
            format: config.format,
            color_space: config.color_space,
            present_mode: config.present_mode,
            transfer_src_supported: config.transfer_src_supported,
            images,
            image_views,
            scene_color: None,
            scene_normal_roughness: None,
            scene_transparent_normal_roughness: None,
            scene_depth: None,
            bloom_levels: Vec::new(),
            god_ray_mask: None,
            god_ray_prefilter: None,
            god_ray_blur: None,
            god_ray_histories: Vec::new(),
            scene_render_pass: None,
            scene_fast_render_pass: None,
            bloom_downsample_render_pass: None,
            bloom_upsample_render_pass: None,
            god_ray_render_pass: None,
            post_render_pass: None,
            mesh_pipeline: None,
            mesh_fast_pipeline: None,
            transparent_mesh_pipeline: None,
            transparent_mesh_fast_pipeline: None,
            bloom_pipeline: None,
            god_rays_pipeline: None,
            post_pipeline: None,
            taa: None,
            scene_framebuffer: None,
            scene_fast_framebuffer: None,
            bloom_downsample_framebuffers: Vec::new(),
            bloom_upsample_framebuffers: Vec::new(),
            god_ray_framebuffers: GodRayFramebuffers::default(),
            post_framebuffers: Vec::new(),
            finished: false,
        }
    }

    /// Moves completed resources into the runtime swapchain and disables failure cleanup.
    fn finish(mut self) -> VulkanSwapchain {
        let image_states = vec![ResourceState::Undefined; self.images.len()];
        let swapchain = VulkanSwapchain {
            handle: self.handle,
            extent: self.extent,
            format: self.format,
            color_space: self.color_space,
            present_mode: self.present_mode,
            transfer_src_supported: self.transfer_src_supported,
            images: std::mem::take(&mut self.images),
            image_views: std::mem::take(&mut self.image_views),
            image_states,
            scene: SceneTargets::new(
                take_created(&mut self.scene_color, "scene color"),
                take_created(&mut self.scene_normal_roughness, "scene normal roughness"),
                take_created(
                    &mut self.scene_transparent_normal_roughness,
                    "scene transparent normal roughness",
                ),
                take_created(&mut self.scene_depth, "scene depth"),
            ),
            scene_render_pass: take_created(&mut self.scene_render_pass, "scene render pass"),
            scene_fast_render_pass: take_created(
                &mut self.scene_fast_render_pass,
                "fast scene render pass",
            ),
            bloom_downsample_render_pass: take_created(
                &mut self.bloom_downsample_render_pass,
                "bloom downsample render pass",
            ),
            bloom_upsample_render_pass: take_created(
                &mut self.bloom_upsample_render_pass,
                "bloom upsample render pass",
            ),
            god_ray_render_pass: take_created(&mut self.god_ray_render_pass, "god-ray render pass"),
            post_render_pass: take_created(&mut self.post_render_pass, "post render pass"),
            mesh_pipeline: take_created(&mut self.mesh_pipeline, "mesh pipeline"),
            mesh_fast_pipeline: take_created(&mut self.mesh_fast_pipeline, "fast mesh pipeline"),
            transparent_mesh_pipeline: take_created(
                &mut self.transparent_mesh_pipeline,
                "transparent mesh pipeline",
            ),
            transparent_mesh_fast_pipeline: take_created(
                &mut self.transparent_mesh_fast_pipeline,
                "fast transparent mesh pipeline",
            ),
            bloom_pipeline: take_created(&mut self.bloom_pipeline, "bloom pipeline"),
            god_rays_pipeline: take_created(&mut self.god_rays_pipeline, "god-ray pipeline"),
            post_pipeline: take_created(&mut self.post_pipeline, "post pipeline"),
            taa: take_created(&mut self.taa, "temporal anti-aliasing"),
            scene_framebuffer: take_created(&mut self.scene_framebuffer, "scene framebuffer"),
            scene_fast_framebuffer: take_created(
                &mut self.scene_fast_framebuffer,
                "fast scene framebuffer",
            ),
            bloom: BloomTargets::new(std::mem::take(&mut self.bloom_levels)),
            god_rays: GodRayTargets::new(
                take_created(&mut self.god_ray_mask, "god-ray mask target"),
                take_created(&mut self.god_ray_prefilter, "god-ray prefilter target"),
                take_created(&mut self.god_ray_blur, "god-ray blur target"),
                std::mem::take(&mut self.god_ray_histories),
            ),
            bloom_downsample_framebuffers: std::mem::take(&mut self.bloom_downsample_framebuffers),
            bloom_upsample_framebuffers: std::mem::take(&mut self.bloom_upsample_framebuffers),
            god_ray_framebuffers: std::mem::take(&mut self.god_ray_framebuffers),
            post_framebuffers: std::mem::take(&mut self.post_framebuffers),
        };
        self.finished = true;
        swapchain
    }
}

impl Drop for SwapchainBuild<'_> {
    /// Releases partially-created swapchain resources when creation exits early.
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        self.device
            .destroy_framebuffers(std::mem::take(&mut self.post_framebuffers));
        std::mem::take(&mut self.god_ray_framebuffers).destroy(&self.device.device);
        self.device
            .destroy_framebuffers(std::mem::take(&mut self.bloom_upsample_framebuffers));
        self.device
            .destroy_framebuffers(std::mem::take(&mut self.bloom_downsample_framebuffers));
        if let Some(framebuffer) = self.scene_fast_framebuffer.take() {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(framebuffer) = self.scene_framebuffer.take() {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(pipeline) = self.post_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.god_rays_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.bloom_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(taa) = self.taa.take() {
            taa.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.mesh_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.mesh_fast_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.transparent_mesh_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.transparent_mesh_fast_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        for render_pass in [
            self.post_render_pass.take(),
            self.god_ray_render_pass.take(),
            self.bloom_upsample_render_pass.take(),
            self.bloom_downsample_render_pass.take(),
            self.scene_fast_render_pass.take(),
            self.scene_render_pass.take(),
        ]
        .into_iter()
        .flatten()
        {
            destroy_render_pass(&self.device.device, render_pass);
        }
        if let Some(depth) = self.scene_depth.take() {
            destroy_depth_target(&self.device.device, depth);
        }
        for level in std::mem::take(&mut self.bloom_levels).into_iter().rev() {
            destroy_color_target(&self.device.device, level.color);
        }
        for target in std::mem::take(&mut self.god_ray_histories)
            .into_iter()
            .rev()
        {
            destroy_color_target(&self.device.device, target.color);
        }
        if let Some(target) = self.god_ray_blur.take() {
            destroy_color_target(&self.device.device, target.color);
        }
        if let Some(target) = self.god_ray_prefilter.take() {
            destroy_color_target(&self.device.device, target.color);
        }
        if let Some(target) = self.god_ray_mask.take() {
            destroy_color_target(&self.device.device, target.color);
        }
        if let Some(normal_roughness) = self.scene_normal_roughness.take() {
            destroy_color_target(&self.device.device, normal_roughness);
        }
        if let Some(transparent_normal_roughness) = self.scene_transparent_normal_roughness.take() {
            destroy_color_target(&self.device.device, transparent_normal_roughness);
        }
        if let Some(color) = self.scene_color.take() {
            destroy_color_target(&self.device.device, color);
        }
        self.device
            .destroy_image_views(std::mem::take(&mut self.image_views));
        self.device.destroy_swapchain_handle(self.handle);
    }
}

struct ShadowBuild<'a> {
    device: &'a VulkanDevice,
    directional_depth: Option<StableCsmDepthArray>,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    translucent_depth: Option<DepthTarget>,
    cascades: Vec<ShadowCascade>,
    local: Vec<LocalShadowCube>,
    shadow_render_pass: Option<vk::RenderPass>,
    local_shadow_render_pass: Option<vk::RenderPass>,
    translucent_render_pass: Option<vk::RenderPass>,
    mesh_pass_resources: Option<MeshPassResources>,
    shadow_pipeline: Option<MeshPipelineSet>,
    local_shadow_pipeline: Option<MeshPipelineSet>,
    translucent_pipeline: Option<MeshPipelineSet>,
    finished: bool,
}

impl<'a> ShadowBuild<'a> {
    /// Captures fixed-size shadow resources while device-level shadow setup is in progress.
    fn new(device: &'a VulkanDevice) -> Self {
        Self {
            device,
            directional_depth: None,
            shadow_framebuffers: Vec::with_capacity(STABLE_CSM_LAYER_COUNT),
            translucent_depth: None,
            cascades: Vec::with_capacity(SHADOW_CASCADE_COUNT),
            local: Vec::with_capacity(MAX_LOCAL_LIGHTS),
            shadow_render_pass: None,
            local_shadow_render_pass: None,
            translucent_render_pass: None,
            mesh_pass_resources: None,
            shadow_pipeline: None,
            local_shadow_pipeline: None,
            translucent_pipeline: None,
            finished: false,
        }
    }

    /// Moves completed shadow resources into the device owner and disables failure cleanup.
    fn finish(mut self) -> ShadowResources {
        let cascades = std::mem::take(&mut self.cascades)
            .try_into()
            .unwrap_or_else(|_| panic!("all shadow cascades must be created before finish"));
        let shadow_framebuffers = std::mem::take(&mut self.shadow_framebuffers)
            .try_into()
            .unwrap_or_else(|_| {
                panic!("all Stable CSM cascade layers must be created before finish")
            });
        if self.local.len() != MAX_LOCAL_LIGHTS {
            panic!("all local shadow cubemaps must be created before finish");
        }
        let resources = ShadowResources {
            directional_depth: take_created(
                &mut self.directional_depth,
                "Stable CSM directional depth array",
            ),
            shadow_framebuffers,
            directional_state: ResourceState::ShaderRead,
            shadow_extent: stable_csm_shadow_extent(self.device.shadow_map_resolution()),
            translucent_depth: take_created(
                &mut self.translucent_depth,
                "shared translucent shadow depth",
            ),
            cascades,
            local: std::mem::take(&mut self.local),
            shadow_render_pass: take_created(&mut self.shadow_render_pass, "shadow render pass"),
            local_shadow_render_pass: take_created(
                &mut self.local_shadow_render_pass,
                "local shadow render pass",
            ),
            translucent_render_pass: take_created(
                &mut self.translucent_render_pass,
                "translucent shadow render pass",
            ),
            mesh_pass_resources: take_created(
                &mut self.mesh_pass_resources,
                "shadow pass resources",
            ),
            shadow_pipeline: take_created(&mut self.shadow_pipeline, "shadow pipeline"),
            local_shadow_pipeline: take_created(
                &mut self.local_shadow_pipeline,
                "local shadow pipeline",
            ),
            translucent_pipeline: take_created(
                &mut self.translucent_pipeline,
                "translucent shadow pipeline",
            ),
        };
        self.finished = true;
        resources
    }
}

impl Drop for ShadowBuild<'_> {
    /// Releases partially-created fixed shadow resources if device shadow setup fails.
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        if let Some(resources) = self.mesh_pass_resources.take() {
            resources.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.translucent_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.shadow_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.local_shadow_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        for cascade in self.cascades.drain(..) {
            destroy_shadow_cascade(&self.device.device, cascade);
        }
        if let Some(depth) = self.translucent_depth.take() {
            destroy_depth_target(&self.device.device, depth);
        }
        for local in self.local.drain(..) {
            destroy_local_shadow_cube(&self.device.device, local);
        }
        for framebuffer in self.shadow_framebuffers.drain(..) {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(depth) = self.directional_depth.take() {
            depth.destroy(&self.device.device);
        }
        for render_pass in [
            self.translucent_render_pass.take(),
            self.local_shadow_render_pass.take(),
            self.shadow_render_pass.take(),
        ]
        .into_iter()
        .flatten()
        {
            destroy_render_pass(&self.device.device, render_pass);
        }
    }
}

/// Takes a required build resource when swapchain construction reaches the success path.
fn take_created<T>(slot: &mut Option<T>, name: &'static str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("{name} must be created before swapchain finish"))
}

impl ShadowSamplerFallback {
    /// Creates one tiny full-light shadow sampler set used before real shadow maps are needed.
    pub(super) fn create(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        meshes: &VulkanMeshStore,
    ) -> Result<Self, VulkanError> {
        let extent = NonZeroExtent::new(1, 1).expect("fallback shadow extent must be non-zero");
        let transmittance = create_color_target(
            device,
            memory_properties,
            extent,
            FALLBACK_SHADOW_TRANSMITTANCE_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )?;
        let local_depth = match create_depth_cube_target(
            device,
            memory_properties,
            extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ) {
            Ok(target) => target,
            Err(error) => {
                destroy_color_target(device, transmittance);
                return Err(error);
            }
        };
        let directional_depth = match StableCsmDepthArray::create(
            device,
            memory_properties,
            extent,
            STABLE_CSM_LAYER_COUNT as u32,
        ) {
            Ok(target) => target,
            Err(error) => {
                destroy_depth_cube_target(device, local_depth);
                destroy_color_target(device, transmittance);
                return Err(error);
            }
        };

        if let Err(error) = initialize_mipped_color_shader_read_image(
            device,
            queue_family_index,
            queue,
            transmittance.image,
            1,
            [1.0; 4],
        ) {
            directional_depth.destroy(device);
            destroy_depth_cube_target(device, local_depth);
            destroy_color_target(device, transmittance);
            return Err(error);
        }
        if let Err(error) = initialize_depth_cube_shader_read_image(
            device,
            queue_family_index,
            queue,
            local_depth.image,
        ) {
            directional_depth.destroy(device);
            destroy_depth_cube_target(device, local_depth);
            destroy_color_target(device, transmittance);
            return Err(error);
        }
        if let Err(error) =
            directional_depth.initialize_shader_read(device, queue_family_index, queue)
        {
            directional_depth.destroy(device);
            destroy_depth_cube_target(device, local_depth);
            destroy_color_target(device, transmittance);
            return Err(error);
        }

        let translucent_views = [transmittance.view; SHADOW_CASCADE_COUNT];
        let mesh_pass_resources = match meshes.create_pass_resources(
            device,
            directional_depth.sampled_view,
            translucent_views,
            [local_depth.view; MAX_LOCAL_LIGHTS],
        ) {
            Ok(resources) => resources,
            Err(error) => {
                directional_depth.destroy(device);
                destroy_depth_cube_target(device, local_depth);
                destroy_color_target(device, transmittance);
                return Err(error);
            }
        };

        tracing::trace!(
            layers = STABLE_CSM_LAYER_COUNT,
            "created tiny Vulkan Stable CSM sampler fallback"
        );
        Ok(ShadowSamplerFallback {
            transmittance,
            directional_depth,
            local_depth,
            mesh_pass_resources,
        })
    }
}

impl VulkanDevice {
    /// Creates fixed shadow resources once per logical device instead of per swapchain resize.
    pub(super) fn create_shadow_resources(&self) -> Result<ShadowResources, VulkanError> {
        let mut build = ShadowBuild::new(self);
        build.shadow_render_pass = Some(create_shadow_render_pass(
            &self.device,
            STABLE_CSM_DEPTH_FORMAT,
        )?);
        build.local_shadow_render_pass =
            Some(create_local_shadow_render_pass(&self.device, DEPTH_FORMAT)?);
        build.translucent_render_pass = Some(create_translucent_shadow_render_pass(
            &self.device,
            TRANSLUCENT_SHADOW_FORMAT,
            DEPTH_FORMAT,
        )?);

        let shadow_render_pass = build
            .shadow_render_pass
            .expect("shadow render pass was just created");
        let local_shadow_render_pass = build
            .local_shadow_render_pass
            .expect("local shadow render pass was just created");
        let translucent_render_pass = build
            .translucent_render_pass
            .expect("translucent shadow render pass was just created");

        let shadow_resolution = self.shadow_map_resolution();
        let shadow_extent = stable_csm_shadow_extent(shadow_resolution);
        build.directional_depth = Some(StableCsmDepthArray::create(
            &self.device,
            &self.memory_properties,
            shadow_extent,
            STABLE_CSM_LAYER_COUNT as u32,
        )?);
        build
            .directional_depth
            .as_ref()
            .expect("Stable CSM directional depth array was just created")
            .initialize_shader_read(&self.device, self.queue_family_index, self.graphics_queue)?;
        for layer in 0..STABLE_CSM_LAYER_COUNT {
            let layer_view = build
                .directional_depth
                .as_ref()
                .expect("Stable CSM directional depth array exists while creating framebuffers")
                .layer_views[layer];
            build.shadow_framebuffers.push(create_shadow_framebuffer(
                &self.device,
                shadow_render_pass,
                layer_view,
                shadow_extent,
            )?);
        }

        build.translucent_depth = Some(create_depth_target(
            &self.device,
            &self.memory_properties,
            shadow_extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?);
        let translucent_depth_view = build
            .translucent_depth
            .as_ref()
            .expect("shared translucent shadow depth was just created")
            .view;
        for _ in 0..SHADOW_CASCADE_COUNT {
            build.cascades.push(create_shadow_cascade_target(
                &self.device,
                &self.memory_properties,
                shadow_extent,
                translucent_render_pass,
                translucent_depth_view,
            )?);
        }
        for cascade in &mut build.cascades {
            initialize_mipped_color_shader_read_image(
                &self.device,
                self.queue_family_index,
                self.graphics_queue,
                cascade.transmittance.image,
                1,
                [1.0; 4],
            )?;
            cascade.transmittance_state = ResourceState::ShaderRead;
        }

        for _ in 0..MAX_LOCAL_LIGHTS {
            build.local.push(create_local_shadow_cube_target(
                &self.device,
                &self.memory_properties,
                local_shadow_extent(shadow_resolution),
                local_shadow_render_pass,
            )?);
        }
        for local in &build.local {
            initialize_depth_cube_shader_read_image(
                &self.device,
                self.queue_family_index,
                self.graphics_queue,
                local.depth.image,
            )?;
        }

        let translucent_views =
            cascade_views(&build.cascades, |cascade| cascade.transmittance.view);
        let depth_array_view = build
            .directional_depth
            .as_ref()
            .expect("Stable CSM directional depth array exists while creating descriptors")
            .sampled_view;
        let local_shadow_views = local_shadow_views(&build.local);

        build.mesh_pass_resources = Some(self.meshes.create_pass_resources(
            &self.device,
            depth_array_view,
            translucent_views,
            local_shadow_views,
        )?);
        build.shadow_pipeline = Some(
            self.meshes
                .create_shadow_pipeline_set(&self.device, shadow_render_pass)?,
        );
        build.local_shadow_pipeline = Some(
            self.meshes
                .create_local_shadow_pipeline_set(&self.device, local_shadow_render_pass)?,
        );
        build.translucent_pipeline = Some(
            self.meshes
                .create_translucent_shadow_pipeline_set(&self.device, translucent_render_pass)?,
        );

        tracing::info!(
            cascade_count = SHADOW_CASCADE_COUNT,
            layer_count = STABLE_CSM_LAYER_COUNT,
            shadow_size = shadow_extent.width(),
            local_count = build.local.len(),
            local_size = build.local.first().map(|local| local.extent.width()).unwrap_or(0),
            depth_format = ?STABLE_CSM_DEPTH_FORMAT,
            translucent_format = ?TRANSLUCENT_SHADOW_FORMAT,
            "created fixed Stable CSM shadows with shared translucent depth"
        );
        Ok(build.finish())
    }

    /// Destroys fixed shadow resources before mesh pipeline layouts are released.
    pub(super) fn destroy_shadow_resources(&self, resources: ShadowResources) {
        resources.destroy(&self.device, &self.meshes);
    }

    /// Creates a swapchain and all minimal render targets needed to clear it.
    pub(super) fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        config: SwapchainConfig,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<VulkanSwapchain, VulkanError> {
        let image_extent = vk::Extent2D {
            width: config.extent.width(),
            height: config.extent.height(),
        };
        let image_usage = swapchain_image_usage(config.transfer_src_supported);
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(config.image_count)
            .image_format(config.format)
            .image_color_space(config.color_space)
            .image_extent(image_extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(config.pre_transform)
            .composite_alpha(config.composite_alpha)
            .present_mode(config.present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        tracing::trace!(
            width = config.extent.width(),
            height = config.extent.height(),
            image_count = config.image_count,
            format = ?config.format,
            present_mode = ?config.present_mode,
            "creating Vulkan swapchain"
        );

        // Safety: the surface belongs to the same instance as this device, the queue family was
        // selected for graphics+present, and all pointers in `create_info` live for this call.
        let handle = unsafe { self.swapchain_loader.create_swapchain(&create_info, None) }?;
        let images = match get_swapchain_images(&self.swapchain_loader, handle) {
            Ok(images) => images,
            Err(error) => {
                self.destroy_swapchain_handle(handle);
                return Err(error);
            }
        };
        let image_views = match self.create_swapchain_image_views(&images, config.format) {
            Ok(image_views) => image_views,
            Err(error) => {
                self.destroy_swapchain_handle(handle);
                return Err(error);
            }
        };
        let mut build = SwapchainBuild::new(self, handle, config, images, image_views);

        build.scene_color = Some(create_color_target(
            &self.device,
            &self.memory_properties,
            config.extent,
            SCENE_COLOR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);
        build.scene_normal_roughness = Some(create_color_target(
            &self.device,
            &self.memory_properties,
            config.extent,
            SCENE_NORMAL_ROUGHNESS_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);
        build.scene_transparent_normal_roughness = Some(create_color_target(
            &self.device,
            &self.memory_properties,
            config.extent,
            SCENE_TRANSPARENT_NORMAL_ROUGHNESS_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);
        build.scene_depth = Some(create_depth_target(
            &self.device,
            &self.memory_properties,
            config.extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);
        build.bloom_levels = create_bloom_targets(
            &self.device,
            &self.memory_properties,
            config.extent,
            SCENE_COLOR_FORMAT,
        )?;
        let god_ray_targets = create_god_ray_targets(
            &self.device,
            &self.memory_properties,
            config.extent,
            SCENE_COLOR_FORMAT,
        )?;
        build.god_ray_mask = Some(god_ray_targets.mask);
        build.god_ray_prefilter = Some(god_ray_targets.prefilter);
        build.god_ray_blur = Some(god_ray_targets.blur);
        build.god_ray_histories = god_ray_targets.histories;

        let scene_color = build
            .scene_color
            .as_ref()
            .expect("scene color was just created");
        let scene_normal_roughness = build
            .scene_normal_roughness
            .as_ref()
            .expect("scene normal roughness was just created");
        let scene_transparent_normal_roughness = build
            .scene_transparent_normal_roughness
            .as_ref()
            .expect("scene transparent normal roughness was just created");
        let scene_depth = build
            .scene_depth
            .as_ref()
            .expect("scene depth was just created");

        build.scene_render_pass = Some(create_scene_render_pass(
            &self.device,
            scene_color.format,
            scene_normal_roughness.format,
            scene_transparent_normal_roughness.format,
            scene_depth.format,
        )?);
        build.scene_fast_render_pass = Some(create_scene_fast_render_pass(
            &self.device,
            scene_color.format,
            scene_depth.format,
        )?);
        build.bloom_downsample_render_pass = Some(create_bloom_downsample_render_pass(
            &self.device,
            SCENE_COLOR_FORMAT,
        )?);
        build.bloom_upsample_render_pass = Some(create_bloom_upsample_render_pass(
            &self.device,
            SCENE_COLOR_FORMAT,
        )?);
        build.god_ray_render_pass = Some(create_god_ray_render_pass(
            &self.device,
            SCENE_COLOR_FORMAT,
        )?);
        build.post_render_pass = Some(create_post_render_pass(&self.device, config.format)?);

        let scene_render_pass = build
            .scene_render_pass
            .expect("scene render pass was just created");
        let scene_fast_render_pass = build
            .scene_fast_render_pass
            .expect("fast scene render pass was just created");
        let bloom_downsample_render_pass = build
            .bloom_downsample_render_pass
            .expect("bloom downsample render pass was just created");
        let bloom_upsample_render_pass = build
            .bloom_upsample_render_pass
            .expect("bloom upsample render pass was just created");
        let god_ray_render_pass = build
            .god_ray_render_pass
            .expect("god-ray render pass was just created");
        let post_render_pass = build
            .post_render_pass
            .expect("post render pass was just created");

        build.mesh_pipeline = Some(
            self.meshes
                .create_scene_pipeline_set(&self.device, scene_render_pass)?,
        );
        build.mesh_fast_pipeline = Some(
            self.meshes
                .create_scene_fast_pipeline_set(&self.device, scene_fast_render_pass)?,
        );
        build.transparent_mesh_pipeline = Some(
            self.meshes
                .create_scene_transparent_pipeline_set(&self.device, scene_render_pass)?,
        );
        build.transparent_mesh_fast_pipeline = Some(
            self.meshes
                .create_scene_transparent_fast_pipeline_set(&self.device, scene_fast_render_pass)?,
        );
        build.taa = Some(TemporalAntiAliasing::create(
            &self.device,
            &self.memory_properties,
            config.extent,
            self.frames.slot_count(),
            scene_color.view,
            scene_depth.view,
            scene_normal_roughness.view,
            scene_transparent_normal_roughness.view,
        )?);
        let taa_history_views = build
            .taa
            .as_ref()
            .expect("temporal anti-aliasing was just created")
            .history_views();
        let bloom_views = build
            .bloom_levels
            .iter()
            .map(|level| level.color.view)
            .collect::<Vec<_>>();
        build.bloom_pipeline = Some(BloomPipeline::create(
            &self.device,
            bloom_downsample_render_pass,
            bloom_upsample_render_pass,
            taa_history_views,
            &bloom_views,
        )?);
        let god_ray_history_views = [
            build.god_ray_histories[0].color.view,
            build.god_ray_histories[1].color.view,
        ];
        build.god_rays_pipeline = Some(GodRaysPipeline::create(
            &self.device,
            god_ray_render_pass,
            self.meshes.frame_set_layout(),
            scene_color.view,
            scene_depth.view,
            scene_transparent_normal_roughness.view,
            self.shadow_fallback.directional_depth.sampled_view,
            build
                .god_ray_mask
                .as_ref()
                .expect("god-ray mask target was just created")
                .color
                .view,
            build
                .god_ray_prefilter
                .as_ref()
                .expect("god-ray prefilter target was just created")
                .color
                .view,
            build
                .god_ray_blur
                .as_ref()
                .expect("god-ray blur target was just created")
                .color
                .view,
            god_ray_history_views,
        )?);
        build.post_pipeline = Some(PostPipeline::create(
            &self.device,
            post_render_pass,
            taa_history_views,
            scene_depth.view,
            scene_normal_roughness.view,
            scene_transparent_normal_roughness.view,
            bloom_views[0],
            god_ray_history_views,
        )?);

        build.scene_framebuffer = Some(create_scene_framebuffer(
            &self.device,
            scene_render_pass,
            scene_color.view,
            scene_normal_roughness.view,
            scene_transparent_normal_roughness.view,
            scene_depth.view,
            config.extent,
        )?);
        build.scene_fast_framebuffer = Some(create_scene_fast_framebuffer(
            &self.device,
            scene_fast_render_pass,
            scene_color.view,
            scene_depth.view,
            config.extent,
        )?);
        build.bloom_downsample_framebuffers = create_bloom_framebuffers(
            &self.device,
            bloom_downsample_render_pass,
            &build.bloom_levels,
        )?;
        let upsample_levels = &build.bloom_levels[..BLOOM_MIP_COUNT - 1];
        build.bloom_upsample_framebuffers =
            create_bloom_framebuffers(&self.device, bloom_upsample_render_pass, upsample_levels)?;
        build.god_ray_framebuffers =
            create_god_ray_framebuffers(&self.device, god_ray_render_pass, &build)?;
        build.post_framebuffers =
            self.create_post_framebuffers(post_render_pass, &build.image_views, config.extent)?;

        tracing::info!(
            width = config.extent.width(),
            height = config.extent.height(),
            image_count = build.images.len(),
            image_view_count = build.image_views.len(),
            framebuffer_count = build.post_framebuffers.len()
                + build.bloom_downsample_framebuffers.len()
                + build.bloom_upsample_framebuffers.len()
                + build.god_ray_framebuffers.count()
                + 2,
            format = ?config.format,
            present_mode = ?config.present_mode,
            transfer_src_supported = config.transfer_src_supported,
            "created Vulkan swapchain resources"
        );

        Ok(build.finish())
    }

    /// Destroys one swapchain and every resource owned by that swapchain.
    pub(super) fn destroy_swapchain(&self, swapchain: VulkanSwapchain) {
        tracing::trace!(
            width = swapchain.extent.width(),
            height = swapchain.extent.height(),
            image_count = swapchain.images.len(),
            image_view_count = swapchain.image_views.len(),
            framebuffer_count = swapchain.post_framebuffers.len()
                + swapchain.bloom_downsample_framebuffers.len()
                + swapchain.bloom_upsample_framebuffers.len()
                + swapchain.god_ray_framebuffers.count()
                + 2,
            format = ?swapchain.format,
            color_space = ?swapchain.color_space,
            present_mode = ?swapchain.present_mode,
            "destroying Vulkan swapchain"
        );

        self.destroy_framebuffers(swapchain.post_framebuffers);
        swapchain.god_ray_framebuffers.destroy(&self.device);
        self.destroy_framebuffers(swapchain.bloom_upsample_framebuffers);
        self.destroy_framebuffers(swapchain.bloom_downsample_framebuffers);
        destroy_framebuffer(&self.device, swapchain.scene_fast_framebuffer);
        destroy_framebuffer(&self.device, swapchain.scene_framebuffer);
        swapchain.post_pipeline.destroy(&self.device);
        swapchain.god_rays_pipeline.destroy(&self.device);
        swapchain.bloom_pipeline.destroy(&self.device);
        swapchain.taa.destroy(&self.device);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.mesh_pipeline);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.mesh_fast_pipeline);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.transparent_mesh_pipeline);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.transparent_mesh_fast_pipeline);
        destroy_render_pass(&self.device, swapchain.post_render_pass);
        destroy_render_pass(&self.device, swapchain.god_ray_render_pass);
        destroy_render_pass(&self.device, swapchain.bloom_upsample_render_pass);
        destroy_render_pass(&self.device, swapchain.bloom_downsample_render_pass);
        destroy_render_pass(&self.device, swapchain.scene_fast_render_pass);
        destroy_render_pass(&self.device, swapchain.scene_render_pass);
        swapchain.bloom.destroy(&self.device);
        swapchain.god_rays.destroy(&self.device);
        swapchain.scene.destroy(&self.device);
        self.destroy_image_views(swapchain.image_views);
        self.destroy_swapchain_handle(swapchain.handle);
    }

    /// Creates one color image view for each swapchain image.
    fn create_swapchain_image_views(
        &self,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, VulkanError> {
        let mut image_views = Vec::with_capacity(images.len());

        for &image in images {
            match create_swapchain_image_view(&self.device, image, format) {
                Ok(image_view) => image_views.push(image_view),
                Err(error) => {
                    self.destroy_image_views(image_views);
                    return Err(error);
                }
            }
        }

        tracing::trace!(
            count = image_views.len(),
            "created Vulkan swapchain image views"
        );
        Ok(image_views)
    }

    /// Creates one post framebuffer for each swapchain image view.
    fn create_post_framebuffers(
        &self,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        extent: NonZeroExtent,
    ) -> Result<Vec<vk::Framebuffer>, VulkanError> {
        let mut framebuffers = Vec::with_capacity(image_views.len());

        for &image_view in image_views {
            match create_post_framebuffer(&self.device, render_pass, image_view, extent) {
                Ok(framebuffer) => framebuffers.push(framebuffer),
                Err(error) => {
                    self.destroy_framebuffers(framebuffers);
                    return Err(error);
                }
            }
        }

        tracing::trace!(
            count = framebuffers.len(),
            width = extent.width(),
            height = extent.height(),
            "created Vulkan post framebuffers"
        );
        Ok(framebuffers)
    }

    /// Destroys image views created for swapchain images.
    fn destroy_image_views(&self, image_views: Vec<vk::ImageView>) {
        for image_view in image_views {
            destroy_image_view(&self.device, image_view);
        }
    }

    /// Destroys framebuffer handles created for swapchain image views.
    fn destroy_framebuffers(&self, framebuffers: Vec<vk::Framebuffer>) {
        for framebuffer in framebuffers {
            destroy_framebuffer(&self.device, framebuffer);
        }
    }

    /// Destroys the raw swapchain handle after child resources are gone.
    fn destroy_swapchain_handle(&self, handle: vk::SwapchainKHR) {
        if handle == vk::SwapchainKHR::null() {
            return;
        }

        // Safety: the swapchain was created by this loader and is destroyed exactly once after
        // framebuffers, render pass, and image views have been destroyed.
        unsafe {
            self.swapchain_loader.destroy_swapchain(handle, None);
        }
    }
}

impl VulkanSwapchain {
    /// Returns the scene render pass that writes scene color and depth targets.
    pub(super) fn scene_render_pass(&self) -> vk::RenderPass {
        self.scene_render_pass
    }

    /// Returns the lightweight scene render pass that omits material metadata attachments.
    pub(super) fn scene_fast_render_pass(&self) -> vk::RenderPass {
        self.scene_fast_render_pass
    }

    /// Returns the post render pass that writes the acquired swapchain image.
    pub(super) fn post_render_pass(&self) -> vk::RenderPass {
        self.post_render_pass
    }

    /// Returns the render pass that extracts and downsamples HDR bloom mips.
    pub(super) fn bloom_downsample_render_pass(&self) -> vk::RenderPass {
        self.bloom_downsample_render_pass
    }

    /// Returns the render pass that additively upsamples bloom mips.
    pub(super) fn bloom_upsample_render_pass(&self) -> vk::RenderPass {
        self.bloom_upsample_render_pass
    }

    /// Returns the render pass that writes the low-resolution god-ray chain.
    pub(super) fn god_ray_render_pass(&self) -> vk::RenderPass {
        self.god_ray_render_pass
    }

    /// Returns the mesh graphics pipeline compatible with the scene pass.
    pub(super) fn mesh_pipeline(&self) -> MeshPipelineSet {
        self.mesh_pipeline
    }

    /// Returns the scene mesh pipeline that skips material metadata writes.
    pub(super) fn mesh_fast_pipeline(&self) -> MeshPipelineSet {
        self.mesh_fast_pipeline
    }

    /// Returns the mesh graphics pipeline that blends transparent scene materials.
    pub(super) fn transparent_mesh_pipeline(&self) -> MeshPipelineSet {
        self.transparent_mesh_pipeline
    }

    /// Returns the transparent scene mesh pipeline that skips material metadata writes.
    pub(super) fn transparent_mesh_fast_pipeline(&self) -> MeshPipelineSet {
        self.transparent_mesh_fast_pipeline
    }

    /// Returns the post pipeline compatible with this swapchain's post pass.
    pub(super) fn post_pipeline(&self) -> &PostPipeline {
        &self.post_pipeline
    }

    pub(super) fn prepare_taa_frame(
        &mut self,
        device: &Device,
        slot_index: usize,
        snapshot: &FrameSnapshot,
        camera: CameraSnapshot,
        quality: RenderQualitySettings,
        use_corrected_scene_color: bool,
    ) -> Result<TaaFrameInfo, VulkanError> {
        self.taa.prepare_frame(
            device,
            slot_index,
            snapshot,
            camera,
            quality,
            self.extent_2d(),
            use_corrected_scene_color,
        )
    }

    pub(super) fn record_taa(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), VulkanError> {
        self.taa.record(device, command_buffer, self.extent_2d())
    }

    pub(super) fn taa_history_write_index(&self) -> usize {
        self.taa.history_write_index()
    }

    pub(super) fn taa_jitter_pixels(&self) -> [f32; 2] {
        self.taa.pending_jitter_pixels()
    }

    /// Returns the bloom pipeline compatible with this swapchain's bloom passes.
    pub(super) fn bloom_pipeline(&self) -> &BloomPipeline {
        &self.bloom_pipeline
    }

    /// Returns the god-ray pipeline compatible with this swapchain's low-resolution targets.
    pub(super) fn god_rays_pipeline(&self) -> &GodRaysPipeline {
        &self.god_rays_pipeline
    }

    /// Updates the CSM view used by the quality volumetric mask pass.
    pub(super) fn update_god_ray_shadow_view(
        &mut self,
        device: &Device,
        image_view: vk::ImageView,
    ) {
        self.god_rays_pipeline
            .update_directional_shadow_view(device, image_view);
    }

    /// Returns the CSM view currently bound by the volumetric mask pass.
    pub(super) fn god_ray_shadow_view(&self) -> vk::ImageView {
        self.god_rays_pipeline.directional_shadow_view()
    }

    /// Returns the number of images owned by this swapchain.
    pub(super) fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Returns whether swapchain images can be copied into app-visible readback buffers.
    pub(super) fn transfer_src_supported(&self) -> bool {
        self.transfer_src_supported
    }

    /// Returns the swapchain image that corresponds to one acquired image index.
    pub(super) fn image_for_index(&self, image_index: u32) -> Result<vk::Image, VulkanError> {
        let index = image_index as usize;
        self.images
            .get(index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index,
                count: self.images.len(),
            })
    }

    /// Returns the framebuffer used by the graph scene pass.
    pub(super) fn scene_framebuffer(&self) -> vk::Framebuffer {
        self.scene_framebuffer
    }

    /// Returns the lightweight scene framebuffer used without material metadata.
    pub(super) fn scene_fast_framebuffer(&self) -> vk::Framebuffer {
        self.scene_fast_framebuffer
    }

    /// Returns the post framebuffer that corresponds to one acquired swapchain image.
    pub(super) fn post_framebuffer_for_image(
        &self,
        image_index: u32,
    ) -> Result<vk::Framebuffer, VulkanError> {
        let index = image_index as usize;
        self.post_framebuffers.get(index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index,
                count: self.post_framebuffers.len(),
            },
        )
    }

    /// Returns one bloom mip extent.
    pub(super) fn bloom_extent_2d(&self, mip_index: usize) -> Result<vk::Extent2D, VulkanError> {
        self.bloom.extent_2d(mip_index)
    }

    /// Returns the shared extent of the low-resolution god-ray targets.
    pub(super) fn god_ray_extent_2d(&self) -> vk::Extent2D {
        self.god_rays.extent_2d()
    }

    /// Returns whether a previous temporal god-ray history is available.
    pub(super) fn god_ray_history_valid(&self) -> bool {
        self.god_rays.history_valid()
    }

    /// Returns the temporal god-ray history target written by the current graph.
    pub(super) fn god_ray_history_write_index(&self) -> usize {
        self.god_rays.history_write_index()
    }

    /// Invalidates the low-resolution God Ray history after a quality-model switch.
    pub(super) fn invalidate_god_ray_history(&mut self) {
        self.god_rays.invalidate_history();
    }

    /// Returns the framebuffer used by a downsample pass writing one bloom mip.
    pub(super) fn bloom_downsample_framebuffer(
        &self,
        mip_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        self.bloom_downsample_framebuffers
            .get(mip_index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: mip_index,
                count: self.bloom_downsample_framebuffers.len(),
            })
    }

    /// Returns the framebuffer used by an upsample pass writing one bloom mip.
    pub(super) fn bloom_upsample_framebuffer(
        &self,
        target_mip_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        self.bloom_upsample_framebuffers
            .get(target_mip_index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: target_mip_index,
                count: self.bloom_upsample_framebuffers.len(),
            })
    }

    pub(super) fn god_ray_mask_framebuffer(&self) -> vk::Framebuffer {
        self.god_ray_framebuffers.mask
    }

    pub(super) fn god_ray_prefilter_framebuffer(&self) -> vk::Framebuffer {
        self.god_ray_framebuffers.prefilter
    }

    pub(super) fn god_ray_radial_framebuffer(&self) -> vk::Framebuffer {
        self.god_ray_framebuffers.radial
    }

    pub(super) fn god_ray_history_framebuffer(
        &self,
        history_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        self.god_ray_framebuffers
            .histories
            .get(history_index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: history_index,
                count: self.god_ray_framebuffers.histories.len(),
            })
    }

    /// Returns the current graph resource states before compiling a frame graph.
    pub(super) fn graph_initial_states(
        &self,
        image_index: u32,
        shadows: Option<&ShadowResources>,
    ) -> Result<FrameGraphInitialStates, VulkanError> {
        let shadow_states = shadows.map_or(
            [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
            ShadowResources::shadow_states,
        );
        let translucent_shadow_states = shadows.map_or(
            [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
            ShadowResources::transmittance_states,
        );
        let (
            scene_color_state,
            scene_normal_roughness_state,
            scene_transparent_normal_roughness_state,
            scene_depth_state,
        ) = self.scene.graph_states();
        let (taa_histories, taa_depth_histories, taa_normal_histories, motion_vectors) =
            self.taa.graph_states();
        Ok(FrameGraphInitialStates::new(
            self.image_state(image_index)?,
            shadow_states,
            translucent_shadow_states,
            scene_color_state,
            scene_normal_roughness_state,
            scene_transparent_normal_roughness_state,
            scene_depth_state,
        )
        .with_bloom_mips(self.bloom.graph_states())
        .with_god_rays(
            self.god_rays.mask.state,
            self.god_rays.prefilter.state,
            self.god_rays.blur.state,
            self.god_rays.history_states(),
        )
        .with_taa(
            taa_histories,
            taa_depth_histories,
            taa_normal_histories,
            motion_vectors,
        ))
    }

    /// Applies graph final states after command recording has committed the resource plan.
    pub(super) fn apply_graph_final_states(
        &mut self,
        image_index: u32,
        plan: &crate::renderer::graph::FrameGraphPlan,
    ) -> Result<(), VulkanError> {
        if let Some(state) = plan.final_state_for(GraphResource::SwapchainImage) {
            *self.image_state_mut(image_index)? = state;
        }
        self.scene.apply_graph_final_states(plan);
        self.bloom.apply_graph_final_states(plan);
        self.god_rays.apply_graph_final_states(plan);
        self.taa.apply_graph_final_states(plan);
        Ok(())
    }

    /// Returns the image and aspect range used by one graph resource barrier.
    pub(super) fn graph_image(
        &self,
        resource: GraphResource,
        image_index: u32,
    ) -> Result<(vk::Image, vk::ImageAspectFlags), VulkanError> {
        if resource.is_shadow_resource() {
            return Err(VulkanError::GraphCompile(format!(
                "shadow graph resource {} is not owned by the swapchain",
                resource.name()
            )));
        }
        if let Some(image) = self.scene.graph_image(resource) {
            return Ok(image);
        }
        if let Some(image) = self.bloom.graph_image(resource) {
            return Ok(image);
        }
        if let Some(image) = self.god_rays.graph_image(resource) {
            return Ok(image);
        }
        if let Some(image) = self.taa.graph_image(resource) {
            return Ok(image);
        }
        match resource {
            GraphResource::SwapchainImage => Ok((
                self.image_for_index(image_index)?,
                vk::ImageAspectFlags::COLOR,
            )),
            GraphResource::SceneColor
            | GraphResource::SceneNormalRoughness
            | GraphResource::SceneTransparentNormalRoughness
            | GraphResource::SceneDepth
            | GraphResource::BloomMip0
            | GraphResource::BloomMip1
            | GraphResource::BloomMip2
            | GraphResource::BloomMip3
            | GraphResource::BloomMip4
            | GraphResource::GodRayMask
            | GraphResource::GodRayPrefilter
            | GraphResource::GodRayBlur
            | GraphResource::GodRayHistory0
            | GraphResource::GodRayHistory1
            | GraphResource::TaaHistory0
            | GraphResource::TaaHistory1
            | GraphResource::TaaDepthHistory0
            | GraphResource::TaaDepthHistory1
            | GraphResource::TaaNormalHistory0
            | GraphResource::TaaNormalHistory1
            | GraphResource::MotionVectors => {
                unreachable!("scene, bloom, god-ray, and TAA resources return early above")
            }
            GraphResource::ShadowCascade0
            | GraphResource::ShadowCascade1
            | GraphResource::ShadowCascade2
            | GraphResource::ShadowCascade3
            | GraphResource::TranslucentShadow0
            | GraphResource::TranslucentShadow1
            | GraphResource::TranslucentShadow2
            | GraphResource::TranslucentShadow3 => {
                unreachable!("shadow resources return early above")
            }
        }
    }

    /// Returns the swapchain extent in the Vulkan API shape used by render areas.
    pub(super) fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.extent.width(),
            height: self.extent.height(),
        }
    }

    /// Returns the tracked layout state for one swapchain image.
    fn image_state(&self, image_index: u32) -> Result<ResourceState, VulkanError> {
        let index = image_index as usize;
        self.image_states
            .get(index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index,
                count: self.image_states.len(),
            })
    }

    /// Returns a mutable tracked layout state for one swapchain image.
    fn image_state_mut(&mut self, image_index: u32) -> Result<&mut ResourceState, VulkanError> {
        let index = image_index as usize;
        let count = self.image_states.len();
        self.image_states
            .get_mut(index)
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange { index, count })
    }
}

impl ShadowResources {
    /// Returns the render pass that writes one opaque cascade depth target.
    pub(super) fn shadow_render_pass(&self) -> vk::RenderPass {
        self.shadow_render_pass
    }

    /// Returns the depth-only render pass for one local shadow cubemap face.
    pub(super) fn local_shadow_render_pass(&self) -> vk::RenderPass {
        self.local_shadow_render_pass
    }

    /// Returns the render pass that writes one translucent cascade transmittance target.
    pub(super) fn translucent_render_pass(&self) -> vk::RenderPass {
        self.translucent_render_pass
    }

    /// Returns the opaque shadow mesh pipelines shared by all cascades.
    pub(super) fn shadow_pipeline(&self) -> MeshPipelineSet {
        self.shadow_pipeline
    }

    /// Returns the depth-only local shadow mesh pipelines shared by cubemap faces.
    pub(super) fn local_shadow_pipeline(&self) -> MeshPipelineSet {
        self.local_shadow_pipeline
    }

    /// Returns the translucent shadow mesh pipelines shared by all cascades.
    pub(super) fn translucent_pipeline(&self) -> MeshPipelineSet {
        self.translucent_pipeline
    }

    /// Returns descriptors used by scene mesh shaders to sample every shadow cascade.
    pub(super) fn mesh_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
    }

    /// Returns the shared raw-depth descriptors used while rendering translucent shadow casters.
    pub(super) fn translucent_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
    }

    /// Returns the common render extent of every Stable CSM cascade layer.
    pub(super) fn extent_2d(&self, cascade_index: usize) -> Result<vk::Extent2D, VulkanError> {
        self.cascade(cascade_index)?;
        Ok(vk::Extent2D {
            width: self.shadow_extent.width(),
            height: self.shadow_extent.height(),
        })
    }

    pub(super) fn shadow_extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.shadow_extent.width(),
            height: self.shadow_extent.height(),
        }
    }

    /// Returns the fixed render extent shared by local-light shadow maps.
    pub(super) fn local_extent_2d(&self) -> vk::Extent2D {
        let local = self
            .local
            .first()
            .expect("local shadow resources are created as a fixed set");
        vk::Extent2D {
            width: local.extent.width(),
            height: local.extent.height(),
        }
    }

    /// Returns the framebuffer for the stable CSM layer belonging to one cascade.
    ///
    /// The legacy backing image may still expose additional layers while a swapchain is being
    /// rebuilt, but the Stable CSM path always addresses sample zero. Keeping that mapping here
    /// makes the render path independent from any removed temporal-direction cursor.
    pub(super) fn shadow_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        let layer = self.shadow_layer(cascade_index, 0)?;
        Ok(self.shadow_framebuffers[layer])
    }

    /// Returns the framebuffer used to render one local-light cubemap face.
    pub(super) fn local_shadow_framebuffer(
        &self,
        light_index: usize,
        face_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        let local = self.local_shadow_cube(light_index)?;
        local.framebuffers.get(face_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: face_index,
                count: local.framebuffers.len(),
            },
        )
    }

    /// Returns the framebuffer used to render one translucent shadow cascade.
    pub(super) fn translucent_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        Ok(self.cascade(cascade_index)?.translucent_framebuffer)
    }

    /// Returns tracked directional shadow-map layouts for graph compilation.
    fn shadow_states(&self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        [self.directional_state; SHADOW_CASCADE_COUNT]
    }

    /// Returns tracked translucent transmittance layouts for graph compilation.
    fn transmittance_states(&self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        std::array::from_fn(|index| self.cascades[index].transmittance_state)
    }

    /// Applies graph final states owned by the fixed shadow resource set.
    pub(super) fn apply_graph_final_states(
        &mut self,
        plan: &crate::renderer::graph::FrameGraphPlan,
    ) {
        if let Some(state) = SHADOW_CASCADE_RESOURCES
            .into_iter()
            .find_map(|resource| plan.final_state_for(resource))
        {
            self.directional_state = state;
        }
        for (cascade, resource) in self.cascades.iter_mut().zip(TRANSLUCENT_SHADOW_RESOURCES) {
            if let Some(state) = plan.final_state_for(resource) {
                cascade.transmittance_state = state;
            }
        }
    }

    /// Returns the image and aspect range used by graph barriers for shadow resources.
    pub(super) fn graph_image(
        &self,
        resource: GraphResource,
    ) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        SHADOW_CASCADE_RESOURCES
            .iter()
            .any(|candidate| *candidate == resource)
            .then_some((self.directional_depth.image, vk::ImageAspectFlags::DEPTH))
            .or_else(|| {
                TRANSLUCENT_SHADOW_RESOURCES
                    .iter()
                    .position(|candidate| *candidate == resource)
                    .map(|index| {
                        (
                            self.cascades[index].transmittance.image,
                            vk::ImageAspectFlags::COLOR,
                        )
                    })
            })
    }

    /// Moves one independently addressed array layer from sampling to depth attachment use.
    pub(super) fn transition_shadow_layer_to_attachment(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        cascade_index: usize,
    ) -> Result<(), VulkanError> {
        let layer = self.shadow_layer(cascade_index, 0)?;
        self.directional_depth
            .transition_layer_to_attachment(device, command_buffer, layer as u32);
        Ok(())
    }

    /// Makes one freshly rendered array layer available to comparison sampling.
    pub(super) fn transition_shadow_layer_to_shader_read(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        cascade_index: usize,
    ) -> Result<(), VulkanError> {
        let layer = self.shadow_layer(cascade_index, 0)?;
        self.directional_depth.transition_layer_to_shader_read(
            device,
            command_buffer,
            layer as u32,
        );
        Ok(())
    }

    /// Returns one local-shadow depth image.
    pub(super) fn local_depth_image(&self, light_index: usize) -> Result<vk::Image, VulkanError> {
        Ok(self.local_shadow_cube(light_index)?.depth.image)
    }

    /// Releases fixed shadow framebuffers, descriptors, pipelines, render passes, and targets.
    fn destroy(self, device: &Device, meshes: &VulkanMeshStore) {
        self.mesh_pass_resources.destroy(device);
        meshes.destroy_pipeline_set(device, self.translucent_pipeline);
        meshes.destroy_pipeline_set(device, self.local_shadow_pipeline);
        meshes.destroy_pipeline_set(device, self.shadow_pipeline);
        for cascade in self.cascades {
            destroy_shadow_cascade(device, cascade);
        }
        destroy_depth_target(device, self.translucent_depth);
        for local in self.local {
            destroy_local_shadow_cube(device, local);
        }
        for framebuffer in self.shadow_framebuffers {
            destroy_framebuffer(device, framebuffer);
        }
        self.directional_depth.destroy(device);
        destroy_render_pass(device, self.translucent_render_pass);
        destroy_render_pass(device, self.local_shadow_render_pass);
        destroy_render_pass(device, self.shadow_render_pass);
    }

    /// Returns one cascade record by index.
    fn cascade(&self, cascade_index: usize) -> Result<&ShadowCascade, VulkanError> {
        self.cascades
            .get(cascade_index)
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: cascade_index,
                count: self.cascades.len(),
            })
    }

    fn shadow_layer(
        &self,
        cascade_index: usize,
        _sample_index: usize,
    ) -> Result<usize, VulkanError> {
        self.cascade(cascade_index)?;
        Ok(cascade_index)
    }

    fn local_shadow_cube(&self, light_index: usize) -> Result<&LocalShadowCube, VulkanError> {
        self.local
            .get(light_index)
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: light_index,
                count: self.local.len(),
            })
    }
}

impl ShadowSamplerFallback {
    /// Returns descriptors that bind full-light dummy shadow maps at the mesh pass set.
    pub(super) fn mesh_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
    }

    /// Releases the dummy descriptor set and its tiny sampled images.
    pub(super) fn destroy(self, device: &Device) {
        self.mesh_pass_resources.destroy(device);
        self.directional_depth.destroy(device);
        destroy_depth_cube_target(device, self.local_depth);
        destroy_color_target(device, self.transmittance);
    }
}

/// Reads all surface facts required to choose a swapchain configuration.
pub(super) fn query_surface_support(
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<SwapchainSupport, VulkanError> {
    let capabilities = get_surface_capabilities(surface_loader, physical_device, surface)?;
    let formats = get_surface_formats(surface_loader, physical_device, surface)?;
    let present_modes = get_surface_present_modes(surface_loader, physical_device, surface)?;

    tracing::trace!(
        min_image_count = capabilities.min_image_count,
        max_image_count = capabilities.max_image_count,
        format_count = formats.len(),
        present_mode_count = present_modes.len(),
        "queried Vulkan surface support"
    );

    Ok(SwapchainSupport {
        capabilities,
        formats,
        present_modes,
    })
}

/// Chooses the concrete swapchain settings from surface capabilities and requested extent.
pub(super) fn choose_swapchain_config(
    support: &SwapchainSupport,
    requested_extent: NonZeroExtent,
) -> Result<SwapchainConfig, VulkanError> {
    let format = choose_surface_format(&support.formats)?;
    let present_mode = choose_present_mode(&support.present_modes)?;
    let extent = choose_swapchain_extent(&support.capabilities, requested_extent)?;

    if !support
        .capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        return Err(VulkanError::ColorAttachmentUnsupported);
    }

    Ok(SwapchainConfig {
        extent,
        image_count: choose_swapchain_image_count(&support.capabilities),
        format: format.format,
        color_space: format.color_space,
        present_mode,
        pre_transform: support.capabilities.current_transform,
        composite_alpha: choose_composite_alpha(support.capabilities.supported_composite_alpha)?,
        transfer_src_supported: support
            .capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC),
    })
}

/// Returns the image usage flags requested when creating swapchain images.
fn swapchain_image_usage(transfer_src_supported: bool) -> vk::ImageUsageFlags {
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if transfer_src_supported {
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    usage
}

/// Selects the surface format, preferring common sRGB output when available.
fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Result<vk::SurfaceFormatKHR, VulkanError> {
    if formats.is_empty() {
        return Err(VulkanError::SurfaceFormatsUnavailable);
    }

    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        return Ok(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });
    }

    Ok(formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0]))
}

/// Selects mailbox present when possible and falls back to FIFO, which Vulkan guarantees.
fn choose_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> Result<vk::PresentModeKHR, VulkanError> {
    if present_modes.is_empty() {
        return Err(VulkanError::PresentModesUnavailable);
    }

    Ok(present_modes
        .iter()
        .copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO))
}

/// Resolves the swapchain extent from the surface capability contract and app request.
fn choose_swapchain_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    requested: NonZeroExtent,
) -> Result<NonZeroExtent, VulkanError> {
    if capabilities.current_extent.width != u32::MAX {
        return NonZeroExtent::new(
            capabilities.current_extent.width,
            capabilities.current_extent.height,
        )
        .ok_or(VulkanError::SurfaceExtentUnavailable);
    }

    let width = requested.width().clamp(
        capabilities.min_image_extent.width,
        capabilities.max_image_extent.width,
    );
    let height = requested.height().clamp(
        capabilities.min_image_extent.height,
        capabilities.max_image_extent.height,
    );

    NonZeroExtent::new(width, height).ok_or(VulkanError::SurfaceExtentUnavailable)
}

/// Chooses one more image than the minimum unless the surface exposes a hard maximum.
fn choose_swapchain_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let requested = capabilities.min_image_count + 1;

    if capabilities.max_image_count > 0 {
        requested.min(capabilities.max_image_count)
    } else {
        requested
    }
}

/// Selects an alpha compositing mode supported by the platform surface.
fn choose_composite_alpha(
    supported: vk::CompositeAlphaFlagsKHR,
) -> Result<vk::CompositeAlphaFlagsKHR, VulkanError> {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|mode| supported.contains(*mode))
    .ok_or(VulkanError::CompositeAlphaUnavailable)
}

/// Reads surface capabilities for swapchain sizing and image usage decisions.
fn get_surface_capabilities(
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::SurfaceCapabilitiesKHR, VulkanError> {
    // Safety: `surface` was created for this instance and `physical_device` belongs to it.
    unsafe { surface_loader.get_physical_device_surface_capabilities(physical_device, surface) }
        .map_err(VulkanError::Vk)
}

/// Reads supported surface formats for swapchain image creation.
fn get_surface_formats(
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<Vec<vk::SurfaceFormatKHR>, VulkanError> {
    // Safety: `surface` was created for this instance and `physical_device` belongs to it.
    unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
        .map_err(VulkanError::Vk)
}

/// Reads supported present modes for swapchain presentation behavior.
fn get_surface_present_modes(
    surface_loader: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<Vec<vk::PresentModeKHR>, VulkanError> {
    // Safety: `surface` was created for this instance and `physical_device` belongs to it.
    unsafe { surface_loader.get_physical_device_surface_present_modes(physical_device, surface) }
        .map_err(VulkanError::Vk)
}

/// Reads the images that belong to a newly created swapchain.
fn get_swapchain_images(
    swapchain_loader: &khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
) -> Result<Vec<vk::Image>, VulkanError> {
    // Safety: `swapchain` was created by this loader and is still alive for the query.
    unsafe { swapchain_loader.get_swapchain_images(swapchain) }.map_err(VulkanError::Vk)
}

/// Creates a color image view for one swapchain image.
fn create_swapchain_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, VulkanError> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(subresource_range);

    // Safety: the image belongs to a live swapchain created from `device`, and the view only
    // exposes the color aspect needed by the swapchain render pass.
    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

fn stable_csm_shadow_extent(size: u32) -> NonZeroExtent {
    NonZeroExtent::new(size, size).expect("stable CSM shadow array extent must be non-zero")
}

fn local_shadow_extent(shadow_resolution: u32) -> NonZeroExtent {
    let size = (shadow_resolution / 2).clamp(512, 2048);
    NonZeroExtent::new(size, size).expect("local shadow map extent must be non-zero")
}

fn bloom_mip_extent(full_extent: NonZeroExtent, mip_index: usize) -> NonZeroExtent {
    let divisor = 1_u32 << (mip_index as u32 + 1);
    let width = (full_extent.width() / divisor).max(1);
    let height = (full_extent.height() / divisor).max(1);

    NonZeroExtent::new(width, height).expect("bloom mip extent must be non-zero")
}

fn god_ray_extent(full_extent: NonZeroExtent) -> NonZeroExtent {
    let width = (full_extent.width() / 4).max(1);
    let height = (full_extent.height() / 4).max(1);

    NonZeroExtent::new(width, height).expect("god-ray extent must be non-zero")
}

fn create_bloom_targets(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    full_extent: NonZeroExtent,
    format: vk::Format,
) -> Result<Vec<BloomLevelTarget>, VulkanError> {
    let mut levels: Vec<BloomLevelTarget> = Vec::with_capacity(BLOOM_MIP_COUNT);
    for mip_index in 0..BLOOM_MIP_COUNT {
        let extent = bloom_mip_extent(full_extent, mip_index);
        let color = match create_color_target(
            device,
            memory_properties,
            extent,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        ) {
            Ok(color) => color,
            Err(error) => {
                for level in levels.into_iter().rev() {
                    destroy_color_target(device, level.color);
                }
                return Err(error);
            }
        };
        levels.push(BloomLevelTarget {
            color,
            extent,
            state: ResourceState::Undefined,
        });
    }

    Ok(levels)
}

fn create_god_ray_targets(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    full_extent: NonZeroExtent,
    format: vk::Format,
) -> Result<GodRayTargetSet, VulkanError> {
    let extent = god_ray_extent(full_extent);
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let mut targets = Vec::with_capacity(3 + GOD_RAY_HISTORY_COUNT);
    for _ in 0..(3 + GOD_RAY_HISTORY_COUNT) {
        match create_color_target(device, memory_properties, extent, format, usage) {
            Ok(color) => targets.push(GodRayTarget {
                color,
                extent,
                state: ResourceState::Undefined,
            }),
            Err(error) => {
                for target in targets.into_iter().rev() {
                    destroy_color_target(device, target.color);
                }
                return Err(error);
            }
        }
    }

    let mut targets = targets.into_iter();
    Ok(GodRayTargetSet {
        mask: targets
            .next()
            .expect("god-ray mask target was just created"),
        prefilter: targets
            .next()
            .expect("god-ray prefilter target was just created"),
        blur: targets
            .next()
            .expect("god-ray radial target was just created"),
        histories: targets.collect(),
    })
}

fn create_bloom_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    levels: &[BloomLevelTarget],
) -> Result<Vec<vk::Framebuffer>, VulkanError> {
    let mut framebuffers = Vec::with_capacity(levels.len());
    for level in levels {
        match create_post_framebuffer(device, render_pass, level.color.view, level.extent) {
            Ok(framebuffer) => framebuffers.push(framebuffer),
            Err(error) => {
                for framebuffer in framebuffers {
                    destroy_framebuffer(device, framebuffer);
                }
                return Err(error);
            }
        }
    }

    Ok(framebuffers)
}

fn create_god_ray_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    build: &SwapchainBuild<'_>,
) -> Result<GodRayFramebuffers, VulkanError> {
    let create = |target: &GodRayTarget| {
        create_post_framebuffer(device, render_pass, target.color.view, target.extent)
    };
    let mut created = Vec::with_capacity(3 + build.god_ray_histories.len());

    let mask = match create(
        build
            .god_ray_mask
            .as_ref()
            .expect("god-ray mask target exists while building framebuffers"),
    ) {
        Ok(framebuffer) => {
            created.push(framebuffer);
            framebuffer
        }
        Err(error) => return Err(error),
    };
    let prefilter = match create(
        build
            .god_ray_prefilter
            .as_ref()
            .expect("god-ray prefilter target exists while building framebuffers"),
    ) {
        Ok(framebuffer) => {
            created.push(framebuffer);
            framebuffer
        }
        Err(error) => {
            for framebuffer in created {
                destroy_framebuffer(device, framebuffer);
            }
            return Err(error);
        }
    };
    let radial = match create(
        build
            .god_ray_blur
            .as_ref()
            .expect("god-ray radial target exists while building framebuffers"),
    ) {
        Ok(framebuffer) => {
            created.push(framebuffer);
            framebuffer
        }
        Err(error) => {
            for framebuffer in created {
                destroy_framebuffer(device, framebuffer);
            }
            return Err(error);
        }
    };

    let mut histories = Vec::with_capacity(build.god_ray_histories.len());
    for target in &build.god_ray_histories {
        match create(target) {
            Ok(framebuffer) => {
                created.push(framebuffer);
                histories.push(framebuffer);
            }
            Err(error) => {
                for framebuffer in created {
                    destroy_framebuffer(device, framebuffer);
                }
                return Err(error);
            }
        }
    }

    Ok(GodRayFramebuffers {
        mask,
        prefilter,
        radial,
        histories,
    })
}

fn create_local_shadow_cube_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    render_pass: vk::RenderPass,
) -> Result<LocalShadowCube, VulkanError> {
    let depth = create_depth_cube_target(
        device,
        memory_properties,
        extent,
        DEPTH_FORMAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST,
    )?;
    let mut framebuffers = [vk::Framebuffer::null(); 6];
    for face in 0..6 {
        framebuffers[face] = match create_local_shadow_framebuffer(
            device,
            render_pass,
            depth.face_views[face],
            extent,
        ) {
            Ok(framebuffer) => framebuffer,
            Err(error) => {
                for framebuffer in framebuffers {
                    destroy_framebuffer(device, framebuffer);
                }
                destroy_depth_cube_target(device, depth);
                return Err(error);
            }
        };
    }

    Ok(LocalShadowCube {
        depth,
        framebuffers,
        extent,
    })
}

fn create_shadow_cascade_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    translucent_render_pass: vk::RenderPass,
    translucent_depth_view: vk::ImageView,
) -> Result<ShadowCascade, VulkanError> {
    let transmittance = create_color_target(
        device,
        memory_properties,
        extent,
        TRANSLUCENT_SHADOW_FORMAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST,
    )?;
    let translucent_framebuffer = match create_translucent_shadow_framebuffer(
        device,
        translucent_render_pass,
        transmittance.view,
        translucent_depth_view,
        extent,
    ) {
        Ok(framebuffer) => framebuffer,
        Err(error) => {
            destroy_color_target(device, transmittance);
            return Err(error);
        }
    };

    Ok(ShadowCascade {
        transmittance,
        transmittance_state: ResourceState::Undefined,
        translucent_framebuffer,
    })
}

/// Extracts one image view per cascade without repeating Vec conversion boilerplate.
fn cascade_views<F>(cascades: &[ShadowCascade], select: F) -> [vk::ImageView; SHADOW_CASCADE_COUNT]
where
    F: Fn(&ShadowCascade) -> vk::ImageView,
{
    assert_eq!(
        cascades.len(),
        SHADOW_CASCADE_COUNT,
        "all shadow cascades must exist before creating descriptors"
    );
    std::array::from_fn(|index| select(&cascades[index]))
}

fn local_shadow_views(local: &[LocalShadowCube]) -> [vk::ImageView; MAX_LOCAL_LIGHTS] {
    assert_eq!(
        local.len(),
        MAX_LOCAL_LIGHTS,
        "all local shadow cubemaps must exist before creating descriptors"
    );
    std::array::from_fn(|index| local[index].depth.view)
}

/// Destroys one fixed shadow cascade after frame work has completed.
fn destroy_shadow_cascade(device: &Device, cascade: ShadowCascade) {
    destroy_framebuffer(device, cascade.translucent_framebuffer);
    destroy_color_target(device, cascade.transmittance);
}

fn destroy_local_shadow_cube(device: &Device, local: LocalShadowCube) {
    for framebuffer in local.framebuffers {
        destroy_framebuffer(device, framebuffer);
    }
    destroy_depth_cube_target(device, local.depth);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_mode_prefers_mailbox_over_fifo() {
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];

        assert_eq!(
            choose_present_mode(&modes).expect("present modes are available"),
            vk::PresentModeKHR::MAILBOX
        );
    }

    #[test]
    fn present_mode_falls_back_to_guaranteed_fifo() {
        assert_eq!(
            choose_present_mode(&[vk::PresentModeKHR::FIFO])
                .expect("FIFO is always supported by Vulkan surfaces"),
            vk::PresentModeKHR::FIFO
        );
        assert!(matches!(
            choose_present_mode(&[]),
            Err(VulkanError::PresentModesUnavailable)
        ));
    }
}
