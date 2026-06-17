use ash::{Device, khr, vk};

use crate::protocol::NonZeroExtent;

use super::{
    VulkanDevice, VulkanError,
    mesh::{MeshPassResources, MeshPipelineSet, VulkanMeshStore},
    post::PostPipeline,
    shadow_blur::ShadowMomentBlurPipeline,
    swapchain_pass::{
        create_post_framebuffer, create_post_render_pass, create_scene_framebuffer,
        create_scene_render_pass, create_shadow_blur_render_pass, create_shadow_framebuffer,
        create_shadow_render_pass, create_translucent_shadow_framebuffer,
        create_translucent_shadow_render_pass, destroy_framebuffer, destroy_render_pass,
    },
    swapchain_target::{
        ColorTarget, DepthTarget, create_color_target, create_depth_target, destroy_color_target,
        destroy_depth_target, destroy_image_view, initialize_shadow_sampler_fallback_images,
    },
};
use crate::renderer::graph::{
    FrameGraphInitialStates, GraphResource, ResourceState, SHADOW_CASCADE_COUNT,
    SHADOW_CASCADE_RESOURCES, SHADOW_MOMENT_BLUR_RESOURCES, SHADOW_MOMENT_RAW_RESOURCES,
    TRANSLUCENT_SHADOW_RESOURCES,
};
use crate::renderer::shadow_cascade_size;

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// Scene color is sampled by the post shader before tonemapping.
// Do not use the swapchain format here: B8G8R8A8_SRGB/RGBA8_UNORM clamps HDR PBR
// lighting to [0, 1], which makes close-camera specular highlights and reflections collapse.
const SCENE_COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Opaque metadata fits in UNORM8: oct-encoded normal.xy, roughness, reflectance.
const SCENE_NORMAL_ROUGHNESS_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

// Transparent metadata stores alpha as 1.0 + gl_FragCoord.z to mark valid pixels.
// UNORM8 would clamp that to 1.0, so post.frag's `transparent.w > 1.0` test can never work.
const SCENE_TRANSPARENT_NORMAL_ROUGHNESS_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

const SHADOW_MOMENT_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const TRANSLUCENT_SHADOW_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const FALLBACK_SHADOW_TRANSMITTANCE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

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
    post_render_pass: vk::RenderPass,
    mesh_pipeline: MeshPipelineSet,
    transparent_mesh_pipeline: MeshPipelineSet,
    post_pipeline: PostPipeline,
    scene_framebuffer: vk::Framebuffer,
    post_framebuffers: Vec<vk::Framebuffer>,
}

pub(super) struct ShadowResources {
    cascades: [ShadowCascade; SHADOW_CASCADE_COUNT],
    shadow_render_pass: vk::RenderPass,
    blur_render_pass: vk::RenderPass,
    translucent_render_pass: vk::RenderPass,
    mesh_pass_resources: MeshPassResources,
    translucent_pass_resources: MeshPassResources,
    shadow_pipeline: MeshPipelineSet,
    blur_pipeline: ShadowMomentBlurPipeline,
    translucent_pipeline: MeshPipelineSet,
}

pub(super) struct ShadowSamplerFallback {
    moments: ColorTarget,
    transmittance: ColorTarget,
    mesh_pass_resources: MeshPassResources,
}

struct ShadowCascade {
    moments: ColorTarget,
    blurred_moments: ColorTarget,
    filtered_moments: ColorTarget,
    depth: DepthTarget,
    transmittance: ColorTarget,
    raw_moment_state: ResourceState,
    blur_moment_state: ResourceState,
    moment_state: ResourceState,
    transmittance_state: ResourceState,
    shadow_framebuffer: vk::Framebuffer,
    blur_h_framebuffer: vk::Framebuffer,
    blur_v_framebuffer: vk::Framebuffer,
    translucent_framebuffer: vk::Framebuffer,
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

    /// Returns the tracked graph states for the scene MRT and depth attachments.
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
    scene_render_pass: Option<vk::RenderPass>,
    post_render_pass: Option<vk::RenderPass>,
    mesh_pipeline: Option<MeshPipelineSet>,
    transparent_mesh_pipeline: Option<MeshPipelineSet>,
    post_pipeline: Option<PostPipeline>,
    scene_framebuffer: Option<vk::Framebuffer>,
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
            scene_render_pass: None,
            post_render_pass: None,
            mesh_pipeline: None,
            transparent_mesh_pipeline: None,
            post_pipeline: None,
            scene_framebuffer: None,
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
            post_render_pass: take_created(&mut self.post_render_pass, "post render pass"),
            mesh_pipeline: take_created(&mut self.mesh_pipeline, "mesh pipeline"),
            transparent_mesh_pipeline: take_created(
                &mut self.transparent_mesh_pipeline,
                "transparent mesh pipeline",
            ),
            post_pipeline: take_created(&mut self.post_pipeline, "post pipeline"),
            scene_framebuffer: take_created(&mut self.scene_framebuffer, "scene framebuffer"),
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
        if let Some(framebuffer) = self.scene_framebuffer.take() {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(pipeline) = self.post_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.mesh_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.transparent_mesh_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        for render_pass in [self.post_render_pass.take(), self.scene_render_pass.take()]
            .into_iter()
            .flatten()
        {
            destroy_render_pass(&self.device.device, render_pass);
        }
        if let Some(depth) = self.scene_depth.take() {
            destroy_depth_target(&self.device.device, depth);
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
    cascades: Vec<ShadowCascade>,
    shadow_render_pass: Option<vk::RenderPass>,
    blur_render_pass: Option<vk::RenderPass>,
    translucent_render_pass: Option<vk::RenderPass>,
    mesh_pass_resources: Option<MeshPassResources>,
    translucent_pass_resources: Option<MeshPassResources>,
    shadow_pipeline: Option<MeshPipelineSet>,
    blur_pipeline: Option<ShadowMomentBlurPipeline>,
    translucent_pipeline: Option<MeshPipelineSet>,
    finished: bool,
}

impl<'a> ShadowBuild<'a> {
    /// Captures fixed-size shadow resources while device-level shadow setup is in progress.
    fn new(device: &'a VulkanDevice) -> Self {
        Self {
            device,
            cascades: Vec::with_capacity(SHADOW_CASCADE_COUNT),
            shadow_render_pass: None,
            blur_render_pass: None,
            translucent_render_pass: None,
            mesh_pass_resources: None,
            translucent_pass_resources: None,
            shadow_pipeline: None,
            blur_pipeline: None,
            translucent_pipeline: None,
            finished: false,
        }
    }

    /// Moves completed shadow resources into the device owner and disables failure cleanup.
    fn finish(mut self) -> ShadowResources {
        let cascades = std::mem::take(&mut self.cascades)
            .try_into()
            .unwrap_or_else(|_| panic!("all shadow cascades must be created before finish"));
        let resources = ShadowResources {
            cascades,
            shadow_render_pass: take_created(&mut self.shadow_render_pass, "shadow render pass"),
            blur_render_pass: take_created(&mut self.blur_render_pass, "shadow blur render pass"),
            translucent_render_pass: take_created(
                &mut self.translucent_render_pass,
                "translucent shadow render pass",
            ),
            mesh_pass_resources: take_created(
                &mut self.mesh_pass_resources,
                "shadow pass resources",
            ),
            translucent_pass_resources: take_created(
                &mut self.translucent_pass_resources,
                "translucent shadow pass resources",
            ),
            shadow_pipeline: take_created(&mut self.shadow_pipeline, "shadow pipeline"),
            blur_pipeline: take_created(&mut self.blur_pipeline, "shadow blur pipeline"),
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
        if let Some(resources) = self.translucent_pass_resources.take() {
            resources.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.translucent_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.blur_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.shadow_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        for cascade in self.cascades.drain(..) {
            destroy_shadow_cascade(&self.device.device, cascade);
        }
        for render_pass in [
            self.translucent_render_pass.take(),
            self.blur_render_pass.take(),
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
        let moments = create_color_target(
            device,
            memory_properties,
            extent,
            SHADOW_MOMENT_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )?;
        let transmittance = match create_color_target(
            device,
            memory_properties,
            extent,
            FALLBACK_SHADOW_TRANSMITTANCE_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        ) {
            Ok(target) => target,
            Err(error) => {
                destroy_color_target(device, moments);
                return Err(error);
            }
        };

        if let Err(error) = initialize_shadow_sampler_fallback_images(
            device,
            queue_family_index,
            queue,
            moments.image,
            transmittance.image,
        ) {
            destroy_color_target(device, transmittance);
            destroy_color_target(device, moments);
            return Err(error);
        }

        let shadow_views = [moments.view; SHADOW_CASCADE_COUNT];
        let translucent_views = [transmittance.view; SHADOW_CASCADE_COUNT];
        let mesh_pass_resources =
            match meshes.create_pass_resources(device, shadow_views, translucent_views) {
                Ok(resources) => resources,
                Err(error) => {
                    destroy_color_target(device, transmittance);
                    destroy_color_target(device, moments);
                    return Err(error);
                }
            };

        tracing::trace!("created tiny Vulkan shadow sampler fallback");
        Ok(ShadowSamplerFallback {
            moments,
            transmittance,
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
            SHADOW_MOMENT_FORMAT,
            DEPTH_FORMAT,
        )?);
        build.blur_render_pass = Some(create_shadow_blur_render_pass(
            &self.device,
            SHADOW_MOMENT_FORMAT,
        )?);
        build.translucent_render_pass = Some(create_translucent_shadow_render_pass(
            &self.device,
            TRANSLUCENT_SHADOW_FORMAT,
        )?);

        let shadow_render_pass = build
            .shadow_render_pass
            .expect("shadow render pass was just created");
        let blur_render_pass = build
            .blur_render_pass
            .expect("shadow blur render pass was just created");
        let translucent_render_pass = build
            .translucent_render_pass
            .expect("translucent shadow render pass was just created");

        for cascade_index in 0..SHADOW_CASCADE_COUNT {
            let extent = shadow_cascade_extent(cascade_index);
            let moments = create_color_target(
                &self.device,
                &self.memory_properties,
                extent,
                SHADOW_MOMENT_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )?;
            let blurred_moments = match create_color_target(
                &self.device,
                &self.memory_properties,
                extent,
                SHADOW_MOMENT_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ) {
                Ok(target) => target,
                Err(error) => {
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let filtered_moments = match create_color_target(
                &self.device,
                &self.memory_properties,
                extent,
                SHADOW_MOMENT_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ) {
                Ok(target) => target,
                Err(error) => {
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let depth = match create_depth_target(
                &self.device,
                &self.memory_properties,
                extent,
                DEPTH_FORMAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ) {
                Ok(target) => target,
                Err(error) => {
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let transmittance = match create_color_target(
                &self.device,
                &self.memory_properties,
                extent,
                TRANSLUCENT_SHADOW_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ) {
                Ok(target) => target,
                Err(error) => {
                    destroy_depth_target(&self.device, depth);
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let shadow_framebuffer = match create_shadow_framebuffer(
                &self.device,
                shadow_render_pass,
                moments.view,
                depth.view,
                extent,
            ) {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    destroy_color_target(&self.device, transmittance);
                    destroy_depth_target(&self.device, depth);
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let blur_h_framebuffer = match create_post_framebuffer(
                &self.device,
                blur_render_pass,
                blurred_moments.view,
                extent,
            ) {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    destroy_framebuffer(&self.device, shadow_framebuffer);
                    destroy_color_target(&self.device, transmittance);
                    destroy_depth_target(&self.device, depth);
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let blur_v_framebuffer = match create_post_framebuffer(
                &self.device,
                blur_render_pass,
                filtered_moments.view,
                extent,
            ) {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    destroy_framebuffer(&self.device, blur_h_framebuffer);
                    destroy_framebuffer(&self.device, shadow_framebuffer);
                    destroy_color_target(&self.device, transmittance);
                    destroy_depth_target(&self.device, depth);
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };
            let translucent_framebuffer = match create_translucent_shadow_framebuffer(
                &self.device,
                translucent_render_pass,
                transmittance.view,
                extent,
            ) {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    destroy_framebuffer(&self.device, blur_v_framebuffer);
                    destroy_framebuffer(&self.device, blur_h_framebuffer);
                    destroy_framebuffer(&self.device, shadow_framebuffer);
                    destroy_color_target(&self.device, transmittance);
                    destroy_depth_target(&self.device, depth);
                    destroy_color_target(&self.device, filtered_moments);
                    destroy_color_target(&self.device, blurred_moments);
                    destroy_color_target(&self.device, moments);
                    return Err(error);
                }
            };

            build.cascades.push(ShadowCascade {
                moments,
                blurred_moments,
                filtered_moments,
                depth,
                transmittance,
                raw_moment_state: ResourceState::Undefined,
                blur_moment_state: ResourceState::Undefined,
                moment_state: ResourceState::Undefined,
                transmittance_state: ResourceState::Undefined,
                shadow_framebuffer,
                blur_h_framebuffer,
                blur_v_framebuffer,
                translucent_framebuffer,
                extent,
            });
        }

        let shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT] = build
            .cascades
            .iter()
            .map(|cascade| cascade.filtered_moments.view)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("all shadow cascade views must exist"));
        let blur_h_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT] = build
            .cascades
            .iter()
            .map(|cascade| cascade.moments.view)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("all raw shadow cascade views must exist"));
        let blur_v_source_views: [vk::ImageView; SHADOW_CASCADE_COUNT] = build
            .cascades
            .iter()
            .map(|cascade| cascade.blurred_moments.view)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("all blurred shadow cascade views must exist"));
        let translucent_views: [vk::ImageView; SHADOW_CASCADE_COUNT] = build
            .cascades
            .iter()
            .map(|cascade| cascade.transmittance.view)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("all translucent shadow views must exist"));

        build.mesh_pass_resources = Some(self.meshes.create_pass_resources(
            &self.device,
            shadow_views,
            translucent_views,
        )?);
        build.translucent_pass_resources = Some(self.meshes.create_pass_resources(
            &self.device,
            blur_h_source_views,
            translucent_views,
        )?);
        build.blur_pipeline = Some(ShadowMomentBlurPipeline::create(
            &self.device,
            blur_render_pass,
            blur_h_source_views,
            blur_v_source_views,
        )?);
        build.shadow_pipeline = Some(
            self.meshes
                .create_shadow_pipeline_set(&self.device, shadow_render_pass)?,
        );
        build.translucent_pipeline = Some(
            self.meshes
                .create_translucent_shadow_pipeline_set(&self.device, translucent_render_pass)?,
        );

        tracing::info!(
            cascade_count = SHADOW_CASCADE_COUNT,
            cascade_0_size = build.cascades[0].extent.width(),
            cascade_1_size = build.cascades[1].extent.width(),
            cascade_2_size = build.cascades[2].extent.width(),
            cascade_3_size = build.cascades[3].extent.width(),
            translucent_format = ?TRANSLUCENT_SHADOW_FORMAT,
            "created fixed Vulkan shadow resources"
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
        build.post_render_pass = Some(create_post_render_pass(&self.device, config.format)?);

        let scene_render_pass = build
            .scene_render_pass
            .expect("scene render pass was just created");
        let post_render_pass = build
            .post_render_pass
            .expect("post render pass was just created");

        build.mesh_pipeline = Some(
            self.meshes
                .create_scene_pipeline_set(&self.device, scene_render_pass)?,
        );
        build.transparent_mesh_pipeline = Some(
            self.meshes
                .create_scene_transparent_pipeline_set(&self.device, scene_render_pass)?,
        );
        build.post_pipeline = Some(PostPipeline::create(
            &self.device,
            post_render_pass,
            scene_color.view,
            scene_depth.view,
            scene_normal_roughness.view,
            scene_transparent_normal_roughness.view,
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
        build.post_framebuffers =
            self.create_post_framebuffers(post_render_pass, &build.image_views, config.extent)?;

        tracing::info!(
            width = config.extent.width(),
            height = config.extent.height(),
            image_count = build.images.len(),
            image_view_count = build.image_views.len(),
            framebuffer_count = build.post_framebuffers.len() + 1,
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
            framebuffer_count = swapchain.post_framebuffers.len() + 1,
            format = ?swapchain.format,
            color_space = ?swapchain.color_space,
            present_mode = ?swapchain.present_mode,
            "destroying Vulkan swapchain"
        );

        self.destroy_framebuffers(swapchain.post_framebuffers);
        destroy_framebuffer(&self.device, swapchain.scene_framebuffer);
        swapchain.post_pipeline.destroy(&self.device);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.mesh_pipeline);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.transparent_mesh_pipeline);
        destroy_render_pass(&self.device, swapchain.post_render_pass);
        destroy_render_pass(&self.device, swapchain.scene_render_pass);
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

    /// Returns the post render pass that writes the acquired swapchain image.
    pub(super) fn post_render_pass(&self) -> vk::RenderPass {
        self.post_render_pass
    }

    /// Returns the mesh graphics pipeline compatible with the scene pass.
    pub(super) fn mesh_pipeline(&self) -> MeshPipelineSet {
        self.mesh_pipeline
    }

    /// Returns the mesh graphics pipeline that blends transparent scene materials.
    pub(super) fn transparent_mesh_pipeline(&self) -> MeshPipelineSet {
        self.transparent_mesh_pipeline
    }

    /// Returns the post pipeline compatible with this swapchain's post pass.
    pub(super) fn post_pipeline(&self) -> &PostPipeline {
        &self.post_pipeline
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

    /// Returns the current graph resource states before compiling a frame graph.
    pub(super) fn graph_initial_states(
        &self,
        image_index: u32,
        shadows: Option<&ShadowResources>,
    ) -> Result<FrameGraphInitialStates, VulkanError> {
        let shadow_moment_states = shadows.map_or(
            [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
            ShadowResources::moment_states,
        );
        let shadow_raw_moment_states = shadows.map_or(
            [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
            ShadowResources::raw_moment_states,
        );
        let shadow_blur_moment_states = shadows.map_or(
            [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
            ShadowResources::blur_moment_states,
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
        Ok(FrameGraphInitialStates::new(
            self.image_state(image_index)?,
            shadow_raw_moment_states,
            shadow_blur_moment_states,
            shadow_moment_states,
            translucent_shadow_states,
            scene_color_state,
            scene_normal_roughness_state,
            scene_transparent_normal_roughness_state,
            scene_depth_state,
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

        match resource {
            GraphResource::SwapchainImage => Ok((
                self.image_for_index(image_index)?,
                vk::ImageAspectFlags::COLOR,
            )),
            GraphResource::SceneColor
            | GraphResource::SceneNormalRoughness
            | GraphResource::SceneTransparentNormalRoughness
            | GraphResource::SceneDepth => {
                unreachable!("scene resources return early above")
            }
            GraphResource::ShadowMomentRaw0
            | GraphResource::ShadowMomentRaw1
            | GraphResource::ShadowMomentRaw2
            | GraphResource::ShadowMomentRaw3
            | GraphResource::ShadowMomentBlur0
            | GraphResource::ShadowMomentBlur1
            | GraphResource::ShadowMomentBlur2
            | GraphResource::ShadowMomentBlur3
            | GraphResource::ShadowCascade0
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

    /// Returns the render pass that writes one separable moment blur target.
    pub(super) fn blur_render_pass(&self) -> vk::RenderPass {
        self.blur_render_pass
    }

    /// Returns the render pass that writes one translucent cascade transmittance target.
    pub(super) fn translucent_render_pass(&self) -> vk::RenderPass {
        self.translucent_render_pass
    }

    /// Returns the opaque shadow mesh pipelines shared by all cascades.
    pub(super) fn shadow_pipeline(&self) -> MeshPipelineSet {
        self.shadow_pipeline
    }

    /// Returns the fullscreen separable blur pipeline for shadow moments.
    pub(super) fn blur_pipeline(&self) -> &ShadowMomentBlurPipeline {
        &self.blur_pipeline
    }

    /// Returns the translucent shadow mesh pipelines shared by all cascades.
    pub(super) fn translucent_pipeline(&self) -> MeshPipelineSet {
        self.translucent_pipeline
    }

    /// Returns descriptors used by scene mesh shaders to sample every shadow cascade.
    pub(super) fn mesh_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
    }

    /// Returns descriptors used by translucent shadow shaders to test unfiltered opaque depth.
    pub(super) fn translucent_pass_resources(&self) -> &MeshPassResources {
        &self.translucent_pass_resources
    }

    /// Returns the fixed render extent for one cascade index.
    pub(super) fn extent_2d(&self, cascade_index: usize) -> Result<vk::Extent2D, VulkanError> {
        let extent = self.cascade(cascade_index)?.extent;
        Ok(vk::Extent2D {
            width: extent.width(),
            height: extent.height(),
        })
    }

    /// Returns the framebuffer used to render one opaque shadow cascade.
    pub(super) fn shadow_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        Ok(self.cascade(cascade_index)?.shadow_framebuffer)
    }

    /// Returns the framebuffer that receives horizontal moment blur for one cascade.
    pub(super) fn blur_h_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        Ok(self.cascade(cascade_index)?.blur_h_framebuffer)
    }

    /// Returns the framebuffer that receives the filtered moment map for one cascade.
    pub(super) fn blur_v_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        Ok(self.cascade(cascade_index)?.blur_v_framebuffer)
    }

    /// Returns the framebuffer used to render one translucent shadow cascade.
    pub(super) fn translucent_framebuffer(
        &self,
        cascade_index: usize,
    ) -> Result<vk::Framebuffer, VulkanError> {
        Ok(self.cascade(cascade_index)?.translucent_framebuffer)
    }

    /// Returns tracked raw moment-map layouts for graph compilation.
    fn raw_moment_states(&self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        std::array::from_fn(|index| self.cascades[index].raw_moment_state)
    }

    /// Returns tracked horizontal blur scratch layouts for graph compilation.
    fn blur_moment_states(&self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        std::array::from_fn(|index| self.cascades[index].blur_moment_state)
    }

    /// Returns tracked moment-map layouts for graph compilation.
    fn moment_states(&self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        std::array::from_fn(|index| self.cascades[index].moment_state)
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
        for (cascade, resource) in self.cascades.iter_mut().zip(SHADOW_MOMENT_RAW_RESOURCES) {
            if let Some(state) = plan.final_state_for(resource) {
                cascade.raw_moment_state = state;
            }
        }
        for (cascade, resource) in self.cascades.iter_mut().zip(SHADOW_MOMENT_BLUR_RESOURCES) {
            if let Some(state) = plan.final_state_for(resource) {
                cascade.blur_moment_state = state;
            }
        }
        for (cascade, resource) in self.cascades.iter_mut().zip(SHADOW_CASCADE_RESOURCES) {
            if let Some(state) = plan.final_state_for(resource) {
                cascade.moment_state = state;
            }
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
        SHADOW_MOMENT_RAW_RESOURCES
            .iter()
            .position(|candidate| *candidate == resource)
            .map(|index| {
                (
                    self.cascades[index].moments.image,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .or_else(|| {
                SHADOW_MOMENT_BLUR_RESOURCES
                    .iter()
                    .position(|candidate| *candidate == resource)
                    .map(|index| {
                        (
                            self.cascades[index].blurred_moments.image,
                            vk::ImageAspectFlags::COLOR,
                        )
                    })
            })
            .or_else(|| {
                SHADOW_CASCADE_RESOURCES
                    .iter()
                    .position(|candidate| *candidate == resource)
                    .map(|index| {
                        (
                            self.cascades[index].filtered_moments.image,
                            vk::ImageAspectFlags::COLOR,
                        )
                    })
            })
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

    /// Releases fixed shadow framebuffers, descriptors, pipelines, render passes, and targets.
    fn destroy(self, device: &Device, meshes: &VulkanMeshStore) {
        self.translucent_pass_resources.destroy(device);
        self.mesh_pass_resources.destroy(device);
        meshes.destroy_pipeline_set(device, self.translucent_pipeline);
        self.blur_pipeline.destroy(device);
        meshes.destroy_pipeline_set(device, self.shadow_pipeline);
        for cascade in self.cascades {
            destroy_shadow_cascade(device, cascade);
        }
        destroy_render_pass(device, self.translucent_render_pass);
        destroy_render_pass(device, self.blur_render_pass);
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
}

impl ShadowSamplerFallback {
    /// Returns descriptors that bind full-light dummy shadow maps at the mesh pass set.
    pub(super) fn mesh_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
    }

    /// Releases the dummy descriptor set and its tiny sampled images.
    pub(super) fn destroy(self, device: &Device) {
        self.mesh_pass_resources.destroy(device);
        destroy_color_target(device, self.transmittance);
        destroy_color_target(device, self.moments);
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
        .find(|mode| *mode == vk::PresentModeKHR::FIFO) // V-Sync
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

fn shadow_cascade_extent(cascade_index: usize) -> NonZeroExtent {
    let size = shadow_cascade_size(cascade_index);
    NonZeroExtent::new(size, size).expect("shadow map extent must be non-zero")
}

/// Destroys one fixed shadow cascade after frame work has completed.
fn destroy_shadow_cascade(device: &Device, cascade: ShadowCascade) {
    destroy_framebuffer(device, cascade.translucent_framebuffer);
    destroy_framebuffer(device, cascade.blur_v_framebuffer);
    destroy_framebuffer(device, cascade.blur_h_framebuffer);
    destroy_framebuffer(device, cascade.shadow_framebuffer);
    destroy_color_target(device, cascade.transmittance);
    destroy_depth_target(device, cascade.depth);
    destroy_color_target(device, cascade.filtered_moments);
    destroy_color_target(device, cascade.blurred_moments);
    destroy_color_target(device, cascade.moments);
}
