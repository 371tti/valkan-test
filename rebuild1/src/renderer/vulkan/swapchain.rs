use ash::{Device, khr, vk};

use crate::protocol::NonZeroExtent;

use super::{
    VulkanDevice, VulkanError,
    buffer::find_memory_type,
    mesh::{MeshPassResources, MeshPipelineSet},
    post::PostPipeline,
    triangle::DebugTrianglePipeline,
};
use crate::renderer::SHADOW_MAP_SIZE;
use crate::renderer::graph::{FrameGraphInitialStates, GraphResource, ResourceState};

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

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
    shadow_map: DepthTarget,
    scene_color: ColorTarget,
    scene_depth: DepthTarget,
    shadow_map_state: ResourceState,
    scene_color_state: ResourceState,
    scene_depth_state: ResourceState,
    shadow_render_pass: vk::RenderPass,
    scene_render_pass: vk::RenderPass,
    post_render_pass: vk::RenderPass,
    debug_triangle_pipeline: DebugTrianglePipeline,
    mesh_pass_resources: MeshPassResources,
    shadow_pipeline: MeshPipelineSet,
    mesh_pipeline: MeshPipelineSet,
    post_pipeline: PostPipeline,
    shadow_framebuffer: vk::Framebuffer,
    scene_framebuffer: vk::Framebuffer,
    post_framebuffers: Vec<vk::Framebuffer>,
}

struct ColorTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
}

struct DepthTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
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
    shadow_map: Option<DepthTarget>,
    scene_color: Option<ColorTarget>,
    scene_depth: Option<DepthTarget>,
    shadow_render_pass: Option<vk::RenderPass>,
    scene_render_pass: Option<vk::RenderPass>,
    post_render_pass: Option<vk::RenderPass>,
    debug_triangle_pipeline: Option<DebugTrianglePipeline>,
    mesh_pass_resources: Option<MeshPassResources>,
    shadow_pipeline: Option<MeshPipelineSet>,
    mesh_pipeline: Option<MeshPipelineSet>,
    post_pipeline: Option<PostPipeline>,
    shadow_framebuffer: Option<vk::Framebuffer>,
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
            shadow_map: None,
            scene_color: None,
            scene_depth: None,
            shadow_render_pass: None,
            scene_render_pass: None,
            post_render_pass: None,
            debug_triangle_pipeline: None,
            mesh_pass_resources: None,
            shadow_pipeline: None,
            mesh_pipeline: None,
            post_pipeline: None,
            shadow_framebuffer: None,
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
            shadow_map: take_created(&mut self.shadow_map, "shadow map"),
            scene_color: take_created(&mut self.scene_color, "scene color"),
            scene_depth: take_created(&mut self.scene_depth, "scene depth"),
            shadow_map_state: ResourceState::Undefined,
            scene_color_state: ResourceState::Undefined,
            scene_depth_state: ResourceState::Undefined,
            shadow_render_pass: take_created(&mut self.shadow_render_pass, "shadow render pass"),
            scene_render_pass: take_created(&mut self.scene_render_pass, "scene render pass"),
            post_render_pass: take_created(&mut self.post_render_pass, "post render pass"),
            debug_triangle_pipeline: take_created(
                &mut self.debug_triangle_pipeline,
                "debug triangle pipeline",
            ),
            mesh_pass_resources: take_created(&mut self.mesh_pass_resources, "mesh pass resources"),
            shadow_pipeline: take_created(&mut self.shadow_pipeline, "shadow pipeline"),
            mesh_pipeline: take_created(&mut self.mesh_pipeline, "mesh pipeline"),
            post_pipeline: take_created(&mut self.post_pipeline, "post pipeline"),
            shadow_framebuffer: take_created(&mut self.shadow_framebuffer, "shadow framebuffer"),
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
        if let Some(framebuffer) = self.shadow_framebuffer.take() {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(framebuffer) = self.scene_framebuffer.take() {
            destroy_framebuffer(&self.device.device, framebuffer);
        }
        if let Some(pipeline) = self.post_pipeline.take() {
            pipeline.destroy(&self.device.device);
        }
        if let Some(resources) = self.mesh_pass_resources.take() {
            resources.destroy(&self.device.device);
        }
        if let Some(pipeline) = self.mesh_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.shadow_pipeline.take() {
            self.device
                .meshes
                .destroy_pipeline_set(&self.device.device, pipeline);
        }
        if let Some(pipeline) = self.debug_triangle_pipeline.take() {
            self.device
                .debug_triangle
                .destroy_pipeline(&self.device.device, pipeline);
        }
        for render_pass in [
            self.post_render_pass.take(),
            self.shadow_render_pass.take(),
            self.scene_render_pass.take(),
        ]
        .into_iter()
        .flatten()
        {
            destroy_render_pass(&self.device.device, render_pass);
        }
        if let Some(depth) = self.shadow_map.take() {
            destroy_depth_target(&self.device.device, depth);
        }
        if let Some(depth) = self.scene_depth.take() {
            destroy_depth_target(&self.device.device, depth);
        }
        if let Some(color) = self.scene_color.take() {
            destroy_color_target(&self.device.device, color);
        }
        self.device
            .destroy_image_views(std::mem::take(&mut self.image_views));
        self.device.destroy_swapchain_handle(self.handle);
    }
}

/// Takes a required build resource when swapchain construction reaches the success path.
fn take_created<T>(slot: &mut Option<T>, name: &'static str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("{name} must be created before swapchain finish"))
}

impl VulkanDevice {
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
            config.format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);
        build.scene_depth = Some(create_depth_target(
            &self.device,
            &self.memory_properties,
            config.extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?);
        build.shadow_map = Some(create_depth_target(
            &self.device,
            &self.memory_properties,
            shadow_extent(),
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?);

        let scene_color = build
            .scene_color
            .as_ref()
            .expect("scene color was just created");
        let scene_depth = build
            .scene_depth
            .as_ref()
            .expect("scene depth was just created");
        let shadow_map = build
            .shadow_map
            .as_ref()
            .expect("shadow map was just created");

        build.scene_render_pass = Some(create_scene_render_pass(
            &self.device,
            scene_color.format,
            scene_depth.format,
        )?);
        build.shadow_render_pass =
            Some(create_shadow_render_pass(&self.device, shadow_map.format)?);
        build.post_render_pass = Some(create_post_render_pass(&self.device, config.format)?);

        let scene_render_pass = build
            .scene_render_pass
            .expect("scene render pass was just created");
        let shadow_render_pass = build
            .shadow_render_pass
            .expect("shadow render pass was just created");
        let post_render_pass = build
            .post_render_pass
            .expect("post render pass was just created");

        build.debug_triangle_pipeline = Some(
            self.debug_triangle
                .create_pipeline(&self.device, scene_render_pass)?,
        );
        build.mesh_pass_resources = Some(
            self.meshes
                .create_pass_resources(&self.device, shadow_map.view)?,
        );
        build.shadow_pipeline = Some(
            self.meshes
                .create_shadow_pipeline_set(&self.device, shadow_render_pass)?,
        );
        build.mesh_pipeline = Some(
            self.meshes
                .create_scene_pipeline_set(&self.device, scene_render_pass)?,
        );
        build.post_pipeline = Some(PostPipeline::create(
            &self.device,
            post_render_pass,
            scene_color.view,
        )?);

        build.scene_framebuffer = Some(create_scene_framebuffer(
            &self.device,
            scene_render_pass,
            scene_color.view,
            scene_depth.view,
            config.extent,
        )?);
        build.shadow_framebuffer = Some(create_depth_framebuffer(
            &self.device,
            shadow_render_pass,
            shadow_map.view,
            shadow_extent(),
        )?);
        build.post_framebuffers =
            self.create_post_framebuffers(post_render_pass, &build.image_views, config.extent)?;

        tracing::info!(
            width = config.extent.width(),
            height = config.extent.height(),
            image_count = build.images.len(),
            image_view_count = build.image_views.len(),
            framebuffer_count = build.post_framebuffers.len() + 2,
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
            framebuffer_count = swapchain.post_framebuffers.len() + 2,
            format = ?swapchain.format,
            color_space = ?swapchain.color_space,
            present_mode = ?swapchain.present_mode,
            "destroying Vulkan swapchain"
        );

        self.destroy_framebuffers(swapchain.post_framebuffers);
        destroy_framebuffer(&self.device, swapchain.shadow_framebuffer);
        destroy_framebuffer(&self.device, swapchain.scene_framebuffer);
        swapchain.post_pipeline.destroy(&self.device);
        swapchain.mesh_pass_resources.destroy(&self.device);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.mesh_pipeline);
        self.meshes
            .destroy_pipeline_set(&self.device, swapchain.shadow_pipeline);
        self.debug_triangle
            .destroy_pipeline(&self.device, swapchain.debug_triangle_pipeline);
        destroy_render_pass(&self.device, swapchain.post_render_pass);
        destroy_render_pass(&self.device, swapchain.shadow_render_pass);
        destroy_render_pass(&self.device, swapchain.scene_render_pass);
        destroy_depth_target(&self.device, swapchain.shadow_map);
        destroy_depth_target(&self.device, swapchain.scene_depth);
        destroy_color_target(&self.device, swapchain.scene_color);
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
    /// Returns the shadow render pass that writes the graph-owned shadow map.
    pub(super) fn shadow_render_pass(&self) -> vk::RenderPass {
        self.shadow_render_pass
    }

    /// Returns the scene render pass that writes scene color and depth targets.
    pub(super) fn scene_render_pass(&self) -> vk::RenderPass {
        self.scene_render_pass
    }

    /// Returns the post render pass that writes the acquired swapchain image.
    pub(super) fn post_render_pass(&self) -> vk::RenderPass {
        self.post_render_pass
    }

    /// Returns the debug triangle graphics pipeline compatible with the scene pass.
    pub(super) fn debug_triangle_pipeline(&self) -> DebugTrianglePipeline {
        self.debug_triangle_pipeline
    }

    /// Returns the mesh graphics pipeline compatible with the scene pass.
    pub(super) fn mesh_pipeline(&self) -> MeshPipelineSet {
        self.mesh_pipeline
    }

    /// Returns the mesh pipeline compatible with the shadow pass.
    pub(super) fn shadow_pipeline(&self) -> MeshPipelineSet {
        self.shadow_pipeline
    }

    /// Returns descriptors used by scene mesh shaders to sample graph-owned targets.
    pub(super) fn mesh_pass_resources(&self) -> &MeshPassResources {
        &self.mesh_pass_resources
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

    /// Returns the framebuffer used by the graph shadow pass.
    pub(super) fn shadow_framebuffer(&self) -> vk::Framebuffer {
        self.shadow_framebuffer
    }

    /// Returns the fixed-resolution shadow-map extent used for shadow pass recording.
    pub(super) fn shadow_extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
        }
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
    ) -> Result<FrameGraphInitialStates, VulkanError> {
        Ok(FrameGraphInitialStates::new(
            self.image_state(image_index)?,
            self.shadow_map_state,
            self.scene_color_state,
            self.scene_depth_state,
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
        if let Some(state) = plan.final_state_for(GraphResource::SceneColor) {
            self.scene_color_state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::SceneDepth) {
            self.scene_depth_state = state;
        }
        if let Some(state) = plan.final_state_for(GraphResource::ShadowMap) {
            self.shadow_map_state = state;
        }

        Ok(())
    }

    /// Returns the image and aspect range used by one graph resource barrier.
    pub(super) fn graph_image(
        &self,
        resource: GraphResource,
        image_index: u32,
    ) -> Result<(vk::Image, vk::ImageAspectFlags), VulkanError> {
        match resource {
            GraphResource::SwapchainImage => Ok((
                self.image_for_index(image_index)?,
                vk::ImageAspectFlags::COLOR,
            )),
            GraphResource::ShadowMap => Ok((self.shadow_map.image, vk::ImageAspectFlags::DEPTH)),
            GraphResource::SceneColor => Ok((self.scene_color.image, vk::ImageAspectFlags::COLOR)),
            GraphResource::SceneDepth => Ok((self.scene_depth.image, vk::ImageAspectFlags::DEPTH)),
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

/// Destroys one swapchain image view.
fn destroy_image_view(device: &Device, image_view: vk::ImageView) {
    if image_view == vk::ImageView::null() {
        return;
    }

    // Safety: the image view was created by this device and is destroyed exactly once.
    unsafe {
        device.destroy_image_view(image_view, None);
    }
}

/// Creates the color/depth render pass used by the graph scene pass.
fn create_scene_render_pass(
    device: &Device,
    format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass, VulkanError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref);
    let attachments = [color_attachment, depth_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all slices in `create_info` live for the duration of the call, and graph barriers
    // transition the swapchain image into and out of the color attachment layout.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the depth-only render pass used to populate the graph shadow map.
fn create_shadow_render_pass(
    device: &Device,
    depth_format: vk::Format,
) -> Result<vk::RenderPass, VulkanError> {
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_attachment_ref);
    let attachments = [depth_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all create-info slices live for this call and graph barriers control depth layout.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the color-only render pass that writes a post result into the swapchain image.
fn create_post_render_pass(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, VulkanError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs);
    let attachments = [color_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all create-info slices are local and graph barriers handle image layout changes.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Destroys one swapchain render pass after its framebuffers are gone.
fn destroy_render_pass(device: &Device, render_pass: vk::RenderPass) {
    if render_pass == vk::RenderPass::null() {
        return;
    }

    // Safety: the render pass was created by this device and is destroyed after its framebuffers.
    unsafe {
        device.destroy_render_pass(render_pass, None);
    }
}

/// Creates the scene framebuffer that binds scene color and scene depth views.
fn create_scene_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    color_view: vk::ImageView,
    depth_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, VulkanError> {
    let attachments = [color_view, depth_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: both image views match the render pass attachments and outlive the framebuffer.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the shadow framebuffer that binds only the shadow-map depth view.
fn create_depth_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    depth_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, VulkanError> {
    let attachments = [depth_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: the depth image view matches the depth-only render pass attachment.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates one post framebuffer for a swapchain image view.
fn create_post_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    image_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, VulkanError> {
    let attachments = [image_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: the image view matches the color-only post render pass.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Destroys one framebuffer created for a swapchain image view.
fn destroy_framebuffer(device: &Device, framebuffer: vk::Framebuffer) {
    if framebuffer == vk::Framebuffer::null() {
        return;
    }

    // Safety: the framebuffer was created by this device and is destroyed exactly once.
    unsafe {
        device.destroy_framebuffer(framebuffer, None);
    }
}

/// Creates a graph-owned color target image, memory allocation, and view.
fn create_color_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<ColorTarget, VulkanError> {
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width(),
            height: extent.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    // Safety: image create info contains only local values and no custom allocator is used.
    let image = unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)?;
    let memory = match allocate_image_memory(device, memory_properties, image) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_image(device, image);
            return Err(error);
        }
    };
    // Safety: the allocation satisfies the requirements returned for this image.
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        free_memory(device, memory);
        destroy_image(device, image);
        return Err(VulkanError::Vk(error));
    }
    let view = match create_color_image_view(device, image, format) {
        Ok(view) => view,
        Err(error) => {
            free_memory(device, memory);
            destroy_image(device, image);
            return Err(error);
        }
    };

    tracing::trace!(
        width = extent.width(),
        height = extent.height(),
        format = ?format,
        usage = ?usage,
        "created Vulkan color target"
    );
    Ok(ColorTarget {
        image,
        memory,
        view,
        format,
    })
}

/// Creates a 2D color view for a graph-owned color target.
fn create_color_image_view(
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
        .subresource_range(subresource_range);

    // Safety: the image is a 2D color image created by this device.
    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Destroys a graph-owned color target after the device is idle.
fn destroy_color_target(device: &Device, color: ColorTarget) {
    destroy_image_view(device, color.view);
    free_memory(device, color.memory);
    destroy_image(device, color.image);
}

/// Creates the depth image, memory, and view shared by the graph scene pass.
fn create_depth_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<DepthTarget, VulkanError> {
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width(),
            height: extent.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    // Safety: image create info contains only local values and no custom allocator is used.
    let image = unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)?;
    let memory = match allocate_image_memory(device, memory_properties, image) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_image(device, image);
            return Err(error);
        }
    };
    // Safety: the allocation satisfies the memory requirements returned for this image.
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        free_memory(device, memory);
        destroy_image(device, image);
        return Err(VulkanError::Vk(error));
    }
    let view = match create_depth_image_view(device, image, format) {
        Ok(view) => view,
        Err(error) => {
            free_memory(device, memory);
            destroy_image(device, image);
            return Err(error);
        }
    };

    tracing::trace!(
        width = extent.width(),
        height = extent.height(),
        format = ?format,
        usage = ?usage,
        "created Vulkan depth target"
    );
    Ok(DepthTarget {
        image,
        memory,
        view,
        format,
    })
}

fn shadow_extent() -> NonZeroExtent {
    NonZeroExtent::new(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
        .expect("shadow map extent must be non-zero")
}

/// Allocates device-local memory compatible with a depth image.
fn allocate_image_memory(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
) -> Result<vk::DeviceMemory, VulkanError> {
    // Safety: the image was created by this device and is alive for the requirement query.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);

    // Safety: the memory type index was selected from this physical device's properties.
    unsafe { device.allocate_memory(&allocate_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the view used by the depth attachment.
fn create_depth_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, VulkanError> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    // Safety: the image is a depth image created by this device and the view is 2D depth-only.
    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Destroys a depth target after all framebuffers that reference it are gone.
fn destroy_depth_target(device: &Device, depth: DepthTarget) {
    destroy_image_view(device, depth.view);
    free_memory(device, depth.memory);
    destroy_image(device, depth.image);
}

/// Destroys one raw image handle.
fn destroy_image(device: &Device, image: vk::Image) {
    if image == vk::Image::null() {
        return;
    }

    // Safety: the image was created by this device and is destroyed after GPU idle.
    unsafe {
        device.destroy_image(image, None);
    }
}

/// Frees one image memory allocation.
fn free_memory(device: &Device, memory: vk::DeviceMemory) {
    if memory == vk::DeviceMemory::null() {
        return;
    }

    // Safety: the allocation belongs to this device and is no longer bound to live work.
    unsafe {
        device.free_memory(memory, None);
    }
}
