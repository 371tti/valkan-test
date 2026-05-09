use std::{path::Path, sync::Arc};

use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::window::Window;

use crate::renderer::init::{create_swapchain, create_swapchain_state};
use crate::renderer::math::probe_binding_index;
use crate::renderer::uniforms::SceneUniform;
use assets::{DepthTarget, GpuAssets, PlanarReflectionTarget, ReflectionProbe, SceneBindings};

mod assets;
#[path = "passes/draw.rs"]
mod draw;
#[path = "lifecycle/drop.rs"]
mod drop;
#[path = "lifecycle/init.rs"]
mod init;
mod math;
mod pipeline;
#[path = "passes/reflections.rs"]
mod reflections;
mod scene;
mod uniforms;
pub use assets::{
    CpuMesh, CpuModel, CpuPrimitive, CpuTexture, TextureFilter, TextureSampler, TextureWrap,
};
pub use pipeline::{
    ColorBlendConfig, DepthConfig, GraphicsPipeline, HotReload, ModelVertex, PipelineDesc,
    PipelineError, RasterizationConfig, ShaderCode, ShaderSet, VertexAttribute, VertexLayout,
    create_pipeline_cache,
};
pub use scene::{
    BoxReflectionSettings, Camera, DirectionalLight, Material, MaterialId, MeshId, ModelId,
    PipelineId, PlanarReflectionSettings, ReflectionSettings, RenderModel, RenderObject,
    RenderScene, SceneContext, SceneController, SceneKey, SceneMessage, TextureId, Transform,
    mat4_mul,
};

/// CPUがGPU完了を待たずに先行して準備できるフレーム数(1..4程度が一般的)
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const MAX_EMISSIVE_LIGHTS: usize = 8;
const REFLECTION_PROBE_SIZE: u32 = 256;
const PLANAR_REFLECTION_MIN_SIZE: u32 = 64;
const PLANAR_REFLECTION_MAX_SIZE: u32 = 4096;

struct SwapchainState {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
    depth: DepthTarget,
    command_buffers: Vec<vk::CommandBuffer>,
    image_layouts: Vec<vk::ImageLayout>,
    images_in_flight: Vec<vk::Fence>,
    render_finished_semaphores: Vec<vk::Semaphore>,
}

struct FrameSync {
    image_available: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

struct SyncState {
    frames: Vec<FrameSync>,
    current_frame: usize,
}

impl SyncState {
    fn advance_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frames.len();
    }
}

struct PipelineSlot {
    desc: PipelineDesc,
    pipeline: GraphicsPipeline,
    hot_reload: HotReload,
}

fn material_texture_slot_count(material: Material) -> usize {
    [
        material.base_color_texture,
        material.metallic_roughness_texture,
        material.normal_texture,
        material.occlusion_texture,
        material.emissive_texture,
    ]
    .into_iter()
    .filter(Option::is_some)
    .count()
}

fn log_model_stats(path: &Path, model: &CpuModel) {
    let texture_slots = model
        .primitives
        .iter()
        .map(|primitive| material_texture_slot_count(primitive.material))
        .sum::<usize>();

    log::info!(
        "renderer: loaded model '{}': primitives={}, textures={}, material_texture_slots={}",
        path.display(),
        model.primitives.len(),
        model.textures.len(),
        texture_slots
    );

    if model.textures.is_empty() {
        log::warn!(
            "renderer: model '{}' has no image textures; only material factors will be drawn",
            path.display()
        );
    } else if texture_slots == 0 {
        log::warn!(
            "renderer: model '{}' contains textures but no material texture references",
            path.display()
        );
    }
}

pub struct Renderer {
    _entry: Entry,
    window_ref: Arc<Window>,

    config: RendererConfig,
    instance: Instance,

    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,

    logical_device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: SwapchainState,
    pipeline_cache: vk::PipelineCache,
    scene_bindings: SceneBindings,
    probe_scene_bindings: SceneBindings,
    planar_scene_bindings: SceneBindings,
    reflection_probe: ReflectionProbe,
    fallback_reflection_probe: ReflectionProbe,
    planar_reflection: PlanarReflectionTarget,
    fallback_planar_reflection: PlanarReflectionTarget,
    assets: GpuAssets,
    pipelines: Vec<PipelineSlot>,

    command_pool: vk::CommandPool,
    sync: SyncState,

    debug_utils_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    needs_swapchain_rebuild: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RendererConfig {
    preferred_present_mode: Option<vk::PresentModeKHR>,
    preferred_surface_format: Option<vk::SurfaceFormatKHR>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Renderer {
    fn has_drawable_extent(&self) -> bool {
        let size = self.window_ref.inner_size();
        size.width > 0 && size.height > 0
    }

    /// Draws one frame and performs any scheduled swapchain rebuild first.
    pub fn draw(&mut self, scene: &RenderScene) {
        if !self.has_drawable_extent() {
            return;
        }

        unsafe {
            // スケジュール済みSwapchain再作成を実行
            if self.needs_swapchain_rebuild {
                self.needs_swapchain_rebuild = false;
                self.recreate_swapchain();
                return;
            }

            self.reload_pipeline_if_changed();

            let frame_index = self.sync.current_frame;
            let fence = self.sync.frames[frame_index].in_flight_fence;

            self.logical_device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fence");

            self.ensure_reflection_targets(scene);

            let reflections = self.prepare_reflections(scene);
            for face in 0..ReflectionProbe::FACE_COUNT {
                let binding_index = probe_binding_index(frame_index, face);
                self.probe_scene_bindings.update(
                    &self.logical_device,
                    binding_index,
                    &SceneUniform::reflection_probe_face(scene, &self.assets, reflections, face),
                );
            }
            self.planar_scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::planar_reflection(
                    scene,
                    self.planar_reflection.extent,
                    &self.assets,
                    reflections,
                ),
            );

            self.scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::new(scene, self.swapchain.extent, &self.assets, reflections),
            );

            let image_available = self.sync.frames[frame_index].image_available;

            let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("renderer: acquire_next_image returned OUT_OF_DATE_KHR");
                    self.schedule_rebuild();
                    return;
                }

                Err(err) => {
                    log::error!("renderer: failed to acquire swapchain image: {err:?}");
                    self.schedule_rebuild();
                    return;
                }
            };

            if self.swapchain.images_in_flight[image_index as usize] != vk::Fence::null() {
                self.logical_device
                    .wait_for_fences(
                        &[self.swapchain.images_in_flight[image_index as usize]],
                        true,
                        u64::MAX,
                    )
                    .expect("failed to wait for image fence");
            }

            self.swapchain.images_in_flight[image_index as usize] = fence;

            self.logical_device
                .reset_fences(&[fence])
                .expect("failed to reset fence");

            let command_buffer = self.swapchain.command_buffers[image_index as usize];

            self.logical_device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            self.record_draw_command_buffer(
                command_buffer,
                image_index as usize,
                frame_index,
                scene,
                reflections,
            );

            let render_finished = self.swapchain.render_finished_semaphores[image_index as usize];

            let wait_semaphores = [image_available];
            let signal_semaphores = [render_finished];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.logical_device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .expect("failed to submit draw command buffer");

            let swapchains = [self.swapchain.swapchain];
            let image_indices = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                Ok(_suboptimal) => {}

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("renderer: queue_present returned OUT_OF_DATE_KHR");
                    self.schedule_rebuild();
                }

                Err(err) => {
                    log::error!("renderer: queue_present failed: {err:?}");
                    self.schedule_rebuild();
                }
            }

            self.sync.advance_frame();
        }
    }

    /// ウィンドウリサイズ: 再作成をスケジュール
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.schedule_rebuild();
    }

    /// PresentMode設定変更: 再作成をスケジュール
    pub fn set_present_mode(&mut self, mode: vk::PresentModeKHR) {
        self.config.preferred_present_mode = Some(mode);
        self.schedule_rebuild();
    }

    /// SurfaceFormat設定変更: 再作成をスケジュール
    pub fn set_surface_format(&mut self, format: vk::SurfaceFormatKHR) {
        self.config.preferred_surface_format = Some(format);
        self.schedule_rebuild();
    }

    pub fn upload_mesh(&mut self, mesh: &CpuMesh) -> MeshId {
        self.assets.upload_mesh(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            mesh,
        )
    }

    pub fn upload_material(&mut self, material: Material) -> MaterialId {
        self.assets.upload_material(&self.logical_device, material)
    }

    pub fn upload_texture(&mut self, texture: &CpuTexture) -> TextureId {
        self.assets.upload_texture(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            texture,
        )
    }

    pub fn upload_model(&mut self, model: &CpuModel) -> ModelId {
        self.assets.upload_model(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            model,
        )
    }

    pub fn load_obj(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<ModelId> {
        let path = path.as_ref();
        let model = CpuModel::load_obj(path)?;
        log_model_stats(path, &model);

        Ok(self.upload_model(&model))
    }

    pub fn load_model(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<ModelId> {
        let path = path.as_ref();
        let model = CpuModel::load(path)?;
        log_model_stats(path, &model);

        Ok(self.upload_model(&model))
    }

    pub fn register_pipeline(&mut self, desc: PipelineDesc) -> Result<PipelineId, PipelineError> {
        let pipeline = desc.build(
            &self.logical_device,
            self.pipeline_cache,
            self.swapchain.format,
        )?;
        let id = PipelineId(self.pipelines.len());
        let hot_reload = HotReload::new(&desc.shaders, std::time::Duration::from_millis(250));

        self.pipelines.push(PipelineSlot {
            desc,
            pipeline,
            hot_reload,
        });

        Ok(id)
    }

    pub fn scene_set_layout(&self) -> vk::DescriptorSetLayout {
        self.scene_bindings.layout
    }

    fn recreate_swapchain(&mut self) {
        let size = self.window_ref.inner_size();

        if size.width == 0 || size.height == 0 {
            log::debug!("renderer: rebuild postponed (zero window size)");
            return;
        }

        let old_format = self.swapchain.format;

        self.wait_for_swapchain_idle();

        unsafe {
            self.cleanup_swapchain();
        }

        let (
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        ) = create_swapchain(
            &self.window_ref,
            &self.instance,
            &self.logical_device,
            self.physical_device,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            self.queue_family_indices,
            self.config.preferred_surface_format,
            self.config.preferred_present_mode,
        );

        if swapchain_format != old_format {
            unsafe { self.reflection_probe.destroy(&self.logical_device) };
            unsafe { self.fallback_reflection_probe.destroy(&self.logical_device) };
            unsafe { self.planar_reflection.destroy(&self.logical_device) };
            unsafe {
                self.fallback_planar_reflection
                    .destroy(&self.logical_device)
            };
            self.reflection_probe = ReflectionProbe::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain_format,
                REFLECTION_PROBE_SIZE,
            );
            self.fallback_reflection_probe = ReflectionProbe::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain_format,
                1,
            );
            self.planar_reflection = PlanarReflectionTarget::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain_format,
                self.planar_reflection.extent,
            );
            self.fallback_planar_reflection = PlanarReflectionTarget::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain_format,
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            );
            self.update_reflection_descriptors();
            self.rebuild_pipelines(swapchain_format)
                .expect("renderer: failed to rebuild pipelines for swapchain format");
        }

        self.swapchain = create_swapchain_state(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        );

        log::trace!(
            "recreated swapchain: {}x{}, format: {:?}, present_mode: {:?}",
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.swapchain.format,
            self.config.preferred_present_mode,
        );
    }

    fn schedule_rebuild(&mut self) {
        self.needs_swapchain_rebuild = true;
    }

    fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        self.transition_color_image_layout(command_buffer, image, old_layout, new_layout, 1);
    }

    fn transition_color_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        layer_count: u32,
    ) {
        let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            | (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),

            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            ),

            (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ) => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),

            (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ) => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),

            _ => panic!("unsupported layout transition"),
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count,
            });

        let dependency =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            self.logical_device
                .cmd_pipeline_barrier2(command_buffer, &dependency);
        }
    }

    fn wait_for_swapchain_idle(&self) {
        let fences: Vec<vk::Fence> = self
            .sync
            .frames
            .iter()
            .map(|frame| frame.in_flight_fence)
            .collect();

        unsafe {
            self.logical_device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("failed to wait for in-flight fences");

            self.logical_device
                .queue_wait_idle(self.present_queue)
                .expect("failed to wait present queue idle");
        }
    }

    fn reload_pipeline_if_changed(&mut self) {
        for index in 0..self.pipelines.len() {
            let changed = {
                let slot = &mut self.pipelines[index];
                slot.hot_reload.changed(&slot.desc.shaders)
            };

            match changed {
                Ok(Some(stamp)) => {
                    self.wait_for_swapchain_idle();

                    match self.rebuild_pipeline(index, self.swapchain.format) {
                        Ok(()) => {
                            self.pipelines[index].hot_reload.accept(stamp);
                            log::debug!(
                                "renderer: hot reloaded shader '{}'",
                                self.pipelines[index].desc.shaders.name
                            );
                        }
                        Err(err) => log::warn!("renderer: shader hot reload failed: {err}"),
                    }
                }
                Ok(None) => {}
                Err(err) => log::warn!("renderer: shader hot reload check failed: {err}"),
            }
        }
    }

    fn rebuild_pipelines(&mut self, format: vk::Format) -> Result<(), PipelineError> {
        for index in 0..self.pipelines.len() {
            self.rebuild_pipeline(index, format)?;
        }

        Ok(())
    }

    fn rebuild_pipeline(&mut self, index: usize, format: vk::Format) -> Result<(), PipelineError> {
        let pipeline =
            self.pipelines[index]
                .desc
                .build(&self.logical_device, self.pipeline_cache, format)?;

        unsafe {
            self.pipelines[index].pipeline.destroy(&self.logical_device);
        }

        self.pipelines[index].pipeline = pipeline;
        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self) {
        if !self.swapchain.command_buffers.is_empty() {
            unsafe {
                self.logical_device
                    .free_command_buffers(self.command_pool, &self.swapchain.command_buffers)
            };

            self.swapchain.command_buffers.clear();
        }

        for semaphore in self.swapchain.render_finished_semaphores.drain(..) {
            unsafe { self.logical_device.destroy_semaphore(semaphore, None) };
        }

        unsafe { self.swapchain.depth.destroy(&self.logical_device) };

        for image_view in self.swapchain.image_views.drain(..) {
            unsafe { self.logical_device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None)
        };

        self.swapchain.images.clear();
        self.swapchain.images_in_flight.clear();
        self.swapchain.image_layouts.clear();
    }
}
