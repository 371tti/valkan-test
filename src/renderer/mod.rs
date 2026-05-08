use std::{mem, path::Path, sync::Arc};

use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::window::Window;

use crate::renderer::init::{create_swapchain, create_swapchain_state};
use resource::{DepthTarget, GpuAssets, SceneBindings};

mod drop;
mod init;
mod pipeline;
mod resource;
mod scene;
pub use pipeline::{
    ColorBlendConfig, DepthConfig, GraphicsPipeline, HotReload, ModelVertex, PipelineDesc,
    PipelineError, RasterizationConfig, ShaderCode, ShaderSet, VertexAttribute, VertexLayout,
    create_pipeline_cache,
};
pub use resource::{
    CpuMesh, CpuModel, CpuPrimitive, CpuTexture, TextureFilter, TextureSampler, TextureWrap,
};
pub use scene::{
    Camera, DirectionalLight, Material, MaterialId, MeshId, ModelId, PipelineId, RenderModel,
    RenderObject, RenderScene, SceneContext, SceneController, SceneKey, SceneMessage, TextureId,
    Transform, mat4_mul,
};

/// CPUがGPU完了を待たずに先行して準備できるフレーム数(1..4程度が一般的)
const MAX_FRAMES_IN_FLIGHT: usize = 2;

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

#[repr(C)]
#[derive(Clone, Copy)]
struct SceneUniform {
    view_proj: [f32; 16],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
    camera_pos: [f32; 4],
}

impl SceneUniform {
    fn new(scene: &RenderScene, extent: vk::Extent2D) -> Self {
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let light = scene.light;

        Self {
            view_proj: scene.camera.view_projection(aspect),
            light_dir: [
                light.direction[0],
                light.direction[1],
                light.direction[2],
                0.0,
            ],
            light_color: [
                light.color[0] * light.intensity,
                light.color[1] * light.intensity,
                light.color[2] * light.intensity,
                0.0,
            ],
            ambient: [light.ambient[0], light.ambient[1], light.ambient[2], 0.0],
            camera_pos: [
                scene.camera.eye[0],
                scene.camera.eye[1],
                scene.camera.eye[2],
                1.0,
            ],
        }
    }
}

impl Default for SceneUniform {
    fn default() -> Self {
        Self::new(
            &RenderScene::default(),
            vk::Extent2D {
                width: 1,
                height: 1,
            },
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ObjectPush {
    model: [f32; 16],
    base_color: [f32; 4],
    emissive_color: [f32; 4],
    material: [f32; 4],
    texture_flags: [f32; 4],
    texture_info: [f32; 4],
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) }
}

fn has_texture(texture: Option<TextureId>) -> f32 {
    texture.is_some() as u8 as f32
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

fn draw_object(
    device: &ash::Device,
    pipelines: &[PipelineSlot],
    assets: &GpuAssets,
    scene_bindings: &SceneBindings,
    command_buffer: vk::CommandBuffer,
    frame_index: usize,
    object: RenderObject,
    bound_pipeline: &mut Option<PipelineId>,
    bound_material: &mut Option<MaterialId>,
) {
    let Some(slot) = pipelines.get(object.pipeline.0) else {
        return;
    };
    let Some(mesh) = assets.mesh(object.mesh) else {
        return;
    };

    unsafe {
        if *bound_pipeline != Some(object.pipeline) {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.handle,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.layout,
                0,
                std::slice::from_ref(&scene_bindings.sets[frame_index]),
                &[],
            );
            *bound_pipeline = Some(object.pipeline);
            *bound_material = None;
        }

        let material = assets.material(object.material);
        if *bound_material != Some(object.material) {
            if let Some(texture_set) = assets.material_texture_set(object.material) {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    slot.pipeline.layout,
                    1,
                    std::slice::from_ref(&texture_set),
                    &[],
                );
                *bound_material = Some(object.material);
            }
        }

        let push = ObjectPush {
            model: object.transform.matrix(),
            base_color: material.base_color,
            emissive_color: [
                material.emissive_color[0],
                material.emissive_color[1],
                material.emissive_color[2],
                0.0,
            ],
            material: [
                material.metallic,
                material.roughness,
                material.specular,
                material.ambient_occlusion,
            ],
            texture_flags: [
                has_texture(material.base_color_texture),
                has_texture(material.metallic_roughness_texture),
                has_texture(material.normal_texture),
                has_texture(material.occlusion_texture),
            ],
            texture_info: [
                has_texture(material.emissive_texture),
                material.normal_scale,
                material.occlusion_strength,
                material.alpha_cutoff,
            ],
        };
        device.cmd_push_constants(
            command_buffer,
            slot.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes_of(&push),
        );

        device.cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertex.buffer], &[0]);
        device.cmd_bind_index_buffer(command_buffer, mesh.index.buffer, 0, vk::IndexType::UINT32);
        device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
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

    /// 描画 高速パス: スケジュール済み再作成をここで実行してからreturn
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

            self.scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::new(scene, self.swapchain.extent),
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

    fn record_draw_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_index: usize,
        scene: &RenderScene,
    ) {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            let image = self.swapchain.images[image_index];
            let old_layout = self.swapchain.image_layouts[image_index];

            self.transition_image_layout(
                command_buffer,
                image,
                old_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.image_views[image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                });

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.extent.width as f32,
                height: self.swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            };

            self.logical_device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.logical_device
                .cmd_set_scissor(command_buffer, 0, &[scissor]);

            let mut bound_pipeline = None;
            let mut bound_material = None;

            for object in &scene.objects {
                draw_object(
                    &self.logical_device,
                    &self.pipelines,
                    &self.assets,
                    &self.scene_bindings,
                    command_buffer,
                    frame_index,
                    *object,
                    &mut bound_pipeline,
                    &mut bound_material,
                );
            }

            for model in &scene.models {
                let Some(gpu_model) = self.assets.model(model.model) else {
                    continue;
                };

                for primitive in &gpu_model.primitives {
                    draw_object(
                        &self.logical_device,
                        &self.pipelines,
                        &self.assets,
                        &self.scene_bindings,
                        command_buffer,
                        frame_index,
                        RenderObject {
                            mesh: primitive.mesh,
                            pipeline: model.pipeline,
                            transform: model.transform,
                            material: primitive.material,
                        },
                        &mut bound_pipeline,
                        &mut bound_material,
                    );
                }
            }
            self.logical_device.cmd_end_rendering(command_buffer);

            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.swapchain.image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;

            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("failed to end command buffer");
        }
    }

    fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
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
                layer_count: 1,
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
