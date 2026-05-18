use std::{path::Path, sync::Arc};

use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::window::Window;

use assets::{
    DepthTarget, GpuAssets, PlanarReflectionTarget, ReflectionProbe, SceneBindings,
    SceneRenderTarget, ShadowMap,
};
use metering::CameraMeter;

mod assets;
#[path = "passes/draw.rs"]
mod draw;
#[path = "lifecycle/drop.rs"]
mod drop;
mod frame;
mod image_layout;
#[path = "lifecycle/init.rs"]
mod init;
mod math;
mod metering;
mod pipeline;
mod pipeline_reload;
#[path = "passes/reflections.rs"]
mod reflections;
mod rendering;
mod scene;
#[path = "passes/shadows.rs"]
mod shadows;
mod swapchain_lifecycle;
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
    BoxReflectionSettings, Camera, CameraMetering, CameraResponse, DEFAULT_CAMERA_FAR,
    DirectionalLight, Material, MaterialId, MeshId, ModelId, PipelineId, PlanarReflectionSettings,
    ReflectionSettings, RenderDebugMode, RenderModel, RenderObject, RenderScene, SceneContext,
    SceneController, SceneKey, SceneMessage, TextureId, Transform, mat4_mul,
};

/// CPUがGPU完了を待たずに先行して準備できるフレーム数(1..4程度が一般的)
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const REFLECTION_PROBE_SIZE: u32 = 128;
const SHADOW_MAP_SIZE: u32 = 8192;
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
    transfer_src_supported: bool,
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

pub struct Renderer {
    _entry: Entry,
    window_ref: Arc<Window>,

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
    shadow_scene_bindings: SceneBindings,
    probe_scene_bindings: SceneBindings,
    planar_scene_bindings: SceneBindings,
    shadow_map: ShadowMap,
    reflection_probe: ReflectionProbe,
    fallback_reflection_probe: ReflectionProbe,
    planar_reflection: PlanarReflectionTarget,
    fallback_planar_reflection: PlanarReflectionTarget,
    scene_target: SceneRenderTarget,
    reflection_probe_face_cursor: usize,
    assets: GpuAssets,
    pipelines: Vec<PipelineSlot>,
    shadow_pipeline: PipelineSlot,
    post_pipeline: PipelineSlot,
    camera_meter: CameraMeter,

    command_pool: vk::CommandPool,
    sync: SyncState,

    debug_utils_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    needs_swapchain_rebuild: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBounds {
    pub center: [f32; 3],
    pub radius: f32,
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

    fn upload_model(&mut self, model: &CpuModel) -> ModelId {
        self.assets.upload_model(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            model,
        )
    }

    pub fn load_model(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<ModelId> {
        let path = path.as_ref();
        let model = CpuModel::load(path)?;
        log_model_stats(path, &model);

        Ok(self.upload_model(&model))
    }

    pub fn camera_metering(&self) -> CameraMetering {
        self.camera_meter.latest()
    }

    pub fn model_bounds(&self, model: ModelId) -> Option<ModelBounds> {
        let model = self.assets.model(model)?;
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut has_value = false;

        for primitive in &model.primitives {
            let Some(mesh) = self.assets.mesh(primitive.mesh) else {
                continue;
            };

            let radius = mesh.radius.max(0.001);
            for axis in 0..3 {
                min[axis] = min[axis].min(mesh.center[axis] - radius);
                max[axis] = max[axis].max(mesh.center[axis] + radius);
            }
            has_value = true;
        }

        has_value.then(|| {
            let center = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            let radius =
                ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2))
                    .sqrt()
                    * 0.5;

            ModelBounds { center, radius }
        })
    }
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
