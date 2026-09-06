use std::{
    collections::BTreeMap,
    ffi::CStr,
    mem::{offset_of, size_of},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use ash::{Device, Instance, vk};

use crate::{
    import::{ImportedMaterial, ImportedMesh},
    math::{dot3, identity_mat4, normalize_or, sub3},
    protocol::{
        AssetHandle, CameraSnapshot, MaterialAlphaMode, MeshHandle, RenderItemPacket,
        RenderOptimizationSettings, SceneBounds, SceneHandle,
    },
    renderer::{
        DEFAULT_AMBIENT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_DIR,
        DEFAULT_SHADOW_CASCADE_METRICS, DEFAULT_SHADOW_CASCADE_SPLITS,
        assets::{MeshGeometry, MeshVertex},
        graph::SHADOW_CASCADE_COUNT,
        pipeline::shader_interface,
        shadow_map_size,
        visibility::{MeshLodLevel, MeshVisibility, classify_mesh},
    },
};

use super::{
    VulkanError,
    buffer::{
        DeviceLocalBufferUploadBatch, GpuBuffer, create_buffer_with_data, destroy_buffers,
        memory_properties, write_buffer_value,
    },
    lod::unique_lod_indices,
    shader::{self, assets},
};

const SHADER_ENTRY: &CStr = shader::ENTRY;
const UNTEXTURED_VERTEX_SHADER: &[u8] = assets::MESH_UNTEXTURED_VERT;
const VERTEX_SHADER: &[u8] = assets::MESH_VERT;
const SCENE_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_FRAG;
const SCENE_OPAQUE_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_OPAQUE_FRAG;
const SCENE_FAST_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_FAST_FRAG;
const SCENE_OPAQUE_FAST_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_OPAQUE_FAST_FRAG;
const SCENE_TEXTURED_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_TEXTURED_FRAG;
const SCENE_OPAQUE_TEXTURED_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_OPAQUE_TEXTURED_FRAG;
const SCENE_TEXTURED_FAST_FRAGMENT_SHADER: &[u8] = assets::MESH_SCENE_TEXTURED_FAST_FRAG;
const SCENE_OPAQUE_TEXTURED_FAST_FRAGMENT_SHADER: &[u8] =
    assets::MESH_SCENE_OPAQUE_TEXTURED_FAST_FRAG;
const DIRECTIONAL_SHADOW_VERTEX_SHADER: &[u8] = assets::SHADOW_DIRECTIONAL_VERT;
const LOCAL_SHADOW_VERTEX_SHADER: &[u8] = assets::SHADOW_LOCAL_VERT;
const DIRECTIONAL_SHADOW_OPAQUE_VERTEX_SHADER: &[u8] = assets::SHADOW_OPAQUE_DIRECTIONAL_VERT;
const LOCAL_SHADOW_OPAQUE_VERTEX_SHADER: &[u8] = assets::SHADOW_OPAQUE_LOCAL_VERT;
const SHADOW_DEPTH_FRAGMENT_SHADER: &[u8] = assets::SHADOW_DEPTH_FRAG;
const SHADOW_DEPTH_OPAQUE_FRAGMENT_SHADER: &[u8] = assets::SHADOW_DEPTH_OPAQUE_FRAG;
const SHADOW_DEPTH_TEXTURED_FRAGMENT_SHADER: &[u8] = assets::SHADOW_DEPTH_TEXTURED_FRAG;
const SHADOW_TRANSLUCENT_FRAGMENT_SHADER: &[u8] = assets::SHADOW_TRANSLUCENT_FRAG;
const SHADOW_TRANSLUCENT_TEXTURED_FRAGMENT_SHADER: &[u8] = assets::SHADOW_TRANSLUCENT_TEXTURED_FRAG;
const MESH_FRONT_FACE: vk::FrontFace = vk::FrontFace::COUNTER_CLOCKWISE;
// Receiver-plane and normal-offset bias handle the remaining precision error while sampling.
// Keep the caster bias small but non-zero: the slope term still separates nearly coplanar
// rasterized depth, without stacking a large fixed offset onto both receiver-side corrections.
const SHADOW_DEPTH_BIAS_CONSTANT: f32 = 0.20;
const SHADOW_DEPTH_BIAS_SLOPE: f32 = 0.40;
const MIN_ADAPTIVE_SHADOW_TRIANGLES: f32 = 128.0;
const ADAPTIVE_SHADOW_TRIANGLES_PER_TEXEL: f32 = 0.25;
const MAX_MESH_PREPARE_WORKERS: usize = 8;
const PARALLEL_MESH_PREPARE_MIN_INDICES: usize = 50_000;
pub(super) const MAX_LOCAL_LIGHTS: usize = 4;
pub(super) const LOCAL_SHADOW_FACE_COUNT: usize = 6;
pub(super) const LOCAL_SHADOW_MATRIX_COUNT: usize = MAX_LOCAL_LIGHTS * LOCAL_SHADOW_FACE_COUNT;

/// Position-only stream consumed by fully opaque shadow passes.
///
/// Keeping this separate from material surface data lets depth-only shaders fetch 12 bytes instead
/// of stepping over the full 32-byte scene vertex for every shadow-map vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct GpuMeshPosition {
    position: [f32; 3],
}

impl GpuMeshPosition {
    fn from_mesh(vertex: MeshVertex) -> Self {
        Self {
            position: vertex.position,
        }
    }
}

/// Compact surface stream used only by passes that need material or shading attributes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct GpuMeshSurface {
    normal: u32,
    uv: [f32; 2],
    tangent: u32,
    color: u32,
}

impl GpuMeshSurface {
    fn from_mesh(vertex: MeshVertex) -> Self {
        Self {
            normal: pack_snorm_10_10_10_2([
                vertex.normal[0],
                vertex.normal[1],
                vertex.normal[2],
                1.0,
            ]),
            uv: vertex.uv,
            tangent: pack_snorm_10_10_10_2(vertex.tangent),
            color: pack_unorm_8_8_8_8(vertex.color),
        }
    }
}

pub(super) struct VulkanMeshStore {
    meshes: BTreeMap<MeshHandle, VulkanMesh>,
    frame_uniforms: Vec<GpuBuffer>,
    frame_descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    frame_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
}

#[derive(Clone, Copy)]
pub(super) struct MeshPipeline {
    handle: vk::Pipeline,
    uses_surface_stream: bool,
}

#[derive(Clone, Copy)]
pub(super) struct MeshPipelineKey {
    pub(super) uses_textures: bool,
    pub(super) double_sided: bool,
    pub(super) opaque_scene: bool,
    pub(super) opaque_shadow: bool,
}

#[derive(Clone, Copy, Default)]
pub(super) struct MeshDrawState {
    pipeline: vk::Pipeline,
    frame_descriptor_set: vk::DescriptorSet,
    material_descriptor_set: vk::DescriptorSet,
    pass_descriptor_set: vk::DescriptorSet,
    position_buffer: vk::Buffer,
    surface_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    index_type: Option<vk::IndexType>,
    extent: vk::Extent2D,
    shadow_cascade_index: Option<u32>,
}

#[derive(Clone, Copy)]
pub(super) struct MeshPipelineSet {
    untextured: MeshPipelineVariants,
    textured: MeshPipelineVariants,
    opaque_scene: Option<OpaqueScenePipelineVariants>,
    opaque_shadow: Option<MeshPipelineVariants>,
}

#[derive(Clone, Copy)]
struct MeshPipelineVariants {
    culled: MeshPipeline,
    double_sided: MeshPipeline,
}

#[derive(Clone, Copy)]
struct OpaqueScenePipelineVariants {
    untextured: MeshPipelineVariants,
    textured: MeshPipelineVariants,
}

pub(super) struct MeshPassResources {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    directional_shadow_view: vk::ImageView,
    transmittance_sampler: vk::Sampler,
    local_shadow_sampler: vk::Sampler,
    depth_shadow_sampler: vk::Sampler,
    depth_shadow_raw_sampler: vk::Sampler,
}

#[derive(Clone, Copy)]
pub(super) struct MeshDrawOptions {
    extent: vk::Extent2D,
    camera: Option<CameraSnapshot>,
    optimization: RenderOptimizationSettings,
    forced_lod: Option<MeshLodLevel>,
    shadow_cascade_index: Option<u32>,
    shadow_cull: Option<ShadowCascadeCull>,
}

/// Mesh facts resolved once while the frame draw lists are prepared.
///
/// `options` is `None` only when an uploaded mesh was camera-culled. Missing mesh handles keep
/// an option so command recording can retain its usual missing-handle diagnostic path.
#[derive(Clone, Copy)]
pub(super) struct SceneMeshDrawInfo {
    pub(super) bounds: Option<SceneBounds>,
    pub(super) options: Option<MeshDrawOptions>,
}

#[derive(Clone, Copy)]
pub(super) struct ShadowCascadeCull {
    camera: CameraSnapshot,
    min_depth: f32,
    max_depth: f32,
    view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
    projection_radius: [[f32; 3]; SHADOW_CASCADE_COUNT],
    view_proj_count: usize,
}

impl ShadowCascadeCull {
    /// Creates the camera-depth window covered by one shadow cascade.
    ///
    /// The mesh store uses this only to drop casters that are far outside the cascade's receiver
    /// range. Lateral camera frustum culling is intentionally avoided because off-screen casters
    /// can still project shadows into the visible receiver range.
    pub(super) fn new(camera: CameraSnapshot, min_depth: f32, max_depth: f32) -> Self {
        Self {
            camera,
            min_depth,
            max_depth,
            view_proj: [identity_mat4(); SHADOW_CASCADE_COUNT],
            projection_radius: [[1.0; 3]; SHADOW_CASCADE_COUNT],
            view_proj_count: 0,
        }
    }

    /// Adds the exact light-space projection used by the cascade being refreshed.
    pub(super) fn with_light_space_projection(mut self, view_proj: [f32; 16]) -> Self {
        self.view_proj[0] = view_proj;
        self.projection_radius[0] = shadow_projection_radius(view_proj);
        self.view_proj_count = 1;
        self
    }

    /// Adds the active stable-CSM light-space projections used by the cascade being refreshed.
    ///
    /// The fixed-capacity array avoids per-cascade allocation. `view_proj_count` is authoritative:
    /// inactive identity slots must never make a caster appear relevant to an active direction.
    #[cfg(test)]
    pub(super) fn with_light_space_projections(
        mut self,
        view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
        view_proj_count: usize,
    ) -> Self {
        self.view_proj = view_proj;
        self.projection_radius =
            std::array::from_fn(|index| shadow_projection_radius(self.view_proj[index]));
        self.view_proj_count = view_proj_count.min(SHADOW_CASCADE_COUNT);
        self
    }

    #[cfg(test)]
    pub(super) fn light_space_projection_count(self) -> usize {
        self.view_proj_count
    }

    /// Tests already-resolved mesh bounds against this cascade without another mesh-store lookup.
    ///
    /// Missing bounds deliberately stay accepted: incomplete metadata must not make a
    /// directional-shadow caster disappear.
    pub(super) fn contains_bounds(&self, bounds: Option<SceneBounds>) -> bool {
        shadow_cascade_contains_bounds(bounds, self)
    }
}

impl MeshDrawOptions {
    /// Creates scene-pass options after the caller has already selected the visible LOD.
    fn scene_preclassified(
        extent: vk::Extent2D,
        camera: CameraSnapshot,
        optimization: RenderOptimizationSettings,
        forced_lod: Option<MeshLodLevel>,
    ) -> Self {
        Self {
            extent,
            camera: Some(camera),
            optimization,
            forced_lod,
            shadow_cascade_index: None,
            shadow_cull: None,
        }
    }

    /// Creates shadow options after the caller has already applied cascade culling.
    pub(super) fn shadow_preculled(
        extent: vk::Extent2D,
        cascade_index: usize,
        lod: MeshLodLevel,
    ) -> Self {
        Self {
            extent,
            camera: None,
            optimization: RenderOptimizationSettings::disabled(),
            forced_lod: Some(lod),
            shadow_cascade_index: Some(cascade_index as u32),
            shadow_cull: None,
        }
    }
}

impl VulkanMeshStore {
    /// Creates mesh draw resources that are independent from the current swapchain.
    pub(super) fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frame_count: usize,
        material_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, VulkanError> {
        shader_interface::validate_mesh_interface()
            .map_err(|message| VulkanError::ShaderInterface(message.into()))?;
        super::swapchain::validate_shadow_format_support(instance, physical_device)?;
        let memory_properties = memory_properties(instance, physical_device);
        let frame_set_layout = create_frame_set_layout(device)?;
        let pass_set_layout = match create_pass_set_layout(device) {
            Ok(layout) => layout,
            Err(error) => {
                destroy_descriptor_set_layout(device, frame_set_layout);
                return Err(error);
            }
        };
        let pipeline_layout = match create_pipeline_layout(
            device,
            frame_set_layout,
            material_set_layout,
            pass_set_layout,
        ) {
            Ok(layout) => layout,
            Err(error) => {
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, frame_set_layout);
                return Err(error);
            }
        };
        let frame_uniforms = match create_frame_uniforms(device, &memory_properties, frame_count) {
            Ok(uniforms) => uniforms,
            Err(error) => {
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, frame_set_layout);
                return Err(error);
            }
        };
        let descriptor_pool = match create_descriptor_pool(device, frame_count) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_buffers(device, frame_uniforms);
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, frame_set_layout);
                return Err(error);
            }
        };
        let frame_descriptor_sets = match allocate_frame_descriptor_sets(
            device,
            descriptor_pool,
            frame_set_layout,
            frame_count,
        ) {
            Ok(sets) => sets,
            Err(error) => {
                destroy_descriptor_pool(device, descriptor_pool);
                destroy_buffers(device, frame_uniforms);
                destroy_pipeline_layout(device, pipeline_layout);
                destroy_descriptor_set_layout(device, pass_set_layout);
                destroy_descriptor_set_layout(device, frame_set_layout);
                return Err(error);
            }
        };
        update_frame_descriptor_sets(device, &frame_descriptor_sets, &frame_uniforms);
        tracing::info!("created Vulkan mesh resources");

        Ok(Self {
            meshes: BTreeMap::new(),
            frame_uniforms,
            frame_descriptor_sets,
            descriptor_pool,
            frame_set_layout,
            pass_set_layout,
            pipeline_layout,
        })
    }

    /// Uploads imported meshes into backend-local Vulkan vertex and index buffers.
    pub(super) fn upload_imported_meshes(
        &mut self,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        queue: vk::Queue,
        scene: SceneHandle,
        handles: &[MeshHandle],
        meshes: &[ImportedMesh],
        materials: &[ImportedMaterial],
    ) -> Result<(), VulkanError> {
        let memory_properties = memory_properties(instance, physical_device);
        let mut total_vertices = 0_usize;
        let mut total_vertex_bytes = 0_usize;
        let mut total_position_bytes = 0_usize;
        let mut total_surface_bytes = 0_usize;
        let mut total_source_indices = 0_usize;
        let mut total_lod_buffers = 0_usize;
        let mut total_lod_indices = 0_usize;
        let mut total_lod_index_bytes = 0_usize;
        let mut overdraw_optimized_meshes = 0_usize;
        let mut upload_batch = DeviceLocalBufferUploadBatch::new(device, &memory_properties);
        let mut pending_meshes = Vec::with_capacity(handles.len().min(meshes.len()));
        let job_count = handles.len().min(meshes.len());
        let total_input_indices = meshes[..job_count]
            .iter()
            .map(imported_mesh_index_count)
            .sum::<usize>();
        let prepare_workers = mesh_prepare_worker_count(job_count, total_input_indices);
        let mut job_order = (0..job_count).collect::<Vec<_>>();
        job_order.sort_unstable_by(|&left, &right| {
            imported_mesh_index_count(&meshes[right]).cmp(&imported_mesh_index_count(&meshes[left]))
        });
        let next_job = AtomicUsize::new(0);
        let (prepared_sender, prepared_receiver) = mpsc::channel();
        let mut prepare_cpu_ms = 0.0_f64;
        let prepare_started = Instant::now();

        thread::scope(|scope| -> Result<(), VulkanError> {
            for _ in 0..prepare_workers {
                let prepared_sender = prepared_sender.clone();
                let job_order = &job_order;
                let next_job = &next_job;
                scope.spawn(move || {
                    loop {
                        let order_index = next_job.fetch_add(1, Ordering::Relaxed);
                        let Some(&mesh_index) = job_order.get(order_index) else {
                            break;
                        };
                        let reduce_overdraw = material_reduces_overdraw(materials.get(mesh_index));
                        let cpu_started = Instant::now();
                        let prepared =
                            PreparedVulkanMesh::from_imported(&meshes[mesh_index], reduce_overdraw);
                        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
                        if prepared_sender
                            .send((mesh_index, reduce_overdraw, cpu_ms, prepared))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
            }
            drop(prepared_sender);

            for _ in 0..job_count {
                let (mesh_index, reduce_overdraw, cpu_ms, prepared) = prepared_receiver
                    .recv()
                    .expect("mesh preparation workers return every queued mesh");
                prepare_cpu_ms += cpu_ms;
                overdraw_optimized_meshes += usize::from(reduce_overdraw);
                let vertex_count = prepared.vertex_count();
                let position_bytes = vertex_count * size_of::<GpuMeshPosition>();
                let surface_bytes = vertex_count * size_of::<GpuMeshSurface>();
                total_vertices += vertex_count;
                total_position_bytes += position_bytes;
                total_surface_bytes += surface_bytes;
                total_vertex_bytes += position_bytes + surface_bytes;
                total_source_indices += prepared.source_index_count();
                let pending = prepared.queue(&mut upload_batch)?;
                total_lod_buffers += pending.lods.len();
                total_lod_indices += pending.lod_index_count_sum();
                total_lod_index_bytes += pending.lod_index_byte_sum();
                pending_meshes.push((handles[mesh_index], pending));
            }
            Ok(())
        })?;
        let prepare_elapsed = prepare_started.elapsed();

        let submit_started = Instant::now();
        let mut uploaded_buffers = upload_batch.finish(queue_family_index, queue)?.into_iter();
        let submit_elapsed = submit_started.elapsed();
        for (handle, pending) in pending_meshes {
            let uploaded = pending.finish(&mut uploaded_buffers, scene);
            if let Some(previous) = self.meshes.insert(handle, uploaded) {
                previous.destroy(device);
            }
        }
        debug_assert!(uploaded_buffers.next().is_none());

        tracing::trace!(
            meshes = job_count,
            total_vertices,
            total_vertex_bytes,
            total_position_bytes,
            total_surface_bytes,
            total_source_indices,
            total_lod_buffers,
            total_lod_indices,
            total_lod_index_bytes,
            overdraw_optimized_meshes,
            prepare_workers,
            prepare_cpu_ms,
            prepare_ms = prepare_elapsed.as_secs_f64() * 1000.0,
            submit_ms = submit_elapsed.as_secs_f64() * 1000.0,
            "uploaded Vulkan mesh batch"
        );
        Ok(())
    }

    /// Creates opaque scene-pass mesh pipelines that write color, depth, and material metadata.
    pub(super) fn create_scene_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            VERTEX_SHADER,
            SCENE_FRAGMENT_SHADER,
            SCENE_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneOpaque,
        )?;
        let pipelines = self.add_opaque_scene_variants(
            device,
            render_pass,
            pipelines,
            SCENE_OPAQUE_FRAGMENT_SHADER,
            SCENE_OPAQUE_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneOpaque,
        )?;
        tracing::info!("created Vulkan opaque scene mesh pipelines");
        Ok(pipelines)
    }

    /// Creates opaque scene-pass mesh pipelines for frames that do not need material metadata.
    pub(super) fn create_scene_fast_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            VERTEX_SHADER,
            SCENE_FAST_FRAGMENT_SHADER,
            SCENE_TEXTURED_FAST_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneOpaqueFast,
        )?;
        let pipelines = self.add_opaque_scene_variants(
            device,
            render_pass,
            pipelines,
            SCENE_OPAQUE_FAST_FRAGMENT_SHADER,
            SCENE_OPAQUE_TEXTURED_FAST_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneOpaqueFast,
        )?;
        tracing::info!("created Vulkan fast opaque scene mesh pipelines");
        Ok(pipelines)
    }

    /// Creates transparent scene-pass mesh pipelines that blend color without writing depth.
    pub(super) fn create_scene_transparent_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            VERTEX_SHADER,
            SCENE_FRAGMENT_SHADER,
            SCENE_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneTransparent,
        )?;
        tracing::info!("created Vulkan transparent scene mesh pipelines");
        Ok(pipelines)
    }

    /// Creates transparent scene-pass mesh pipelines for frames that do not need material metadata.
    pub(super) fn create_scene_transparent_fast_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            VERTEX_SHADER,
            SCENE_FAST_FRAGMENT_SHADER,
            SCENE_TEXTURED_FAST_FRAGMENT_SHADER,
            MeshPipelineTarget::SceneTransparentFast,
        )?;
        tracing::info!("created Vulkan fast transparent scene mesh pipelines");
        Ok(pipelines)
    }

    /// Creates depth-shadow mesh pipelines compatible with the shadow graph pass.
    pub(super) fn create_shadow_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            DIRECTIONAL_SHADOW_VERTEX_SHADER,
            SHADOW_DEPTH_FRAGMENT_SHADER,
            SHADOW_DEPTH_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::OpaqueShadow,
        )?;
        let pipelines = self.add_opaque_shadow_variant(
            device,
            render_pass,
            pipelines,
            DIRECTIONAL_SHADOW_OPAQUE_VERTEX_SHADER,
            SHADOW_DEPTH_OPAQUE_FRAGMENT_SHADER,
            MeshPipelineTarget::OpaqueShadow,
        )?;
        tracing::info!("created Vulkan shadow mesh pipelines");
        Ok(pipelines)
    }

    /// Creates depth-only cubemap pipelines for point/local light shadow faces.
    pub(super) fn create_local_shadow_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            LOCAL_SHADOW_VERTEX_SHADER,
            SHADOW_DEPTH_FRAGMENT_SHADER,
            SHADOW_DEPTH_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::LocalShadowDepth,
        )?;
        let pipelines = self.add_opaque_shadow_variant(
            device,
            render_pass,
            pipelines,
            LOCAL_SHADOW_OPAQUE_VERTEX_SHADER,
            SHADOW_DEPTH_OPAQUE_FRAGMENT_SHADER,
            MeshPipelineTarget::LocalShadowDepth,
        )?;
        tracing::info!("created Vulkan local shadow cubemap mesh pipelines");
        Ok(pipelines)
    }

    /// Creates multiplicative-transmittance pipelines for transparent shadow casters.
    pub(super) fn create_translucent_shadow_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            DIRECTIONAL_SHADOW_VERTEX_SHADER,
            SHADOW_TRANSLUCENT_FRAGMENT_SHADER,
            SHADOW_TRANSLUCENT_TEXTURED_FRAGMENT_SHADER,
            MeshPipelineTarget::TranslucentShadow,
        )?;
        tracing::info!("created Vulkan translucent shadow mesh pipelines");
        Ok(pipelines)
    }

    /// Creates one untextured/textured pair that shares the same vertex and render-pass layout.
    fn create_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
        vertex_shader: &[u8],
        untextured_fragment: &[u8],
        textured_fragment: &[u8],
        target: MeshPipelineTarget,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let untextured_vertex_shader = if target.uses_surface_normal() {
            UNTEXTURED_VERTEX_SHADER
        } else {
            vertex_shader
        };
        let untextured_vertex_layout = if target.uses_surface_normal() {
            MeshVertexLayout::SceneUntextured
        } else {
            MeshVertexLayout::Shadow
        };
        let textured_vertex_layout = if target.uses_surface_normal() {
            MeshVertexLayout::SceneTextured
        } else {
            MeshVertexLayout::Shadow
        };
        let untextured = create_mesh_pipeline_variants(
            device,
            self.pipeline_layout,
            render_pass,
            untextured_vertex_shader,
            untextured_fragment,
            untextured_vertex_layout,
            target,
        )?;
        let textured = match create_mesh_pipeline_variants(
            device,
            self.pipeline_layout,
            render_pass,
            vertex_shader,
            textured_fragment,
            textured_vertex_layout,
            target,
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                destroy_pipeline_variants(device, untextured);
                return Err(error);
            }
        };

        Ok(MeshPipelineSet {
            untextured,
            textured,
            opaque_scene: None,
            opaque_shadow: None,
        })
    }

    /// Adds alpha-test-free fragment variants for the fully opaque scene-material class.
    fn add_opaque_scene_variants(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
        mut pipelines: MeshPipelineSet,
        untextured_fragment_shader: &[u8],
        textured_fragment_shader: &[u8],
        target: MeshPipelineTarget,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let untextured = match create_mesh_pipeline_variants(
            device,
            self.pipeline_layout,
            render_pass,
            UNTEXTURED_VERTEX_SHADER,
            untextured_fragment_shader,
            MeshVertexLayout::SceneUntextured,
            target,
        ) {
            Ok(variants) => variants,
            Err(error) => {
                self.destroy_pipeline_set(device, pipelines);
                return Err(error);
            }
        };
        let textured = match create_mesh_pipeline_variants(
            device,
            self.pipeline_layout,
            render_pass,
            VERTEX_SHADER,
            textured_fragment_shader,
            MeshVertexLayout::SceneTextured,
            target,
        ) {
            Ok(variants) => variants,
            Err(error) => {
                destroy_pipeline_variants(device, untextured);
                self.destroy_pipeline_set(device, pipelines);
                return Err(error);
            }
        };
        pipelines.opaque_scene = Some(OpaqueScenePipelineVariants {
            untextured,
            textured,
        });
        Ok(pipelines)
    }

    /// Adds the position-only, alpha-test-free path used by fully opaque shadow casters.
    fn add_opaque_shadow_variant(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
        mut pipelines: MeshPipelineSet,
        vertex_shader: &[u8],
        fragment_shader: &[u8],
        target: MeshPipelineTarget,
    ) -> Result<MeshPipelineSet, VulkanError> {
        match create_mesh_pipeline_variants(
            device,
            self.pipeline_layout,
            render_pass,
            vertex_shader,
            fragment_shader,
            MeshVertexLayout::ShadowOpaque,
            target,
        ) {
            Ok(opaque_shadow) => {
                pipelines.opaque_shadow = Some(opaque_shadow);
                Ok(pipelines)
            }
            Err(error) => {
                self.destroy_pipeline_set(device, pipelines);
                Err(error)
            }
        }
    }

    /// Creates descriptors that let scene shaders sample the graph-owned shadow map.
    pub(super) fn create_pass_resources(
        &self,
        device: &Device,
        depth_shadow_view: vk::ImageView,
        translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
    ) -> Result<MeshPassResources, VulkanError> {
        MeshPassResources::create(
            device,
            self.pass_set_layout,
            depth_shadow_view,
            translucent_shadow_views,
            local_shadow_views,
        )
    }

    /// Returns the frame-uniform descriptor-set layout shared with the scene shaders.
    pub(super) fn frame_set_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_set_layout
    }

    /// Returns the frame-uniform descriptor set for one in-flight slot.
    pub(super) fn frame_descriptor_set(
        &self,
        frame_slot: usize,
    ) -> Result<vk::DescriptorSet, VulkanError> {
        self.frame_descriptor_sets.get(frame_slot).copied().ok_or(
            VulkanError::FrameSlotIndexOutOfRange {
                index: frame_slot,
                count: self.frame_descriptor_sets.len(),
            },
        )
    }

    /// Updates the optional PCSS visibility history sampled by full scene fragment shaders.
    pub(super) fn update_pcss_history_descriptor(
        &self,
        device: &Device,
        frame_slot: usize,
        sampler: vk::Sampler,
        view: vk::ImageView,
    ) -> Result<(), VulkanError> {
        let descriptor_set = self.frame_descriptor_sets.get(frame_slot).copied().ok_or(
            VulkanError::FrameSlotIndexOutOfRange {
                index: frame_slot,
                count: self.frame_descriptor_sets.len(),
            },
        )?;
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    /// Writes camera, shadow, and lighting parameters used by one frame slot.
    pub(super) fn write_frame_uniform(
        &self,
        device: &Device,
        frame_slot: usize,
        value: MeshFrameUniform,
    ) -> Result<(), VulkanError> {
        let buffer =
            self.frame_uniforms
                .get(frame_slot)
                .ok_or(VulkanError::FrameSlotIndexOutOfRange {
                    index: frame_slot,
                    count: self.frame_uniforms.len(),
                })?;

        write_buffer_value(device, buffer, &value)
    }

    /// Resolves bounds plus scene draw options with one mesh-store lookup.
    pub(super) fn scene_draw_info(
        &self,
        mesh: MeshHandle,
        extent: vk::Extent2D,
        camera: CameraSnapshot,
        optimization: RenderOptimizationSettings,
    ) -> SceneMeshDrawInfo {
        let Some(mesh) = self.meshes.get(&mesh) else {
            return SceneMeshDrawInfo {
                bounds: None,
                options: Some(MeshDrawOptions::scene_preclassified(
                    extent,
                    camera,
                    optimization,
                    None,
                )),
            };
        };

        let bounds = mesh.bounds;
        let aspect = if extent.height > 0 {
            extent.width as f32 / extent.height as f32
        } else {
            1.0
        };
        let forced_lod = match classify_mesh(
            camera,
            aspect,
            extent.height,
            bounds,
            mesh.full_index_count() as usize / 3,
            optimization,
        ) {
            MeshVisibility::Visible { lod } => Some(lod),
            MeshVisibility::Culled { .. } => {
                return SceneMeshDrawInfo {
                    bounds,
                    options: None,
                };
            }
        };

        SceneMeshDrawInfo {
            bounds,
            options: Some(MeshDrawOptions::scene_preclassified(
                extent,
                camera,
                optimization,
                forced_lod,
            )),
        }
    }

    /// Selects one directional-shadow LOD once while its cascade draw list is built.
    pub(super) fn directional_shadow_lod(
        &self,
        mesh: MeshHandle,
        cascade_index: usize,
        texel_world: f32,
        shadow_resolution: u32,
    ) -> MeshLodLevel {
        let requested = shadow_lod_for_cascade(cascade_index);
        let Some(mesh) = self.meshes.get(&mesh) else {
            return requested;
        };
        // All Stable CSM layers have one shared resolution.  Keep the caster LOD decision tied to
        // that actual texel budget instead of applying a near/far resolution curve.
        let extent = shadow_resolution.max(1);
        let levels = [
            MeshLodLevel::Full,
            MeshLodLevel::Medium,
            MeshLodLevel::Low,
            MeshLodLevel::VeryLow,
        ];
        let lod_triangles = std::array::from_fn(|index| {
            let level = levels[index];
            mesh.lod(level)
                .map_or(0, |lod| lod.index_count as usize / 3)
        });
        let requested = adaptive_directional_shadow_lod(
            mesh.bounds,
            texel_world,
            vk::Extent2D {
                width: extent,
                height: extent,
            },
            lod_triangles,
            requested,
        );

        mesh.lod(requested).map_or(requested, |lod| lod.level)
    }

    /// Returns the index count that one already-selected shadow draw will actually submit.
    pub(super) fn shadow_index_count(&self, mesh: MeshHandle, lod: MeshLodLevel) -> usize {
        self.meshes
            .get(&mesh)
            .and_then(|mesh| mesh.lod(lod))
            .map_or(0, |lod| lod.index_count as usize)
    }

    /// Binds the mesh pipeline and records one indexed mesh draw if the handle is live.
    ///
    /// The returned boolean is true only when a Vulkan draw command was recorded, so pass-level
    /// diagnostics can distinguish accepted packets from backend culling.
    pub(super) fn bind_and_draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        pipeline_set: MeshPipelineSet,
        pass_resources: Option<&MeshPassResources>,
        frame_slot: usize,
        item: &RenderItemPacket,
        material_descriptor_set: vk::DescriptorSet,
        options: MeshDrawOptions,
        pipeline_key: MeshPipelineKey,
        state: &mut MeshDrawState,
    ) -> Result<bool, VulkanError> {
        if !item.flags.visible {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                "mesh draw skipped because the item is not visible"
            );
            return Ok(false);
        }

        let Some(mesh) = self.meshes.get(&item.mesh) else {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                "mesh draw skipped because the Vulkan mesh is missing"
            );
            return Ok(false);
        };
        let Some(lod) = mesh.visible_lod(item, options) else {
            return Ok(false);
        };
        let frame_descriptor_set = self.frame_descriptor_sets.get(frame_slot).copied().ok_or(
            VulkanError::FrameSlotIndexOutOfRange {
                index: frame_slot,
                count: self.frame_descriptor_sets.len(),
            },
        )?;
        let pipeline = pipeline_set.choose(pipeline_key);
        let pass_descriptor_set = pass_resources
            .map(MeshPassResources::descriptor_set)
            .unwrap_or(vk::DescriptorSet::null());

        let position_buffer = mesh.position_buffer.handle();
        let surface_buffer = mesh.surface_buffer.handle();
        let index_buffer = lod.index_buffer.handle();

        // Safety: the command buffer is recording inside a compatible render pass. The pipeline
        // was created for that pass, and mesh buffers are owned by the renderer until frame end.
        unsafe {
            if state.pipeline != pipeline.handle {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle,
                );
                state.pipeline = pipeline.handle;
            }
            if state.extent != options.extent {
                let viewports = [vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(options.extent.width as f32)
                    .height(options.extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)];
                let scissors = [vk::Rect2D::default()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(options.extent)];
                device.cmd_set_viewport(command_buffer, 0, &viewports);
                device.cmd_set_scissor(command_buffer, 0, &scissors);
                state.extent = options.extent;
            }
            if state.frame_descriptor_set != frame_descriptor_set {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    shader_interface::FRAME_SET,
                    &[frame_descriptor_set],
                    &[],
                );
                state.frame_descriptor_set = frame_descriptor_set;
            }
            if !pipeline_key.opaque_shadow
                && state.material_descriptor_set != material_descriptor_set
            {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    shader_interface::MATERIAL_SET,
                    &[material_descriptor_set],
                    &[],
                );
                state.material_descriptor_set = material_descriptor_set;
            }
            if pass_descriptor_set != vk::DescriptorSet::null()
                && state.pass_descriptor_set != pass_descriptor_set
            {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    shader_interface::PASS_SET,
                    &[pass_descriptor_set],
                    &[],
                );
                state.pass_descriptor_set = pass_descriptor_set;
            }
            if state.shadow_cascade_index != options.shadow_cascade_index {
                if let Some(cascade_index) = options.shadow_cascade_index {
                    device.cmd_push_constants(
                        command_buffer,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytes_of_u32(&cascade_index),
                    );
                }
                state.shadow_cascade_index = options.shadow_cascade_index;
            }
            if pipeline.uses_surface_stream {
                if state.position_buffer != position_buffer
                    || state.surface_buffer != surface_buffer
                {
                    device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &[position_buffer, surface_buffer],
                        &[0, 0],
                    );
                    state.position_buffer = position_buffer;
                    state.surface_buffer = surface_buffer;
                }
            } else if state.position_buffer != position_buffer {
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[position_buffer], &[0]);
                state.position_buffer = position_buffer;
            }
            if state.index_buffer != index_buffer || state.index_type != Some(lod.index_type) {
                device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, lod.index_type);
                state.index_buffer = index_buffer;
                state.index_type = Some(lod.index_type);
            }
            device.cmd_draw_indexed(command_buffer, lod.index_count, 1, 0, 0, 0);
        }

        Ok(true)
    }

    /// Destroys one swapchain-owned mesh pipeline pair.
    pub(super) fn destroy_pipeline_set(&self, device: &Device, pipeline_set: MeshPipelineSet) {
        destroy_pipeline_variants(device, pipeline_set.untextured);
        destroy_pipeline_variants(device, pipeline_set.textured);
        if let Some(opaque_scene) = pipeline_set.opaque_scene {
            destroy_pipeline_variants(device, opaque_scene.untextured);
            destroy_pipeline_variants(device, opaque_scene.textured);
        }
        if let Some(opaque_shadow) = pipeline_set.opaque_shadow {
            destroy_pipeline_variants(device, opaque_shadow);
        }
    }

    /// Destroys backend mesh resources whose protocol handles have retired.
    pub(super) fn destroy_retired(&mut self, device: &Device, retired: &[AssetHandle]) {
        for asset in retired {
            if let AssetHandle::Mesh(mesh) = *asset {
                self.destroy_mesh(device, mesh);
            }
        }
    }

    /// Destroys every mesh buffer and pipeline layout still owned by the backend.
    pub(super) fn destroy(self, device: &Device) {
        for mesh in self.meshes.into_values() {
            mesh.destroy(device);
        }
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_buffers(device, self.frame_uniforms);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.pass_set_layout);
        destroy_descriptor_set_layout(device, self.frame_set_layout);
    }

    /// Destroys one mesh buffer pair when a mesh handle becomes GPU-safe to retire.
    fn destroy_mesh(&mut self, device: &Device, mesh: MeshHandle) {
        if let Some(uploaded) = self.meshes.remove(&mesh) {
            tracing::trace!(
                mesh = mesh.raw(),
                scene = uploaded.scene.raw(),
                indices = uploaded.full_index_count(),
                "destroying retired Vulkan mesh"
            );
            uploaded.destroy(device);
        }
    }
}

impl Default for VulkanMeshStore {
    /// Creates an inert store for tests that do not construct Vulkan resources.
    fn default() -> Self {
        Self {
            meshes: BTreeMap::new(),
            frame_uniforms: Vec::new(),
            frame_descriptor_sets: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            frame_set_layout: vk::DescriptorSetLayout::null(),
            pass_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
        }
    }
}

struct VulkanMesh {
    position_buffer: GpuBuffer,
    surface_buffer: GpuBuffer,
    lods: Vec<VulkanMeshLod>,
    bounds: Option<SceneBounds>,
    scene: SceneHandle,
}

struct VulkanMeshLod {
    level: MeshLodLevel,
    index_buffer: GpuBuffer,
    index_count: u32,
    index_type: vk::IndexType,
}

struct PendingVulkanMesh {
    bounds: Option<SceneBounds>,
    lods: Vec<PendingVulkanMeshLod>,
}

struct PreparedVulkanMesh {
    positions: Vec<GpuMeshPosition>,
    surfaces: Vec<GpuMeshSurface>,
    source_index_count: usize,
    bounds: Option<SceneBounds>,
    lods: Vec<PreparedVulkanMeshLod>,
}

struct PreparedVulkanMeshLod {
    level: MeshLodLevel,
    indices: LodIndexData,
}

struct PendingVulkanMeshLod {
    level: MeshLodLevel,
    index_count: u32,
    index_type: vk::IndexType,
    index_bytes: usize,
}

impl MeshPipelineSet {
    /// Selects the shader variant that matches the material descriptor contract.
    fn choose(self, key: MeshPipelineKey) -> MeshPipeline {
        let variants = if key.opaque_shadow {
            self.opaque_shadow.unwrap_or(self.untextured)
        } else if key.opaque_scene {
            let opaque_scene = self.opaque_scene.unwrap_or(OpaqueScenePipelineVariants {
                untextured: self.untextured,
                textured: self.textured,
            });
            if key.uses_textures {
                opaque_scene.textured
            } else {
                opaque_scene.untextured
            }
        } else if key.uses_textures {
            self.textured
        } else {
            self.untextured
        };
        if key.double_sided {
            variants.double_sided
        } else {
            variants.culled
        }
    }
}

impl MeshPassResources {
    /// Creates one descriptor set for scene-pass sampled graph resources.
    fn create(
        device: &Device,
        pass_set_layout: vk::DescriptorSetLayout,
        depth_shadow_view: vk::ImageView,
        translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
    ) -> Result<Self, VulkanError> {
        let transmittance_sampler = create_pass_sampler(device, vk::Filter::NEAREST)?;
        let local_shadow_sampler = match create_pass_sampler(device, vk::Filter::NEAREST) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_sampler(device, transmittance_sampler);
                return Err(error);
            }
        };
        let depth_shadow_sampler = match create_depth_comparison_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_sampler(device, local_shadow_sampler);
                destroy_sampler(device, transmittance_sampler);
                return Err(error);
            }
        };
        let depth_shadow_raw_sampler = match create_depth_raw_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_sampler(device, depth_shadow_sampler);
                destroy_sampler(device, local_shadow_sampler);
                destroy_sampler(device, transmittance_sampler);
                return Err(error);
            }
        };
        let descriptor_pool = match create_pass_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_sampler(device, local_shadow_sampler);
                destroy_sampler(device, depth_shadow_sampler);
                destroy_sampler(device, depth_shadow_raw_sampler);
                destroy_sampler(device, transmittance_sampler);
                return Err(error);
            }
        };
        let descriptor_set =
            match allocate_pass_descriptor_set(device, descriptor_pool, pass_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    destroy_descriptor_pool(device, descriptor_pool);
                    destroy_sampler(device, local_shadow_sampler);
                    destroy_sampler(device, depth_shadow_sampler);
                    destroy_sampler(device, depth_shadow_raw_sampler);
                    destroy_sampler(device, transmittance_sampler);
                    return Err(error);
                }
            };

        update_pass_descriptor_set(
            device,
            descriptor_set,
            transmittance_sampler,
            local_shadow_sampler,
            depth_shadow_sampler,
            depth_shadow_raw_sampler,
            depth_shadow_view,
            translucent_shadow_views,
            local_shadow_views,
        );
        tracing::info!("created Vulkan mesh pass descriptors");
        Ok(Self {
            descriptor_pool,
            descriptor_set,
            directional_shadow_view: depth_shadow_view,
            transmittance_sampler,
            local_shadow_sampler,
            depth_shadow_sampler,
            depth_shadow_raw_sampler,
        })
    }

    /// Returns the descriptor set bound at `set = 2` for the scene mesh shaders.
    fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    /// Returns the directional CSM view bound to the scene pass.
    pub(super) fn directional_shadow_view(&self) -> vk::ImageView {
        self.directional_shadow_view
    }

    /// Destroys scene-pass descriptor resources before graph target image views are released.
    pub(super) fn destroy(self, device: &Device) {
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.local_shadow_sampler);
        destroy_sampler(device, self.depth_shadow_sampler);
        destroy_sampler(device, self.depth_shadow_raw_sampler);
        destroy_sampler(device, self.transmittance_sampler);
    }
}

impl PreparedVulkanMesh {
    /// Performs polygon-count-dependent conversion and LOD generation without touching Vulkan.
    fn from_imported(imported: &ImportedMesh, reduce_overdraw: bool) -> Self {
        let geometry = MeshGeometry::from_imported(imported);
        let mut positions = Vec::with_capacity(geometry.vertices().len());
        let mut surfaces = Vec::with_capacity(geometry.vertices().len());
        for &vertex in geometry.vertices() {
            positions.push(GpuMeshPosition::from_mesh(vertex));
            surfaces.push(GpuMeshSurface::from_mesh(vertex));
        }
        let pending_lods =
            unique_lod_indices(geometry.vertices(), geometry.indices(), reduce_overdraw);
        let (positions, surfaces, pending_lods) =
            optimize_vertex_fetch_streams(positions, surfaces, pending_lods);
        let lods = compact_lod_buffers(pending_lods);
        Self {
            positions,
            surfaces,
            source_index_count: geometry.index_count(),
            bounds: geometry.bounds(),
            lods,
        }
    }

    fn vertex_count(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.surfaces.len());
        self.positions.len()
    }

    fn source_index_count(&self) -> usize {
        self.source_index_count
    }

    /// Creates staging/device-local buffers in deterministic per-mesh order on the Vulkan thread.
    fn queue(
        self,
        upload_batch: &mut DeviceLocalBufferUploadBatch<'_>,
    ) -> Result<PendingVulkanMesh, VulkanError> {
        upload_batch.push(vk::BufferUsageFlags::VERTEX_BUFFER, &self.positions)?;
        upload_batch.push(vk::BufferUsageFlags::VERTEX_BUFFER, &self.surfaces)?;
        let mut queued_lods = Vec::with_capacity(self.lods.len());
        for lod in self.lods {
            let index_count = lod.indices.len() as u32;
            match &lod.indices {
                LodIndexData::U16(indices) => {
                    upload_batch.push(vk::BufferUsageFlags::INDEX_BUFFER, indices)?
                }
                LodIndexData::U32(indices) => {
                    upload_batch.push(vk::BufferUsageFlags::INDEX_BUFFER, indices)?
                }
            }
            queued_lods.push(PendingVulkanMeshLod {
                level: lod.level,
                index_count,
                index_type: lod.indices.index_type(),
                index_bytes: lod.indices.byte_len(),
            });
        }
        Ok(PendingVulkanMesh {
            bounds: self.bounds,
            lods: queued_lods,
        })
    }
}

impl PendingVulkanMesh {
    /// Pairs buffers returned by the upload batch with the metadata queued for this mesh.
    fn finish(self, buffers: &mut std::vec::IntoIter<GpuBuffer>, scene: SceneHandle) -> VulkanMesh {
        let position_buffer = buffers
            .next()
            .expect("mesh upload batch returns one position buffer per queued mesh");
        let surface_buffer = buffers
            .next()
            .expect("mesh upload batch returns one surface buffer per queued mesh");
        let lods = self
            .lods
            .into_iter()
            .map(|lod| VulkanMeshLod {
                level: lod.level,
                index_buffer: buffers
                    .next()
                    .expect("mesh upload batch returns every queued LOD index buffer"),
                index_count: lod.index_count,
                index_type: lod.index_type,
            })
            .collect();
        VulkanMesh {
            position_buffer,
            surface_buffer,
            lods,
            bounds: self.bounds,
            scene,
        }
    }

    /// Returns total uploaded LOD indices for batch-level upload diagnostics.
    fn lod_index_count_sum(&self) -> usize {
        self.lods.iter().map(|lod| lod.index_count as usize).sum()
    }

    /// Returns actual uploaded index bytes after per-LOD 16-bit compaction.
    fn lod_index_byte_sum(&self) -> usize {
        self.lods.iter().map(|lod| lod.index_bytes).sum()
    }
}

impl VulkanMesh {
    /// Returns the index count for the full-detail LOD.
    fn full_index_count(&self) -> u32 {
        self.lods.first().map_or(0, |lod| lod.index_count)
    }

    /// Returns the chosen LOD unless the mesh is culled by the active scene camera.
    fn visible_lod(
        &self,
        item: &RenderItemPacket,
        options: MeshDrawOptions,
    ) -> Option<&VulkanMeshLod> {
        if let Some(shadow_cull) = options.shadow_cull
            && !shadow_cascade_contains_bounds(self.bounds, &shadow_cull)
        {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                cascade_index = options.shadow_cascade_index,
                "mesh draw skipped by shadow cascade depth culling"
            );
            return None;
        }

        if let Some(level) = options.forced_lod {
            return self.lod(level);
        }
        let Some(camera) = options.camera else {
            return self.lod(MeshLodLevel::Full);
        };
        let aspect = if options.extent.height > 0 {
            options.extent.width as f32 / options.extent.height as f32
        } else {
            1.0
        };
        match classify_mesh(
            camera,
            aspect,
            options.extent.height,
            self.bounds,
            self.full_index_count() as usize / 3,
            options.optimization,
        ) {
            MeshVisibility::Visible { lod } => self.lod(lod),
            MeshVisibility::Culled { reason } => {
                tracing::trace!(
                    mesh = item.mesh.raw(),
                    material = item.material.raw(),
                    reason = ?reason,
                    "mesh draw skipped by renderer visibility culling"
                );
                None
            }
        }
    }

    /// Returns the nearest available backend LOD buffer for the requested detail level.
    fn lod(&self, level: MeshLodLevel) -> Option<&VulkanMeshLod> {
        self.lods
            .iter()
            .filter(|lod| lod.level.index() <= level.index())
            .max_by_key(|lod| lod.level.index())
            .or_else(|| self.lods.first())
            .filter(|lod| lod.index_count >= 3)
    }

    /// Destroys the uploaded vertex and index buffers for one mesh.
    fn destroy(self, device: &Device) {
        for lod in self.lods {
            lod.index_buffer.destroy(device);
        }
        self.surface_buffer.destroy(device);
        self.position_buffer.destroy(device);
    }
}

/// Selects cheaper geometry for shadow cascades whose map texels cannot show full mesh detail.
fn shadow_lod_for_cascade(cascade_index: usize) -> MeshLodLevel {
    if cascade_index >= SHADOW_CASCADE_COUNT {
        return MeshLodLevel::Medium;
    }
    match cascade_index {
        0 => MeshLodLevel::Full,
        // Volumetric shadowing evaluates the same CSM silhouettes at many depths.  A
        // VeryLow caster simplification can collapse thin/alpha-tested structures into one
        // large planar silhouette; the repeated silhouette then appears as a stack of boxes
        // after volumetric prefix integration. Keep the distant floors at Low (the previous
        // quality/performance balance) and let adaptive_directional_shadow_lod promote them
        // further when their projected texel budget requires it.
        1 => MeshLodLevel::Medium,
        2 => MeshLodLevel::Low,
        _ => MeshLodLevel::Low,
    }
}

/// Uses only geometry that cascade one has enough shadow texels to resolve.
fn adaptive_directional_shadow_lod(
    bounds: Option<SceneBounds>,
    texel_world: f32,
    extent: vk::Extent2D,
    lod_triangles: [usize; 4],
    cascade_floor: MeshLodLevel,
) -> MeshLodLevel {
    let Some(bounds) = bounds else {
        return cascade_floor;
    };
    if !texel_world.is_finite()
        || texel_world <= f32::EPSILON
        || extent.width == 0
        || extent.height == 0
    {
        return cascade_floor;
    }

    let radius_texels = bounds.radius() / texel_world;
    if !radius_texels.is_finite() || radius_texels < 0.0 {
        return cascade_floor;
    }
    let map_texels = extent.width as f32 * extent.height as f32;
    let projected_texels =
        (std::f32::consts::PI * radius_texels * radius_texels).clamp(1.0, map_texels);
    let required_triangles =
        (projected_texels * ADAPTIVE_SHADOW_TRIANGLES_PER_TEXEL).max(MIN_ADAPTIVE_SHADOW_TRIANGLES);

    // Pick the cheapest representation that still supplies enough silhouette triangles for the
    // number of shadow texels the caster can cover. The cascade floor preserves the deliberately
    // cheaper policy of distant maps; cascade 0 is no longer forced to Full for sub-pixel detail.
    [
        MeshLodLevel::VeryLow,
        MeshLodLevel::Low,
        MeshLodLevel::Medium,
        MeshLodLevel::Full,
    ]
    .into_iter()
    .find(|level| {
        level.index() >= cascade_floor.index()
            && lod_triangles[level.index()] as f32 >= required_triangles
    })
    .unwrap_or(cascade_floor)
}

/// Returns whether mesh bounds overlap the camera-depth range covered by one shadow cascade.
///
/// Missing bounds stay visible so incomplete import metadata never drops a caster. The padding is
/// intentionally wide because directional shadows can reach into a neighboring cascade.
fn shadow_cascade_contains_bounds(
    bounds: Option<SceneBounds>,
    shadow_cull: &ShadowCascadeCull,
) -> bool {
    let Some(bounds) = bounds else {
        return true;
    };
    if shadow_cull.view_proj_count > 0 {
        return shadow_cull.view_proj[..shadow_cull.view_proj_count]
            .iter()
            .zip(shadow_cull.projection_radius[..shadow_cull.view_proj_count].iter())
            .any(|(view_proj, projection_radius)| {
                shadow_projection_contains_bounds_cached(*view_proj, *projection_radius, bounds)
            });
    }

    // Retain the legacy depth-only fallback for callers that do not have an exact projection.
    // Production directional-shadow culls use the authoritative active projection union above;
    // the heuristic cannot safely account for arbitrary light directions.
    let forward = normalize_or(
        sub3(shadow_cull.camera.target, shadow_cull.camera.eye),
        [0.0, 0.0, -1.0],
    );
    let to_center = sub3(bounds.center(), shadow_cull.camera.eye);
    let depth = dot3(to_center, forward);
    let radius = bounds.radius();
    let range = (shadow_cull.max_depth - shadow_cull.min_depth).max(1.0);
    let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
    if dot3(forward, light_dir).abs() > 0.82 || radius >= range * 0.75 {
        true
    } else {
        let padding = shadow_cascade_depth_padding(shadow_cull, radius);
        let min_depth = shadow_cull.min_depth - radius - padding;
        let max_depth = shadow_cull.max_depth + radius + padding;
        depth.is_finite() && depth >= min_depth && depth <= max_depth
    }
}

/// Conservatively intersects one world-space sphere with a Vulkan orthographic clip volume.
#[cfg(test)]
fn shadow_projection_contains_bounds(view_proj: [f32; 16], bounds: SceneBounds) -> bool {
    shadow_projection_contains_bounds_cached(view_proj, shadow_projection_radius(view_proj), bounds)
}

/// Precomputes the world-sphere expansion along each clip-space row. These values depend only on
/// the cascade projection and are reused for every caster tested against that projection.
fn shadow_projection_radius(view_proj: [f32; 16]) -> [f32; 3] {
    [
        (view_proj[0] * view_proj[0] + view_proj[4] * view_proj[4] + view_proj[8] * view_proj[8])
            .sqrt(),
        (view_proj[1] * view_proj[1] + view_proj[5] * view_proj[5] + view_proj[9] * view_proj[9])
            .sqrt(),
        (view_proj[2] * view_proj[2] + view_proj[6] * view_proj[6] + view_proj[10] * view_proj[10])
            .sqrt(),
    ]
}

fn shadow_projection_contains_bounds_cached(
    view_proj: [f32; 16],
    projection_radius: [f32; 3],
    bounds: SceneBounds,
) -> bool {
    let center = bounds.center();
    let x = view_proj[0] * center[0]
        + view_proj[4] * center[1]
        + view_proj[8] * center[2]
        + view_proj[12];
    let y = view_proj[1] * center[0]
        + view_proj[5] * center[1]
        + view_proj[9] * center[2]
        + view_proj[13];
    let z = view_proj[2] * center[0]
        + view_proj[6] * center[1]
        + view_proj[10] * center[2]
        + view_proj[14];
    let w = view_proj[3] * center[0]
        + view_proj[7] * center[1]
        + view_proj[11] * center[2]
        + view_proj[15];
    if !x.is_finite()
        || !y.is_finite()
        || !z.is_finite()
        || !w.is_finite()
        || w.abs() <= f32::EPSILON
    {
        return true;
    }

    let radius = bounds.radius();
    let radius_x = radius * projection_radius[0];
    let radius_y = radius * projection_radius[1];
    let radius_z = radius * projection_radius[2];
    let limit = w.abs();
    let margin = limit * 0.02;
    x + radius_x >= -limit - margin
        && x - radius_x <= limit + margin
        && y + radius_y >= -limit - margin
        && y - radius_y <= limit + margin
        && z + radius_z >= -margin
        && z - radius_z <= limit + margin
}

/// Computes conservative depth padding for shadow-cascade caster culling.
///
/// Near cascades get a fixed safety margin, while wider cascades receive proportionally more room
/// so long shadows and large meshes are not clipped by the optimization.
fn shadow_cascade_depth_padding(shadow_cull: &ShadowCascadeCull, radius: f32) -> f32 {
    let range = (shadow_cull.max_depth - shadow_cull.min_depth).max(1.0);
    (range * 0.45).clamp(12.0, 96.0) + radius * 2.0
}

/// Generates and compacts all LOD index streams without making Vulkan calls.
fn compact_lod_buffers(pending: Vec<(MeshLodLevel, Vec<u32>)>) -> Vec<PreparedVulkanMeshLod> {
    pending
        .into_iter()
        .map(|(level, indices)| PreparedVulkanMeshLod {
            level,
            indices: compact_lod_indices(indices),
        })
        .collect()
}

/// Reorders both split vertex streams into first-use order and remaps every LOD consistently.
/// This preserves topology while turning indexed vertex reads into substantially more sequential
/// memory traffic. If an imported mesh contains unreferenced vertices, keep the original layout:
/// meshopt's compact remap intentionally omits those entries and cannot remap every LOD safely.
fn optimize_vertex_fetch_streams(
    positions: Vec<GpuMeshPosition>,
    surfaces: Vec<GpuMeshSurface>,
    mut lods: Vec<(MeshLodLevel, Vec<u32>)>,
) -> (
    Vec<GpuMeshPosition>,
    Vec<GpuMeshSurface>,
    Vec<(MeshLodLevel, Vec<u32>)>,
) {
    let Some((_, full_indices)) = lods.first() else {
        return (positions, surfaces, lods);
    };
    let remap = meshopt::optimize_vertex_fetch_remap(full_indices, positions.len());
    if remap.len() != positions.len() || surfaces.len() != positions.len() {
        return (positions, surfaces, lods);
    }

    let vertex_count = positions.len();
    let positions = meshopt::remap_vertex_buffer(&positions, vertex_count, &remap);
    let surfaces = meshopt::remap_vertex_buffer(&surfaces, vertex_count, &remap);
    for (_, indices) in &mut lods {
        *indices = meshopt::remap_index_buffer(Some(indices), vertex_count, &remap);
    }
    (positions, surfaces, lods)
}

/// Returns whether triangle order is free to change during upload-time overdraw reduction.
fn material_reduces_overdraw(material: Option<&ImportedMaterial>) -> bool {
    matches!(
        material.map(ImportedMaterial::alpha_mode),
        Some(MaterialAlphaMode::Opaque | MaterialAlphaMode::Cutout)
    )
}

/// Bounds upload-time CPU concurrency so dense meshes scale across cores without unbounded memory.
fn mesh_prepare_worker_count(mesh_count: usize, total_index_count: usize) -> usize {
    if mesh_count <= 1 || total_index_count < PARALLEL_MESH_PREPARE_MIN_INDICES {
        return 1;
    }
    thread::available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_MESH_PREPARE_WORKERS)
        .min(mesh_count)
        .max(1)
}

fn imported_mesh_index_count(mesh: &ImportedMesh) -> usize {
    match mesh {
        ImportedMesh::Plane => 6,
        ImportedMesh::Indexed(data) => data.indices().len(),
    }
}

enum LodIndexData {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl LodIndexData {
    fn len(&self) -> usize {
        match self {
            Self::U16(indices) => indices.len(),
            Self::U32(indices) => indices.len(),
        }
    }

    fn index_type(&self) -> vk::IndexType {
        match self {
            Self::U16(_) => vk::IndexType::UINT16,
            Self::U32(_) => vk::IndexType::UINT32,
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::U16(indices) => std::mem::size_of_val(indices.as_slice()),
            Self::U32(indices) => std::mem::size_of_val(indices.as_slice()),
        }
    }
}

/// Uses half-width indices whenever one mesh primitive fits Vulkan's 16-bit index range.
fn compact_lod_indices(indices: Vec<u32>) -> LodIndexData {
    if indices.iter().all(|&index| u16::try_from(index).is_ok()) {
        LodIndexData::U16(indices.into_iter().map(|index| index as u16).collect())
    } else {
        LodIndexData::U32(indices)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MeshFrameUniform {
    pub(super) view_proj: [f32; 16],
    pub(super) view: [f32; 16],
    pub(super) shadow_view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
    /// x=blocker-search taps, y=PCF taps, z=light angular radius, w=reserved.
    pub(super) stable_csm_pcss_params: [f32; 4],
    /// x=constant bias, y=slope bias, z=normal offset, w=receiver-plane bias scales.
    pub(super) stable_csm_receiver_params: [f32; 4],
    pub(super) shadow_cascade_splits: [f32; 4],
    pub(super) shadow_cascade_texel_world: [f32; 4],
    pub(super) shadow_cascade_depth_span: [f32; 4],
    /// Physical resolution of each cascade used to convert PCSS radii from texels to UVs.
    pub(super) shadow_cascade_resolution: [f32; 4],
    pub(super) camera_pos: [f32; 4],
    pub(super) light_dir: [f32; 4],
    pub(super) light_color: [f32; 4],
    pub(super) ambient_color: [f32; 4],
    pub(super) local_shadow_view_proj: [[f32; 16]; LOCAL_SHADOW_MATRIX_COUNT],
    pub(super) local_shadow_params: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_position_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_color: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_direction_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_size_kind: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_count: [f32; 4],
    /// x=DebugViewMode value; remaining lanes reserved for future scene debug controls.
    pub(super) debug_view: [f32; 4],
    /// Previous unjittered camera transform used to reproject PCSS visibility.
    pub(super) previous_view_projection: [f32; 16],
    /// x=history valid, y=feedback, z=shared sample phase, w=PCSS light/camera reactivity.
    pub(super) pcss_temporal: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EmissiveLightUniforms {
    pub(super) position_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) color: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) direction_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) size_kind: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) count: [f32; 4],
}

impl EmissiveLightUniforms {
    /// Returns an empty local-light payload for frames without explicit local lights.
    pub(super) fn disabled() -> Self {
        Self {
            position_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
            color: [[0.0; 4]; MAX_LOCAL_LIGHTS],
            direction_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
            size_kind: [[0.0; 4]; MAX_LOCAL_LIGHTS],
            count: [0.0; 4],
        }
    }
}

/// Creates the frame descriptor set layout shared with `shaders/scene/mesh.vert.slang`.
fn create_frame_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(shader_interface::FRAME_CAMERA_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::COMPUTE,
            ),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the pass descriptor set layout for graph-produced scene inputs.
fn create_pass_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    assert_eq!(
        shader_interface::PASS_SHADOW_DEPTH_BINDING,
        9,
        "scene and translucent shadow shaders require directional depth binding 9",
    );
    let bindings = pass_shadow_bindings()
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(pass_shadow_binding_count(binding))
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .to_vec();
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the mesh pipeline layout with frame, material, and pass descriptor sets.
fn create_pipeline_layout(
    device: &Device,
    frame_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    pass_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let mut set_layouts = [frame_set_layout; 3];
    set_layouts[shader_interface::MATERIAL_SET as usize] = material_set_layout;
    set_layouts[shader_interface::PASS_SET as usize] = pass_set_layout;
    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<u32>() as u32)];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);

    // Safety: the descriptor set layout is alive for the duration of this pipeline layout.
    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates one host-visible camera uniform buffer for each frame slot.
fn create_frame_uniforms(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>, VulkanError> {
    let initial = [identity_frame_uniform()];
    let mut buffers = Vec::with_capacity(frame_count);

    for _ in 0..frame_count {
        match create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &initial,
        ) {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => {
                destroy_buffers(device, buffers);
                return Err(error);
            }
        }
    }

    Ok(buffers)
}

/// Creates the descriptor pool used by mesh frame uniforms.
fn create_descriptor_pool(
    device: &Device,
    frame_count: usize,
) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frame_count as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frame_count as u32),
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frame_count as u32)
        .pool_sizes(&pool_sizes);

    // Safety: pool sizes are local values and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates one camera descriptor set per frame slot.
fn allocate_frame_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    frame_set_layout: vk::DescriptorSetLayout,
    frame_count: usize,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    let layouts = vec![frame_set_layout; frame_count];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and layouts are alive for the allocation call.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

/// Writes camera uniform descriptors for every frame descriptor set.
fn update_frame_descriptor_sets(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    frame_uniforms: &[GpuBuffer],
) {
    for (&descriptor_set, uniform) in descriptor_sets.iter().zip(frame_uniforms) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform.handle())
            .offset(0)
            .range(size_of::<MeshFrameUniform>() as vk::DeviceSize)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(shader_interface::FRAME_CAMERA_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];

        // Safety: descriptor sets were allocated from the pool and the buffer infos are valid.
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

/// Creates a sampler for graph target reads performed by mesh scene shaders.
fn create_pass_sampler(device: &Device, filter: vk::Filter) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .min_lod(0.0)
        .max_lod(0.0);

    // Safety: sampler create info contains only local scalar values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the comparison sampler used by Stable CSM's final PCSS filter.
///
/// Linear comparison filtering is Vulkan's hardware 2x2 PCF footprint. The shader performs one
/// `SampleCmpLevelZero` per selected PCSS tap, so every tap retains the hardware-filtered result.
fn create_depth_comparison_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .min_lod(0.0)
        .max_lod(0.0);

    // Safety: sampler create info contains only local scalar values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the bilinear raw-depth sampler used by the PCSS blocker search.
///
/// Blocker taps need the stored depth value rather than a comparison result. Bilinear filtering
/// softens the coverage transition when a search tap crosses a depth texel, while the final
/// percentage-closer filter still uses the comparison sampler above and remains hardware
/// accelerated.
fn create_depth_raw_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .compare_enable(false)
        .min_lod(0.0)
        .max_lod(0.0);

    // Safety: sampler create info contains only local scalar values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Returns pass descriptor bindings for translucent, local, and directional depth shadows.
fn pass_shadow_bindings() -> [u32; SHADOW_CASCADE_COUNT + 3] {
    std::array::from_fn(|index| {
        if index < SHADOW_CASCADE_COUNT {
            shader_interface::PASS_TRANSLUCENT_SHADOW_BINDINGS[index]
        } else if index == SHADOW_CASCADE_COUNT {
            shader_interface::PASS_LOCAL_SHADOW_BINDING
        } else if index == SHADOW_CASCADE_COUNT + 1 {
            shader_interface::PASS_SHADOW_DEPTH_BINDING
        } else {
            shader_interface::PASS_SHADOW_DEPTH_RAW_BINDING
        }
    })
}

fn pass_shadow_binding_count(binding: u32) -> u32 {
    if binding == shader_interface::PASS_LOCAL_SHADOW_BINDING {
        MAX_LOCAL_LIGHTS as u32
    } else {
        1
    }
}

fn pass_descriptor_count() -> u32 {
    pass_shadow_bindings()
        .into_iter()
        .map(pass_shadow_binding_count)
        .sum()
}

/// Views a single `u32` as push-constant bytes for the duration of one Vulkan call.
fn bytes_of_u32(value: &u32) -> &[u8] {
    // Safety: `value` is a plain integer and the returned slice never outlives this stack value.
    unsafe { std::slice::from_raw_parts((value as *const u32).cast::<u8>(), size_of::<u32>()) }
}

/// Creates the descriptor pool for the scene mesh pass resource set.
fn create_pass_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(pass_descriptor_count());
    let pool_sizes = [pool_size];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);

    // Safety: pool size data is local and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates the descriptor set that holds sampled graph targets for mesh scene shaders.
fn allocate_pass_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    pass_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VulkanError> {
    let layouts = [pass_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and set layout are alive for the allocation call.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map(|mut sets| sets.remove(0))
        .map_err(VulkanError::Vk)
}

/// Writes graph-owned shadow target views into the scene mesh pass descriptor set.
fn update_pass_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    transmittance_sampler: vk::Sampler,
    local_shadow_sampler: vk::Sampler,
    depth_shadow_sampler: vk::Sampler,
    depth_shadow_raw_sampler: vk::Sampler,
    depth_shadow_view: vk::ImageView,
    translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
) {
    let transmittance_image_infos = translucent_shadow_views.map(|view| {
        vk::DescriptorImageInfo::default()
            .sampler(transmittance_sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    });
    let local_image_infos = local_shadow_views.map(|view| {
        vk::DescriptorImageInfo::default()
            .sampler(local_shadow_sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    });
    let depth_image_info = [vk::DescriptorImageInfo::default()
        .sampler(depth_shadow_sampler)
        .image_view(depth_shadow_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];
    let depth_raw_image_info = [vk::DescriptorImageInfo::default()
        .sampler(depth_shadow_raw_sampler)
        .image_view(depth_shadow_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];
    let writes = pass_shadow_bindings()
        .iter()
        .map(|&binding| {
            if binding == shader_interface::PASS_LOCAL_SHADOW_BINDING {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&local_image_infos)
            } else if binding == shader_interface::PASS_SHADOW_DEPTH_BINDING {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_image_info)
            } else if binding == shader_interface::PASS_SHADOW_DEPTH_RAW_BINDING {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_raw_image_info)
            } else {
                let transmittance_index = shader_interface::PASS_TRANSLUCENT_SHADOW_BINDINGS
                    .iter()
                    .position(|&transmittance_binding| transmittance_binding == binding)
                    .expect("pass binding must identify a translucent shadow cascade");
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(
                        &transmittance_image_infos[transmittance_index],
                    ))
            }
        })
        .collect::<Vec<_>>();

    // Safety: descriptor set, sampler, and image views belong to this device and remain alive.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Returns a stable initial uniform before the first extracted frame writes camera data.
fn identity_frame_uniform() -> MeshFrameUniform {
    let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);

    MeshFrameUniform {
        view_proj: identity_mat4(),
        view: identity_mat4(),
        shadow_view_proj: [identity_mat4(); SHADOW_CASCADE_COUNT],
        stable_csm_pcss_params: [8.0, 16.0, 0.4_f32.to_radians(), 0.0],
        // Keep startup consistent with the default balanced profile's adaptive bias policy.
        stable_csm_receiver_params: [4.0, 1.5, 1.5, 1.5],
        shadow_cascade_splits: DEFAULT_SHADOW_CASCADE_SPLITS,
        shadow_cascade_texel_world: DEFAULT_SHADOW_CASCADE_METRICS,
        shadow_cascade_depth_span: DEFAULT_SHADOW_CASCADE_METRICS,
        shadow_cascade_resolution: [shadow_map_size() as f32; SHADOW_CASCADE_COUNT],
        camera_pos: [0.0, 0.0, 0.0, 1.0],
        light_dir: [light_dir[0], light_dir[1], light_dir[2], 0.0],
        light_color: [
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[0],
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[1],
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[2],
            0.0,
        ],
        ambient_color: DEFAULT_AMBIENT_COLOR,
        local_shadow_view_proj: [identity_mat4(); LOCAL_SHADOW_MATRIX_COUNT],
        local_shadow_params: [[0.0, 0.0, 1.0, 1.0]; MAX_LOCAL_LIGHTS],
        emissive_light_position_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_color: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_direction_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_size_kind: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_count: [0.0; 4],
        debug_view: [0.0; 4],
        previous_view_projection: identity_mat4(),
        pcss_temporal: [0.0; 4],
    }
}

#[derive(Clone, Copy)]
enum MeshPipelineTarget {
    SceneOpaque,
    SceneOpaqueFast,
    SceneTransparent,
    SceneTransparentFast,
    OpaqueShadow,
    TranslucentShadow,
    LocalShadowDepth,
}

#[derive(Clone, Copy)]
enum MeshVertexLayout {
    SceneUntextured,
    SceneTextured,
    Shadow,
    ShadowOpaque,
}

impl MeshVertexLayout {
    /// Returns whether this pipeline reads the material/shading vertex stream at binding 1.
    fn uses_surface_stream(self) -> bool {
        !matches!(self, Self::ShadowOpaque)
    }
}

impl MeshPipelineTarget {
    /// Returns how many color attachments the target writes.
    fn color_attachment_count(self) -> usize {
        match self {
            Self::SceneOpaque | Self::SceneTransparent => 4,
            Self::SceneOpaqueFast | Self::SceneTransparentFast => 1,
            Self::TranslucentShadow => 1,
            Self::OpaqueShadow | Self::LocalShadowDepth => 0,
        }
    }

    /// Returns whether polygon depth bias should be enabled for this pass.
    fn uses_depth_bias(self) -> bool {
        matches!(self, Self::OpaqueShadow | Self::LocalShadowDepth)
    }

    /// Returns whether the vertex shader consumes per-vertex normals.
    fn uses_surface_normal(self) -> bool {
        matches!(
            self,
            Self::SceneOpaque
                | Self::SceneOpaqueFast
                | Self::SceneTransparent
                | Self::SceneTransparentFast
        )
    }

    /// Returns whether fixed-function depth testing is active for this pass.
    fn uses_depth_test(self) -> bool {
        !matches!(self, Self::TranslucentShadow)
    }

    /// Returns whether the pass should write depth values.
    fn writes_depth(self) -> bool {
        matches!(
            self,
            Self::SceneOpaque | Self::SceneOpaqueFast | Self::OpaqueShadow | Self::LocalShadowDepth
        )
    }

    /// Uses strict depth rejection for opaque geometry so equal-depth duplicate surfaces never
    /// enter expensive fragment shading. Blended/transmittance passes retain equal-depth access.
    fn depth_compare_op(self) -> vk::CompareOp {
        match self {
            Self::SceneOpaque
            | Self::SceneOpaqueFast
            | Self::OpaqueShadow
            | Self::LocalShadowDepth => vk::CompareOp::LESS,
            Self::SceneTransparent | Self::SceneTransparentFast | Self::TranslucentShadow => {
                vk::CompareOp::LESS_OR_EQUAL
            }
        }
    }

    /// Creates the color blend state for scene color, scene metadata, or shadow transmittance.
    fn color_blend_attachments(self) -> Vec<vk::PipelineColorBlendAttachmentState> {
        let write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        match self {
            Self::SceneOpaque => vec![
                vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask),
                vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask),
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::empty()),
                // `out_pcss_shadow.a` is 1 for normal scene shading and 0 for diagnostics. The
                // alpha gate lets the same pipeline preserve the LOADed history attachment while
                // a debug view is active; normal opaque fragments still replace the cleared value
                // exactly once.
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .color_write_mask(write_mask),
            ],
            Self::SceneOpaqueFast => {
                vec![vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask)]
            }
            Self::SceneTransparent => vec![
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .color_write_mask(write_mask),
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::empty()),
                vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask),
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::empty()),
            ],
            Self::SceneTransparentFast => vec![
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .color_write_mask(write_mask),
            ],
            Self::OpaqueShadow | Self::LocalShadowDepth => Vec::new(),
            Self::TranslucentShadow => vec![
                // Transparent casters are accumulated front-to-back independently of draw order:
                // RGB adds log(transmittance), while alpha keeps the nearest layer depth.
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::ONE)
                    .dst_color_blend_factor(vk::BlendFactor::ONE)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                    .alpha_blend_op(vk::BlendOp::MIN)
                    .color_write_mask(write_mask),
            ],
        }
    }
}

/// Creates the graphics pipeline that renders indexed mesh packets into the swapchain pass.
fn create_mesh_pipeline_variants(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader_bytes: &[u8],
    fragment_shader_bytes: &[u8],
    vertex_layout: MeshVertexLayout,
    target: MeshPipelineTarget,
) -> Result<MeshPipelineVariants, VulkanError> {
    let uses_surface_stream = vertex_layout.uses_surface_stream();
    let culled = create_mesh_pipeline(
        device,
        pipeline_layout,
        render_pass,
        vertex_shader_bytes,
        fragment_shader_bytes,
        vertex_layout,
        target,
        vk::CullModeFlags::BACK,
    )?;
    let double_sided = match create_mesh_pipeline(
        device,
        pipeline_layout,
        render_pass,
        vertex_shader_bytes,
        fragment_shader_bytes,
        vertex_layout,
        target,
        vk::CullModeFlags::NONE,
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            destroy_pipeline(device, culled);
            return Err(error);
        }
    };

    Ok(MeshPipelineVariants {
        culled: MeshPipeline {
            handle: culled,
            uses_surface_stream,
        },
        double_sided: MeshPipeline {
            handle: double_sided,
            uses_surface_stream,
        },
    })
}

/// Creates one graphics pipeline with an explicit culling policy.
fn create_mesh_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader_bytes: &[u8],
    fragment_shader_bytes: &[u8],
    vertex_layout: MeshVertexLayout,
    target: MeshPipelineTarget,
    cull_mode: vk::CullModeFlags,
) -> Result<vk::Pipeline, VulkanError> {
    let vertex_shader = shader::create_shader_module(device, vertex_shader_bytes)?;
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
        vertex_layout,
        target,
        cull_mode,
    );

    shader::destroy_shader_module(device, fragment_shader);
    shader::destroy_shader_module(device, vertex_shader);
    pipeline
}

/// Creates the fixed-function pipeline state for the current mesh vertex format.
fn create_graphics_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    vertex_layout: MeshVertexLayout,
    target: MeshPipelineTarget,
    cull_mode: vk::CullModeFlags,
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
    let vertex_bindings = vertex_bindings_for_layout(vertex_layout);
    let vertex_attributes = vertex_attributes_for_layout(vertex_layout);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull_mode)
        .front_face(MESH_FRONT_FACE)
        .depth_bias_enable(target.uses_depth_bias())
        .depth_bias_constant_factor(if target.uses_depth_bias() {
            SHADOW_DEPTH_BIAS_CONSTANT
        } else {
            0.0
        })
        .depth_bias_slope_factor(if target.uses_depth_bias() {
            SHADOW_DEPTH_BIAS_SLOPE
        } else {
            0.0
        })
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(target.uses_depth_test())
        .depth_write_enable(target.writes_depth())
        .depth_compare_op(target.depth_compare_op());
    let color_blend_attachments = target.color_blend_attachments();
    debug_assert_eq!(
        color_blend_attachments.len(),
        target.color_attachment_count()
    );
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
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_infos = [pipeline_info];

    // Safety: all pipeline state references live for the duration of the call, and the render pass
    // is compatible with the framebuffer pass used during command recording.
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

/// Returns the tightly packed streams consumed by the selected shader path.
fn vertex_bindings_for_layout(layout: MeshVertexLayout) -> Vec<vk::VertexInputBindingDescription> {
    let mut bindings = Vec::with_capacity(2);
    bindings.push(
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<GpuMeshPosition>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX),
    );
    if layout.uses_surface_stream() {
        bindings.push(
            vk::VertexInputBindingDescription::default()
                .binding(1)
                .stride(size_of::<GpuMeshSurface>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
        );
    }
    bindings
}

/// Returns only the vertex attributes consumed by the selected shader path.
fn vertex_attributes_for_layout(
    layout: MeshVertexLayout,
) -> Vec<vk::VertexInputAttributeDescription> {
    let mut attributes = Vec::with_capacity(5);
    attributes.push(
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(GpuMeshPosition, position) as u32),
    );
    if matches!(
        layout,
        MeshVertexLayout::SceneUntextured | MeshVertexLayout::SceneTextured
    ) {
        attributes.push(
            vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(1)
                .format(vk::Format::A2B10G10R10_SNORM_PACK32)
                .offset(offset_of!(GpuMeshSurface, normal) as u32),
        );
    }

    match layout {
        MeshVertexLayout::SceneUntextured => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(4)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(GpuMeshSurface, color) as u32),
            );
        }
        MeshVertexLayout::SceneTextured => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(GpuMeshSurface, uv) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(3)
                    .format(vk::Format::A2B10G10R10_SNORM_PACK32)
                    .offset(offset_of!(GpuMeshSurface, tangent) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(4)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(GpuMeshSurface, color) as u32),
            );
        }
        MeshVertexLayout::Shadow => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(GpuMeshSurface, uv) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(1)
                    .location(3)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(GpuMeshSurface, color) as u32),
            );
        }
        MeshVertexLayout::ShadowOpaque => {}
    }

    attributes
}

fn pack_snorm_10_10_10_2(value: [f32; 4]) -> u32 {
    pack_snorm_component(value[0], 10)
        | (pack_snorm_component(value[1], 10) << 10)
        | (pack_snorm_component(value[2], 10) << 20)
        | (pack_snorm_component(value[3], 2) << 30)
}

fn pack_snorm_component(value: f32, bits: u32) -> u32 {
    let max = (1_i32 << (bits - 1)) - 1;
    let mask = (1_u32 << bits) - 1;
    let encoded = (value.clamp(-1.0, 1.0) * max as f32).round() as i32;
    (encoded as u32) & mask
}

fn pack_unorm_8_8_8_8(value: [f32; 4]) -> u32 {
    value
        .into_iter()
        .enumerate()
        .fold(0_u32, |packed, (index, component)| {
            let encoded = (component.clamp(0.0, 1.0) * 255.0).round() as u32;
            packed | (encoded << (index * 8))
        })
}

/// Destroys one pipeline layout after swapchain-owned pipelines are gone.
fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout == vk::PipelineLayout::null() {
        return;
    }

    // Safety: all pipelines that reference this layout are destroyed before the layout.
    unsafe {
        device.destroy_pipeline_layout(layout, None);
    }
}

/// Destroys one descriptor set layout after dependent pools and pipelines are gone.
fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout == vk::DescriptorSetLayout::null() {
        return;
    }

    // Safety: descriptor set layouts are destroyed after dependent pools and pipeline layouts.
    unsafe {
        device.destroy_descriptor_set_layout(layout, None);
    }
}

/// Destroys one descriptor pool and all descriptor sets allocated from it.
fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool == vk::DescriptorPool::null() {
        return;
    }

    // Safety: descriptor sets from this pool are no longer referenced by in-flight commands.
    unsafe {
        device.destroy_descriptor_pool(pool, None);
    }
}

/// Destroys one sampler created for mesh pass resources.
fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler == vk::Sampler::null() {
        return;
    }

    // Safety: the sampler was created by this device and is no longer referenced by commands.
    unsafe {
        device.destroy_sampler(sampler, None);
    }
}

/// Destroys one graphics pipeline.
fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline == vk::Pipeline::null() {
        return;
    }

    // Safety: the pipeline was created by this device and is no longer referenced by commands.
    unsafe {
        device.destroy_pipeline(pipeline, None);
    }
}

/// Destroys both culling variants for one shader/material path.
fn destroy_pipeline_variants(device: &Device, variants: MeshPipelineVariants) {
    destroy_pipeline(device, variants.culled.handle);
    destroy_pipeline(device, variants.double_sided.handle);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shadow_projection() -> [f32; 16] {
        let mut projection = identity_mat4();
        projection[10] = 0.0;
        projection[14] = 0.5;
        projection
    }

    #[test]
    fn stable_csm_frame_uniform_matches_shader_abi() {
        assert_eq!(offset_of!(MeshFrameUniform, view_proj), 0);
        assert_eq!(offset_of!(MeshFrameUniform, view), 64);
        assert_eq!(offset_of!(MeshFrameUniform, shadow_view_proj), 128);
        assert_eq!(offset_of!(MeshFrameUniform, stable_csm_pcss_params), 384);
        assert_eq!(
            offset_of!(MeshFrameUniform, stable_csm_receiver_params),
            400
        );
        assert_eq!(offset_of!(MeshFrameUniform, shadow_cascade_splits), 416);
        assert_eq!(
            offset_of!(MeshFrameUniform, shadow_cascade_texel_world),
            432
        );
        assert_eq!(offset_of!(MeshFrameUniform, shadow_cascade_depth_span), 448);
        assert_eq!(offset_of!(MeshFrameUniform, shadow_cascade_resolution), 464);
        assert_eq!(offset_of!(MeshFrameUniform, camera_pos), 480);
        assert_eq!(offset_of!(MeshFrameUniform, light_dir), 496);
        assert_eq!(offset_of!(MeshFrameUniform, light_color), 512);
        assert_eq!(offset_of!(MeshFrameUniform, ambient_color), 528);
        assert_eq!(offset_of!(MeshFrameUniform, local_shadow_view_proj), 544);
        assert_eq!(offset_of!(MeshFrameUniform, debug_view), 2416);
        assert_eq!(offset_of!(MeshFrameUniform, previous_view_projection), 2432);
        assert_eq!(offset_of!(MeshFrameUniform, pcss_temporal), 2496);
        assert_eq!(size_of::<MeshFrameUniform>(), 2512);
    }

    #[test]
    fn split_gpu_vertex_layout_halves_full_precision_vertex_bandwidth() {
        assert_eq!(size_of::<MeshVertex>(), 64);
        assert_eq!(size_of::<GpuMeshPosition>(), 12);
        assert_eq!(size_of::<GpuMeshSurface>(), 20);
        assert_eq!(
            size_of::<GpuMeshPosition>() + size_of::<GpuMeshSurface>(),
            32
        );

        let vertex = MeshVertex::new_with_tangent(
            [1.0, 2.0, 3.0],
            [1.0, -1.0, 0.0],
            [0.25, 0.75],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.5, 1.0, 1.0],
        );
        let position = GpuMeshPosition::from_mesh(vertex);
        let surface = GpuMeshSurface::from_mesh(vertex);
        assert_eq!(position.position, [1.0, 2.0, 3.0]);
        assert_eq!(surface.uv, [0.25, 0.75]);
        assert_eq!(surface.normal & 0x3ff, 511);
        assert_eq!((surface.normal >> 10) & 0x3ff, 513);
        assert_eq!((surface.tangent >> 30) & 0x3, 3);
        assert_eq!(surface.color, 0xffff_8000);
    }

    #[test]
    fn compact_lod_indices_uses_smallest_supported_vulkan_index_type() {
        let small = compact_lod_indices(vec![0, 1, u16::MAX as u32]);
        assert_eq!(small.index_type(), vk::IndexType::UINT16);
        assert_eq!(small.byte_len(), 3 * size_of::<u16>());

        let large = compact_lod_indices(vec![0, u16::MAX as u32 + 1, 1]);
        assert_eq!(large.index_type(), vk::IndexType::UINT32);
        assert_eq!(large.byte_len(), 3 * size_of::<u32>());
    }

    #[test]
    fn vertex_fetch_remap_preserves_split_stream_topology_for_every_lod() {
        let positions = (0..4)
            .map(|index| GpuMeshPosition {
                position: [index as f32, 0.0, 0.0],
            })
            .collect::<Vec<_>>();
        let surfaces = (0..4)
            .map(|index| GpuMeshSurface {
                color: index,
                ..GpuMeshSurface::default()
            })
            .collect::<Vec<_>>();
        let full = vec![3, 1, 2, 3, 0, 1];
        let low = vec![3, 0, 1];
        let (positions, surfaces, lods) = optimize_vertex_fetch_streams(
            positions,
            surfaces,
            vec![
                (MeshLodLevel::Full, full.clone()),
                (MeshLodLevel::Low, low.clone()),
            ],
        );

        for ((_, remapped), original) in lods.iter().zip([full, low]) {
            let remapped_positions = remapped
                .iter()
                .map(|index| positions[*index as usize].position[0] as u32)
                .collect::<Vec<_>>();
            assert_eq!(remapped_positions, original);
        }
        for (position, surface) in positions.iter().zip(surfaces) {
            assert_eq!(position.position[0] as u32, surface.color);
        }
    }

    // Verifies that mesh back-face culling keeps the glTF/old renderer winding convention.
    #[test]
    fn mesh_front_face_matches_imported_winding() {
        assert_eq!(MESH_FRONT_FACE, vk::FrontFace::COUNTER_CLOCKWISE);
    }

    // Verifies that distant global cascades are cheap while local cubemap faces keep close detail.
    #[test]
    fn shadow_cascades_select_progressively_cheaper_lods() {
        assert_eq!(shadow_lod_for_cascade(0), MeshLodLevel::Full);
        assert_eq!(shadow_lod_for_cascade(1), MeshLodLevel::Medium);
        assert_eq!(shadow_lod_for_cascade(2), MeshLodLevel::Low);
        assert_eq!(shadow_lod_for_cascade(3), MeshLodLevel::Low);
        assert_eq!(
            shadow_lod_for_cascade(SHADOW_CASCADE_COUNT),
            MeshLodLevel::Medium
        );
    }

    #[test]
    fn adaptive_shadow_lod_uses_very_low_for_small_dense_casters() {
        let bounds = SceneBounds::new([0.0, 0.0, 0.0], 1.0).expect("test bounds are finite");
        let extent = vk::Extent2D {
            width: 1536,
            height: 1536,
        };

        assert_eq!(
            adaptive_directional_shadow_lod(
                Some(bounds),
                1.0,
                extent,
                [2_560, 1_536, 768, 256],
                MeshLodLevel::Full,
            ),
            MeshLodLevel::VeryLow
        );
    }

    #[test]
    fn adaptive_shadow_lod_keeps_resolvable_detail_for_large_casters() {
        let bounds = SceneBounds::new([0.0, 0.0, 0.0], 100.0).expect("test bounds are finite");
        let extent = vk::Extent2D {
            width: 1536,
            height: 1536,
        };

        assert_eq!(
            adaptive_directional_shadow_lod(
                Some(bounds),
                1.0,
                extent,
                [25_600, 15_360, 7_680, 2_560],
                MeshLodLevel::Full,
            ),
            MeshLodLevel::Medium
        );
    }

    #[test]
    fn adaptive_shadow_lod_fails_open_without_bounds() {
        let extent = vk::Extent2D {
            width: 1536,
            height: 1536,
        };

        assert_eq!(
            adaptive_directional_shadow_lod(None, 1.0, extent, [usize::MAX; 4], MeshLodLevel::Full,),
            MeshLodLevel::Full
        );
    }

    // Verifies that shadow cascade culling keeps overlapping casters and drops only far outsiders.
    #[test]
    fn shadow_cascade_depth_culling_is_conservative() {
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let cull = ShadowCascadeCull::new(camera, 0.1, 12.0);
        let near_bounds = SceneBounds::new([0.0, 0.0, -10.0], 1.0);
        let far_bounds = SceneBounds::new([0.0, 0.0, -80.0], 1.0);
        let large_bounds = SceneBounds::new([0.0, 0.0, -80.0], 12.0);

        assert!(shadow_cascade_contains_bounds(near_bounds, &cull));
        assert!(!shadow_cascade_contains_bounds(far_bounds, &cull));
        assert!(shadow_cascade_contains_bounds(large_bounds, &cull));
        assert!(shadow_cascade_contains_bounds(None, &cull));
    }

    #[test]
    fn shadow_projection_culling_keeps_edge_crossings_and_drops_lateral_outsiders() {
        let inside =
            SceneBounds::new([0.0, 0.0, 0.0], 0.1).expect("inside shadow bounds are finite");
        let edge = SceneBounds::new([1.05, 0.0, 0.0], 0.1).expect("edge shadow bounds are finite");
        let outside =
            SceneBounds::new([3.0, 0.0, 0.0], 0.1).expect("outside shadow bounds are finite");

        assert!(shadow_projection_contains_bounds(identity_mat4(), inside));
        assert!(shadow_projection_contains_bounds(identity_mat4(), edge));
        assert!(!shadow_projection_contains_bounds(identity_mat4(), outside));
    }

    #[test]
    fn shadow_projection_culling_matches_vulkan_depth_clip() {
        let inside =
            SceneBounds::new([0.0, 0.0, 0.5], 0.01).expect("inside depth bounds are finite");
        let before_near =
            SceneBounds::new([0.0, 0.0, -0.2], 0.01).expect("near-clipped bounds are finite");
        let beyond_far =
            SceneBounds::new([0.0, 0.0, 1.2], 0.01).expect("far-clipped bounds are finite");

        assert!(shadow_projection_contains_bounds(identity_mat4(), inside));
        assert!(!shadow_projection_contains_bounds(
            identity_mat4(),
            before_near
        ));
        assert!(!shadow_projection_contains_bounds(
            identity_mat4(),
            beyond_far
        ));
    }

    #[test]
    fn shadow_projection_culling_keeps_a_projected_caster() {
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let witness =
            SceneBounds::new([1.2, 0.0, -10.0], 0.01).expect("projected witness bounds are finite");
        let primary_only = ShadowCascadeCull::new(camera, 0.1, 12.0)
            .with_light_space_projection(test_shadow_projection());
        let mut projections = [test_shadow_projection(); SHADOW_CASCADE_COUNT];
        projections[0][0] = 0.5;
        let projected =
            ShadowCascadeCull::new(camera, 0.1, 12.0).with_light_space_projections(projections, 1);

        assert!(!primary_only.contains_bounds(Some(witness)));
        assert!(projected.contains_bounds(Some(witness)));
    }

    #[test]
    fn shadow_projection_culling_uses_only_active_cascade_projection() {
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let witness = SceneBounds::new([0.0, 0.0, 0.5], 0.01)
            .expect("inactive-slot witness bounds are finite");
        let mut projections = [identity_mat4(); SHADOW_CASCADE_COUNT];
        projections[0][12] = 2.0;
        let active_only =
            ShadowCascadeCull::new(camera, 0.1, 12.0).with_light_space_projections(projections, 1);
        let including_identity =
            ShadowCascadeCull::new(camera, 0.1, 12.0).with_light_space_projections(projections, 2);

        assert_eq!(active_only.light_space_projection_count(), 1);
        assert!(!active_only.contains_bounds(Some(witness)));
        assert!(including_identity.contains_bounds(Some(witness)));
    }

    #[test]
    fn exact_shadow_projection_bypasses_camera_depth_heuristic() {
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let caster = SceneBounds::new([0.0, 0.0, -80.0], 0.01)
            .expect("projection-authoritative caster bounds are finite");
        let depth_only = ShadowCascadeCull::new(camera, 0.1, 12.0);
        let mut projection = identity_mat4();
        projection[10] = -0.01;
        let exact =
            ShadowCascadeCull::new(camera, 0.1, 12.0).with_light_space_projection(projection);

        assert!(!depth_only.contains_bounds(Some(caster)));
        assert!(exact.contains_bounds(Some(caster)));
    }

    #[test]
    fn transparent_scene_target_preserves_depth_and_writes_material_metadata() {
        let attachments = MeshPipelineTarget::SceneTransparent.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneTransparent.uses_depth_test());
        assert!(!MeshPipelineTarget::SceneTransparent.writes_depth());
        assert_eq!(
            MeshPipelineTarget::SceneTransparent.depth_compare_op(),
            vk::CompareOp::LESS_OR_EQUAL
        );
        assert_eq!(attachments.len(), 4);
        assert!(attachments[0].blend_enable != 0);
        assert_eq!(
            attachments[1].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_ne!(
            attachments[2].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_eq!(
            attachments[3].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
    }

    #[test]
    fn opaque_scene_target_writes_depth_and_material_metadata() {
        let attachments = MeshPipelineTarget::SceneOpaque.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneOpaque.uses_depth_test());
        assert!(MeshPipelineTarget::SceneOpaque.writes_depth());
        assert_eq!(
            MeshPipelineTarget::SceneOpaque.depth_compare_op(),
            vk::CompareOp::LESS
        );
        assert_eq!(attachments.len(), 4);
        assert!(attachments[0].blend_enable == 0);
        assert_ne!(
            attachments[1].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_eq!(
            attachments[2].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_ne!(
            attachments[3].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
    }

    #[test]
    fn transparent_fast_scene_target_blends_color_without_material_metadata() {
        let attachments = MeshPipelineTarget::SceneTransparentFast.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneTransparentFast.uses_depth_test());
        assert!(!MeshPipelineTarget::SceneTransparentFast.writes_depth());
        assert_eq!(attachments.len(), 1);
        assert!(attachments[0].blend_enable != 0);
    }

    #[test]
    fn opaque_fast_scene_target_writes_depth_without_material_metadata() {
        let attachments = MeshPipelineTarget::SceneOpaqueFast.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneOpaqueFast.uses_depth_test());
        assert!(MeshPipelineTarget::SceneOpaqueFast.writes_depth());
        assert_eq!(attachments.len(), 1);
        assert!(attachments[0].blend_enable == 0);
    }

    #[test]
    fn translucent_shadow_target_accumulates_deep_log_transmittance() {
        let attachments = MeshPipelineTarget::TranslucentShadow.color_blend_attachments();

        assert!(!MeshPipelineTarget::TranslucentShadow.uses_depth_test());
        assert!(!MeshPipelineTarget::TranslucentShadow.writes_depth());
        assert_eq!(attachments.len(), 1);
        assert!(attachments[0].blend_enable != 0);
        assert_eq!(attachments[0].color_blend_op, vk::BlendOp::ADD);
        assert_eq!(attachments[0].alpha_blend_op, vk::BlendOp::MIN);
        assert_ne!(
            attachments[0].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
    }

    #[test]
    fn overdraw_reordering_excludes_alpha_blended_materials() {
        let opaque = ImportedMaterial::opaque();
        let cutout = ImportedMaterial::new(MaterialAlphaMode::Cutout, 500, Vec::new());
        let transparent = ImportedMaterial::new(MaterialAlphaMode::Transparent, 500, Vec::new());

        assert!(material_reduces_overdraw(Some(&opaque)));
        assert!(material_reduces_overdraw(Some(&cutout)));
        assert!(!material_reduces_overdraw(Some(&transparent)));
        assert!(!material_reduces_overdraw(None));
    }

    #[test]
    fn untextured_scene_vertex_layout_skips_texture_attributes() {
        let attributes = vertex_attributes_for_layout(MeshVertexLayout::SceneUntextured);
        let locations = attributes
            .into_iter()
            .map(|attribute| attribute.location)
            .collect::<Vec<_>>();

        assert_eq!(locations, vec![0, 1, 4]);
    }

    #[test]
    fn textured_scene_vertex_layout_keeps_normal_mapping_attributes() {
        let attributes = vertex_attributes_for_layout(MeshVertexLayout::SceneTextured);
        let locations = attributes
            .into_iter()
            .map(|attribute| attribute.location)
            .collect::<Vec<_>>();

        assert_eq!(locations, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn opaque_shadow_vertex_layout_binds_only_the_position_stream() {
        let bindings = vertex_bindings_for_layout(MeshVertexLayout::ShadowOpaque);
        let attributes = vertex_attributes_for_layout(MeshVertexLayout::ShadowOpaque);

        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[0].stride, 12);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0].binding, 0);
        assert_eq!(attributes[0].location, 0);
    }

    #[test]
    fn alpha_shadow_layout_skips_unused_normal_but_keeps_surface_stream() {
        let bindings = vertex_bindings_for_layout(MeshVertexLayout::Shadow);
        let attributes = vertex_attributes_for_layout(MeshVertexLayout::Shadow);
        let locations = attributes
            .iter()
            .map(|attribute| attribute.location)
            .collect::<Vec<_>>();

        assert_eq!(bindings.len(), 2);
        assert_eq!((bindings[0].binding, bindings[0].stride), (0, 12));
        assert_eq!((bindings[1].binding, bindings[1].stride), (1, 20));
        assert_eq!(locations, vec![0, 2, 3]);
        assert_eq!(attributes[0].binding, 0);
        assert!(
            attributes[1..]
                .iter()
                .all(|attribute| attribute.binding == 1)
        );
        assert!(!bindings.iter().any(|binding| binding.binding == 2));
    }

    #[test]
    fn scene_layout_uses_position_and_surface_streams_only() {
        for layout in [
            MeshVertexLayout::SceneUntextured,
            MeshVertexLayout::SceneTextured,
        ] {
            let bindings = vertex_bindings_for_layout(layout);
            let attributes = vertex_attributes_for_layout(layout);

            assert_eq!(bindings.len(), 2);
            assert_eq!((bindings[0].binding, bindings[0].stride), (0, 12));
            assert_eq!((bindings[1].binding, bindings[1].stride), (1, 20));
            assert!(attributes.iter().all(|attribute| attribute.binding <= 1));
        }
    }
}
