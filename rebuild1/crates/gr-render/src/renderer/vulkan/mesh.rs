use std::{
    collections::BTreeMap,
    ffi::CStr,
    io::Cursor,
    mem::{offset_of, size_of},
};

use ash::{Device, Instance, util, vk};

use crate::{
    import::ImportedMesh,
    math::{dot3, identity_mat4, normalize_or, sub3},
    protocol::{
        AssetHandle, CameraSnapshot, MeshHandle, RenderItemPacket, RenderOptimizationSettings,
        SceneBounds,
    },
    renderer::{
        DEFAULT_AMBIENT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_COLOR, DEFAULT_DIRECTIONAL_LIGHT_DIR,
        DEFAULT_SHADOW_CASCADE_METRICS, DEFAULT_SHADOW_CASCADE_SPLITS,
        assets::{MeshGeometry, MeshVertex},
        graph::SHADOW_CASCADE_COUNT,
        pipeline::shader_interface,
        visibility::{MeshLodLevel, MeshVisibility, classify_mesh},
    },
};

use super::{
    VulkanError,
    buffer::{
        GpuBuffer, create_buffer_with_data, create_device_local_buffer_with_data, destroy_buffers,
        memory_properties, write_buffer_value,
    },
    lod::unique_lod_indices,
};

const SHADER_ENTRY: &CStr = c"main";
const UNTEXTURED_VERTEX_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_untextured.vert.spv"));
const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.vert.spv"));
const SCENE_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_scene.frag.spv"));
const SCENE_FAST_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_scene_fast.frag.spv"));
const SCENE_TEXTURED_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_scene_textured.frag.spv"));
const SCENE_TEXTURED_FAST_FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/mesh_scene_textured_fast.frag.spv"
));
const SHADOW_VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
const SHADOW_FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));
const SHADOW_TEXTURED_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_textured.frag.spv"));
const SHADOW_DEPTH_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_depth.frag.spv"));
const SHADOW_DEPTH_TEXTURED_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_depth_textured.frag.spv"));
const SHADOW_TRANSLUCENT_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_translucent.frag.spv"));
const SHADOW_TRANSLUCENT_TEXTURED_FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/shadow_translucent_textured.frag.spv"
));
const MESH_FRONT_FACE: vk::FrontFace = vk::FrontFace::COUNTER_CLOCKWISE;
const SHADOW_DEPTH_BIAS_CONSTANT: f32 = 0.35;
const SHADOW_DEPTH_BIAS_SLOPE: f32 = 0.65;
pub(super) const MAX_LOCAL_LIGHTS: usize = 4;
pub(super) const LOCAL_SHADOW_FACE_COUNT: usize = 6;
pub(super) const LOCAL_SHADOW_MATRIX_COUNT: usize = MAX_LOCAL_LIGHTS * LOCAL_SHADOW_FACE_COUNT;

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
}

#[derive(Clone, Copy)]
pub(super) struct MeshPipelineKey {
    pub(super) uses_textures: bool,
    pub(super) double_sided: bool,
}

#[derive(Clone, Copy, Default)]
pub(super) struct MeshDrawState {
    pipeline: vk::Pipeline,
    frame_descriptor_set: vk::DescriptorSet,
    material_descriptor_set: vk::DescriptorSet,
    pass_descriptor_set: vk::DescriptorSet,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    extent: vk::Extent2D,
    shadow_cascade_index: Option<u32>,
}

#[derive(Clone, Copy)]
pub(super) struct MeshPipelineSet {
    untextured: MeshPipelineVariants,
    textured: MeshPipelineVariants,
}

#[derive(Clone, Copy)]
struct MeshPipelineVariants {
    culled: MeshPipeline,
    double_sided: MeshPipeline,
}

pub(super) struct MeshPassResources {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    shadow_sampler: vk::Sampler,
    transmittance_sampler: vk::Sampler,
    local_shadow_sampler: vk::Sampler,
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

#[derive(Clone, Copy)]
pub(super) struct ShadowCascadeCull {
    camera: CameraSnapshot,
    min_depth: f32,
    max_depth: f32,
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
        }
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
    pub(super) fn shadow_preculled(extent: vk::Extent2D, cascade_index: usize) -> Self {
        Self {
            extent,
            camera: None,
            optimization: RenderOptimizationSettings::disabled(),
            forced_lod: Some(shadow_lod_for_cascade(cascade_index)),
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
        handles: &[MeshHandle],
        meshes: &[ImportedMesh],
    ) -> Result<(), VulkanError> {
        let memory_properties = memory_properties(instance, physical_device);
        let mut total_vertices = 0_usize;
        let mut total_source_indices = 0_usize;
        let mut total_lod_buffers = 0_usize;
        let mut total_lod_indices = 0_usize;

        for (handle, mesh) in handles.iter().copied().zip(meshes.iter()) {
            let geometry = MeshGeometry::from_imported(mesh);
            let uploaded = VulkanMesh::upload(
                device,
                &memory_properties,
                queue_family_index,
                queue,
                &geometry,
            )?;
            total_vertices += geometry.vertex_count();
            total_source_indices += geometry.index_count();
            total_lod_buffers += uploaded.lods.len();
            total_lod_indices += uploaded.lod_index_count_sum();
            self.meshes.insert(handle, uploaded);
        }

        tracing::trace!(
            meshes = handles.len(),
            total_vertices,
            total_source_indices,
            total_lod_buffers,
            total_lod_indices,
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

    /// Creates moment-shadow mesh pipelines compatible with the shadow graph pass.
    pub(super) fn create_shadow_pipeline_set(
        &self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let pipelines = self.create_pipeline_set(
            device,
            render_pass,
            SHADOW_VERTEX_SHADER,
            SHADOW_FRAGMENT_SHADER,
            SHADOW_TEXTURED_FRAGMENT_SHADER,
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
            SHADOW_VERTEX_SHADER,
            SHADOW_DEPTH_FRAGMENT_SHADER,
            SHADOW_DEPTH_TEXTURED_FRAGMENT_SHADER,
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
            SHADOW_VERTEX_SHADER,
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
        })
    }

    /// Creates descriptors that let scene shaders sample the graph-owned shadow map.
    pub(super) fn create_pass_resources(
        &self,
        device: &Device,
        shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        raw_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
    ) -> Result<MeshPassResources, VulkanError> {
        MeshPassResources::create(
            device,
            self.pass_set_layout,
            shadow_views,
            raw_shadow_views,
            translucent_shadow_views,
            local_shadow_views,
        )
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

    /// Returns uploaded mesh bounds used by CPU-side emissive-light extraction.
    pub(super) fn bounds_for(&self, mesh: MeshHandle) -> Option<SceneBounds> {
        self.meshes.get(&mesh).and_then(|mesh| mesh.bounds)
    }

    /// Builds scene draw options while applying camera culling and LOD before sort/bind work.
    pub(super) fn scene_draw_options(
        &self,
        mesh: MeshHandle,
        extent: vk::Extent2D,
        camera: CameraSnapshot,
        optimization: RenderOptimizationSettings,
    ) -> Option<MeshDrawOptions> {
        let forced_lod = match self.meshes.get(&mesh) {
            Some(mesh) => {
                let aspect = if extent.height > 0 {
                    extent.width as f32 / extent.height as f32
                } else {
                    1.0
                };
                match classify_mesh(camera, aspect, extent.height, mesh.bounds, optimization) {
                    MeshVisibility::Visible { lod } => Some(lod),
                    MeshVisibility::Culled { .. } => return None,
                }
            }
            None => None,
        };

        Some(MeshDrawOptions::scene_preclassified(
            extent,
            camera,
            optimization,
            forced_lod,
        ))
    }

    /// Returns whether an uploaded mesh can affect the selected shadow cascade.
    ///
    /// Missing mesh handles stay accepted so `bind_and_draw` can emit the usual diagnostic path.
    pub(super) fn accepts_shadow_cascade(
        &self,
        mesh: MeshHandle,
        shadow_cull: ShadowCascadeCull,
    ) -> bool {
        self.meshes.get(&mesh).map_or(true, |mesh| {
            shadow_cascade_contains_bounds(mesh.bounds, shadow_cull)
        })
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
        let pipeline = pipeline_set.choose(pipeline_key.uses_textures, pipeline_key.double_sided);
        let pass_descriptor_set = pass_resources
            .map(MeshPassResources::descriptor_set)
            .unwrap_or(vk::DescriptorSet::null());

        let vertex_buffer = mesh.vertex_buffer.handle();
        let vertex_buffers = [vertex_buffer];
        let index_buffer = lod.index_buffer.handle();
        let offsets = [0_u64];

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
            if state.material_descriptor_set != material_descriptor_set {
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
            if state.vertex_buffer != vertex_buffer {
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                state.vertex_buffer = vertex_buffer;
            }
            if state.index_buffer != index_buffer {
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                state.index_buffer = index_buffer;
            }
            device.cmd_draw_indexed(command_buffer, lod.index_count, 1, 0, 0, 0);
        }

        Ok(true)
    }

    /// Destroys one swapchain-owned mesh pipeline pair.
    pub(super) fn destroy_pipeline_set(&self, device: &Device, pipeline_set: MeshPipelineSet) {
        destroy_pipeline_variants(device, pipeline_set.untextured);
        destroy_pipeline_variants(device, pipeline_set.textured);
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
    vertex_buffer: GpuBuffer,
    lods: Vec<VulkanMeshLod>,
    bounds: Option<SceneBounds>,
}

struct VulkanMeshLod {
    level: MeshLodLevel,
    index_buffer: GpuBuffer,
    index_count: u32,
}

impl MeshPipelineSet {
    /// Selects the shader variant that matches the material descriptor contract.
    fn choose(self, textured: bool, double_sided: bool) -> MeshPipeline {
        let variants = if textured {
            self.textured
        } else {
            self.untextured
        };
        if double_sided {
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
        shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        raw_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
        local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
    ) -> Result<Self, VulkanError> {
        let shadow_sampler = create_pass_sampler(device, vk::Filter::LINEAR)?;
        let transmittance_sampler = match create_pass_sampler(device, vk::Filter::LINEAR) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_sampler(device, shadow_sampler);
                return Err(error);
            }
        };
        let local_shadow_sampler = match create_pass_sampler(device, vk::Filter::NEAREST) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_sampler(device, transmittance_sampler);
                destroy_sampler(device, shadow_sampler);
                return Err(error);
            }
        };
        let descriptor_pool = match create_pass_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_sampler(device, local_shadow_sampler);
                destroy_sampler(device, transmittance_sampler);
                destroy_sampler(device, shadow_sampler);
                return Err(error);
            }
        };
        let descriptor_set =
            match allocate_pass_descriptor_set(device, descriptor_pool, pass_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    destroy_descriptor_pool(device, descriptor_pool);
                    destroy_sampler(device, local_shadow_sampler);
                    destroy_sampler(device, transmittance_sampler);
                    destroy_sampler(device, shadow_sampler);
                    return Err(error);
                }
            };

        update_pass_descriptor_set(
            device,
            descriptor_set,
            shadow_sampler,
            transmittance_sampler,
            local_shadow_sampler,
            shadow_views,
            raw_shadow_views,
            translucent_shadow_views,
            local_shadow_views,
        );
        tracing::info!("created Vulkan mesh pass descriptors");
        Ok(Self {
            descriptor_pool,
            descriptor_set,
            shadow_sampler,
            transmittance_sampler,
            local_shadow_sampler,
        })
    }

    /// Returns the descriptor set bound at `set = 2` for the scene mesh shaders.
    fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    /// Destroys scene-pass descriptor resources before graph target image views are released.
    pub(super) fn destroy(self, device: &Device) {
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.local_shadow_sampler);
        destroy_sampler(device, self.transmittance_sampler);
        destroy_sampler(device, self.shadow_sampler);
    }
}

impl VulkanMesh {
    /// Creates device-local vertex and LOD index buffers for one renderer mesh geometry.
    fn upload(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        geometry: &MeshGeometry,
    ) -> Result<Self, VulkanError> {
        let vertex_buffer = create_device_local_buffer_with_data(
            device,
            memory_properties,
            queue_family_index,
            queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            geometry.vertices(),
        )?;
        let lods = match upload_lod_buffers(
            device,
            memory_properties,
            queue_family_index,
            queue,
            geometry.vertices(),
            geometry.indices(),
        ) {
            Ok(lods) => lods,
            Err(error) => {
                vertex_buffer.destroy(device);
                return Err(error);
            }
        };

        Ok(Self {
            vertex_buffer,
            lods,
            bounds: geometry.bounds(),
        })
    }

    /// Returns the index count for the full-detail LOD.
    fn full_index_count(&self) -> u32 {
        self.lods.first().map_or(0, |lod| lod.index_count)
    }

    /// Returns total uploaded LOD indices for batch-level upload diagnostics.
    fn lod_index_count_sum(&self) -> usize {
        self.lods.iter().map(|lod| lod.index_count as usize).sum()
    }

    /// Returns the chosen LOD unless the mesh is culled by the active scene camera.
    fn visible_lod(
        &self,
        item: &RenderItemPacket,
        options: MeshDrawOptions,
    ) -> Option<&VulkanMeshLod> {
        if let Some(shadow_cull) = options.shadow_cull
            && !shadow_cascade_contains_bounds(self.bounds, shadow_cull)
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
        self.vertex_buffer.destroy(device);
    }
}

/// Selects cheaper geometry for shadow cascades whose map texels cannot show full mesh detail.
fn shadow_lod_for_cascade(cascade_index: usize) -> MeshLodLevel {
    if cascade_index >= SHADOW_CASCADE_COUNT {
        return MeshLodLevel::Medium;
    }
    match cascade_index {
        0 => MeshLodLevel::Full,
        1 => MeshLodLevel::Medium,
        _ => MeshLodLevel::Low,
    }
}

/// Returns whether mesh bounds overlap the camera-depth range covered by one shadow cascade.
///
/// Missing bounds stay visible so incomplete import metadata never drops a caster. The padding is
/// intentionally wide because directional shadows can reach into a neighboring cascade.
fn shadow_cascade_contains_bounds(
    bounds: Option<SceneBounds>,
    shadow_cull: ShadowCascadeCull,
) -> bool {
    let Some(bounds) = bounds else {
        return true;
    };
    let forward = normalize_or(
        sub3(shadow_cull.camera.target, shadow_cull.camera.eye),
        [0.0, 0.0, -1.0],
    );
    let light_dir = normalize_or(DEFAULT_DIRECTIONAL_LIGHT_DIR, [0.0, -1.0, 0.0]);
    if dot3(forward, light_dir).abs() > 0.82 {
        return true;
    }

    let to_center = sub3(bounds.center(), shadow_cull.camera.eye);
    let depth = dot3(to_center, forward);
    let radius = bounds.radius();
    let range = (shadow_cull.max_depth - shadow_cull.min_depth).max(1.0);
    if radius >= range * 0.75 {
        return true;
    }
    let padding = shadow_cascade_depth_padding(shadow_cull, radius);
    let min_depth = shadow_cull.min_depth - radius - padding;
    let max_depth = shadow_cull.max_depth + radius + padding;

    depth.is_finite() && depth >= min_depth && depth <= max_depth
}

/// Computes conservative depth padding for shadow-cascade caster culling.
///
/// Near cascades get a fixed safety margin, while wider cascades receive proportionally more room
/// so long shadows and large meshes are not clipped by the optimization.
fn shadow_cascade_depth_padding(shadow_cull: ShadowCascadeCull, radius: f32) -> f32 {
    let range = (shadow_cull.max_depth - shadow_cull.min_depth).max(1.0);
    (range * 0.45).clamp(12.0, 96.0) + radius * 2.0
}

/// Uploads full, medium, and low index buffers generated by geometric simplification.
fn upload_lod_buffers(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    queue: vk::Queue,
    vertices: &[MeshVertex],
    indices: &[u32],
) -> Result<Vec<VulkanMeshLod>, VulkanError> {
    let pending = unique_lod_indices(vertices, indices);
    let mut uploaded: Vec<VulkanMeshLod> = Vec::with_capacity(pending.len());

    for (level, lod_indices) in pending {
        let buffer = match create_device_local_buffer_with_data(
            device,
            memory_properties,
            queue_family_index,
            queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &lod_indices,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                for lod in uploaded {
                    lod.index_buffer.destroy(device);
                }
                return Err(error);
            }
        };

        uploaded.push(VulkanMeshLod {
            level,
            index_buffer: buffer,
            index_count: lod_indices.len() as u32,
        });
    }

    Ok(uploaded)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MeshFrameUniform {
    pub(super) view_proj: [f32; 16],
    pub(super) view: [f32; 16],
    pub(super) shadow_view_proj: [[f32; 16]; SHADOW_CASCADE_COUNT],
    pub(super) shadow_cascade_splits: [f32; 4],
    pub(super) shadow_cascade_texel_world: [f32; 4],
    pub(super) shadow_cascade_depth_span: [f32; 4],
    pub(super) camera_pos: [f32; 4],
    pub(super) light_dir: [f32; 4],
    pub(super) light_color: [f32; 4],
    pub(super) ambient_color: [f32; 4],
    pub(super) contact_shadow: [f32; 4],
    pub(super) local_shadow_view_proj: [[f32; 16]; LOCAL_SHADOW_MATRIX_COUNT],
    pub(super) local_shadow_params: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_position_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_color: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_direction_radius: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_size_kind: [[f32; 4]; MAX_LOCAL_LIGHTS],
    pub(super) emissive_light_count: [f32; 4],
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
    /// Returns an empty local-light payload for frames without emissive mesh lights.
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

/// Creates the frame descriptor set layout shared with `shaders/mesh.vert`.
fn create_frame_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(shader_interface::FRAME_CAMERA_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the pass descriptor set layout for graph-produced scene inputs.
fn create_pass_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
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
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(frame_count as u32);
    let pool_sizes = [pool_size];
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

/// Returns pass descriptor bindings in cascade order for filtered, translucent, raw, and local shadows.
fn pass_shadow_bindings() -> [u32; SHADOW_CASCADE_COUNT * 3 + 1] {
    std::array::from_fn(|index| {
        if index < SHADOW_CASCADE_COUNT {
            shader_interface::PASS_SHADOW_CASCADE_BINDINGS[index]
        } else if index < SHADOW_CASCADE_COUNT * 2 {
            shader_interface::PASS_TRANSLUCENT_SHADOW_BINDINGS[index - SHADOW_CASCADE_COUNT]
        } else if index < SHADOW_CASCADE_COUNT * 3 {
            shader_interface::PASS_RAW_SHADOW_CASCADE_BINDINGS[index - SHADOW_CASCADE_COUNT * 2]
        } else {
            shader_interface::PASS_LOCAL_SHADOW_BINDING
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

/// Returns the image views written into pass descriptors in binding order.
fn pass_shadow_views(
    shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    raw_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
) -> [vk::ImageView; SHADOW_CASCADE_COUNT * 3] {
    std::array::from_fn(|index| {
        if index < SHADOW_CASCADE_COUNT {
            shadow_views[index]
        } else if index < SHADOW_CASCADE_COUNT * 2 {
            translucent_shadow_views[index - SHADOW_CASCADE_COUNT]
        } else {
            raw_shadow_views[index - SHADOW_CASCADE_COUNT * 2]
        }
    })
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
    shadow_sampler: vk::Sampler,
    transmittance_sampler: vk::Sampler,
    local_shadow_sampler: vk::Sampler,
    shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    raw_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    translucent_shadow_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    local_shadow_views: [vk::ImageView; MAX_LOCAL_LIGHTS],
) {
    let views = pass_shadow_views(shadow_views, raw_shadow_views, translucent_shadow_views);
    let image_infos = views
        .into_iter()
        .enumerate()
        .map(|(index, view)| {
            let sampler = if index >= SHADOW_CASCADE_COUNT && index < SHADOW_CASCADE_COUNT * 2 {
                transmittance_sampler
            } else {
                shadow_sampler
            };
            vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();
    let local_image_infos = local_shadow_views
        .into_iter()
        .map(|view| {
            vk::DescriptorImageInfo::default()
                .sampler(local_shadow_sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();
    let writes = pass_shadow_bindings()
        .iter()
        .map(|&binding| {
            if binding == shader_interface::PASS_LOCAL_SHADOW_BINDING {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&local_image_infos)
            } else {
                let image_index = binding as usize;
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_infos[image_index]))
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
        shadow_cascade_splits: DEFAULT_SHADOW_CASCADE_SPLITS,
        shadow_cascade_texel_world: DEFAULT_SHADOW_CASCADE_METRICS,
        shadow_cascade_depth_span: DEFAULT_SHADOW_CASCADE_METRICS,
        camera_pos: [0.0, 0.0, 0.0, 1.0],
        light_dir: [light_dir[0], light_dir[1], light_dir[2], 0.0],
        light_color: [
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[0],
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[1],
            DEFAULT_DIRECTIONAL_LIGHT_COLOR[2],
            0.0,
        ],
        ambient_color: DEFAULT_AMBIENT_COLOR,
        contact_shadow: [0.0, 0.0, 0.0, 0.0],
        local_shadow_view_proj: [identity_mat4(); LOCAL_SHADOW_MATRIX_COUNT],
        local_shadow_params: [[0.0, 0.0, 1.0, 1.0]; MAX_LOCAL_LIGHTS],
        emissive_light_position_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_color: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_direction_radius: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_size_kind: [[0.0; 4]; MAX_LOCAL_LIGHTS],
        emissive_light_count: [0.0; 4],
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
}

impl MeshPipelineTarget {
    /// Returns how many color attachments the target writes.
    fn color_attachment_count(self) -> usize {
        match self {
            Self::SceneOpaque | Self::SceneTransparent => 3,
            Self::SceneOpaqueFast | Self::SceneTransparentFast => 1,
            Self::OpaqueShadow | Self::TranslucentShadow => 1,
            Self::LocalShadowDepth => 0,
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
            Self::OpaqueShadow => {
                vec![vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask)]
            }
            Self::LocalShadowDepth => Vec::new(),
            Self::TranslucentShadow => vec![
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::ZERO)
                    .dst_color_blend_factor(vk::BlendFactor::SRC_COLOR)
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
        culled: MeshPipeline { handle: culled },
        double_sided: MeshPipeline {
            handle: double_sided,
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
    let vertex_shader = create_shader_module(device, vertex_shader_bytes)?;
    let fragment_shader = match create_shader_module(device, fragment_shader_bytes) {
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
        vertex_layout,
        target,
        cull_mode,
    );

    destroy_shader_module(device, fragment_shader);
    destroy_shader_module(device, vertex_shader);
    pipeline
}

/// Creates one shader module from build-script compiled SPIR-V bytes.
fn create_shader_module(device: &Device, bytes: &[u8]) -> Result<vk::ShaderModule, VulkanError> {
    let code = util::read_spv(&mut Cursor::new(bytes)).map_err(VulkanError::ShaderCodeRead)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    // Safety: SPIR-V bytes are generated by `build.rs` and copied into a local word vector.
    unsafe { device.create_shader_module(&create_info, None) }.map_err(VulkanError::Vk)
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
    let vertex_bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<MeshVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
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
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
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
            .offset(offset_of!(MeshVertex, position) as u32),
    );
    if matches!(
        layout,
        MeshVertexLayout::SceneUntextured | MeshVertexLayout::SceneTextured
    ) {
        attributes.push(
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(MeshVertex, normal) as u32),
        );
    }

    match layout {
        MeshVertexLayout::SceneUntextured => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(4)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(offset_of!(MeshVertex, color) as u32),
            );
        }
        MeshVertexLayout::SceneTextured => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(MeshVertex, uv) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(3)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(offset_of!(MeshVertex, tangent) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(4)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(offset_of!(MeshVertex, color) as u32),
            );
        }
        MeshVertexLayout::Shadow => {
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(MeshVertex, uv) as u32),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(3)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(offset_of!(MeshVertex, color) as u32),
            );
        }
    }

    attributes
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

/// Destroys one temporary shader module.
fn destroy_shader_module(device: &Device, shader: vk::ShaderModule) {
    if shader == vk::ShaderModule::null() {
        return;
    }

    // Safety: pipeline creation has finished before temporary shader modules are destroyed.
    unsafe {
        device.destroy_shader_module(shader, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            shadow_lod_for_cascade(SHADOW_CASCADE_COUNT),
            MeshLodLevel::Medium
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

        assert!(shadow_cascade_contains_bounds(near_bounds, cull));
        assert!(!shadow_cascade_contains_bounds(far_bounds, cull));
        assert!(shadow_cascade_contains_bounds(large_bounds, cull));
        assert!(shadow_cascade_contains_bounds(None, cull));
    }

    #[test]
    fn transparent_scene_target_preserves_depth_and_writes_material_metadata() {
        let attachments = MeshPipelineTarget::SceneTransparent.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneTransparent.uses_depth_test());
        assert!(!MeshPipelineTarget::SceneTransparent.writes_depth());
        assert_eq!(attachments.len(), 3);
        assert!(attachments[0].blend_enable != 0);
        assert_eq!(
            attachments[1].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_ne!(
            attachments[2].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
    }

    #[test]
    fn opaque_scene_target_writes_depth_and_material_metadata() {
        let attachments = MeshPipelineTarget::SceneOpaque.color_blend_attachments();

        assert!(MeshPipelineTarget::SceneOpaque.uses_depth_test());
        assert!(MeshPipelineTarget::SceneOpaque.writes_depth());
        assert_eq!(attachments.len(), 3);
        assert!(attachments[0].blend_enable == 0);
        assert_ne!(
            attachments[1].color_write_mask,
            vk::ColorComponentFlags::empty()
        );
        assert_eq!(
            attachments[2].color_write_mask,
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
    fn untextured_scene_vertex_layout_skips_texture_attributes() {
        let locations = vertex_attributes_for_layout(MeshVertexLayout::SceneUntextured)
            .into_iter()
            .map(|attribute| attribute.location)
            .collect::<Vec<_>>();

        assert_eq!(locations, vec![0, 1, 4]);
    }

    #[test]
    fn textured_scene_vertex_layout_keeps_normal_mapping_attributes() {
        let locations = vertex_attributes_for_layout(MeshVertexLayout::SceneTextured)
            .into_iter()
            .map(|attribute| attribute.location)
            .collect::<Vec<_>>();

        assert_eq!(locations, vec![0, 1, 2, 3, 4]);
    }
}
