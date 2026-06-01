use std::{
    collections::BTreeMap,
    ffi::CStr,
    io::Cursor,
    mem::{offset_of, size_of},
};

use ash::{Device, Instance, util, vk};

use crate::{
    import::ImportedMesh,
    protocol::{AssetHandle, MeshHandle, RenderItemPacket},
    renderer::assets::{MeshGeometry, MeshVertex},
    renderer::pipeline::shader_interface,
};

use super::{
    VulkanError,
    buffer::{
        GpuBuffer, create_buffer_with_data, destroy_buffers, memory_properties, write_buffer_value,
    },
    material::VulkanMaterialStore,
};

const SHADER_ENTRY: &CStr = c"main";
const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.vert.spv"));
const SCENE_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_scene.frag.spv"));
const SCENE_TEXTURED_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh_scene_textured.frag.spv"));
const SHADOW_VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
const SHADOW_FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));
const SHADOW_TEXTURED_FRAGMENT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/shadow_textured.frag.spv"));
const DEFAULT_AMBIENT_COLOR: [f32; 4] = [0.014, 0.016, 0.020, 1.0];

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
pub(super) struct MeshPipelineSet {
    untextured: MeshPipeline,
    textured: MeshPipeline,
}

pub(super) struct MeshPassResources {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
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
        handles: &[MeshHandle],
        meshes: &[ImportedMesh],
    ) -> Result<(), VulkanError> {
        let memory_properties = memory_properties(instance, physical_device);
        for (handle, mesh) in handles.iter().copied().zip(meshes.iter()) {
            let geometry = MeshGeometry::from_imported(mesh);
            let uploaded = VulkanMesh::upload(device, &memory_properties, &geometry)?;
            tracing::trace!(
                mesh = handle.raw(),
                vertices = geometry.vertex_count(),
                indices = geometry.index_count(),
                "uploaded Vulkan mesh buffers"
            );
            self.meshes.insert(handle, uploaded);
        }

        Ok(())
    }

    /// Creates scene-pass mesh pipelines that can sample graph-produced targets.
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
            true,
        )?;
        tracing::info!("created Vulkan scene mesh pipelines");
        Ok(pipelines)
    }

    /// Creates depth-only mesh pipelines compatible with the shadow graph pass.
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
            false,
        )?;
        tracing::info!("created Vulkan shadow mesh pipelines");
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
        color_output: bool,
    ) -> Result<MeshPipelineSet, VulkanError> {
        let untextured = create_mesh_pipeline(
            device,
            self.pipeline_layout,
            render_pass,
            vertex_shader,
            untextured_fragment,
            color_output,
        )?;
        let textured = match create_mesh_pipeline(
            device,
            self.pipeline_layout,
            render_pass,
            vertex_shader,
            textured_fragment,
            color_output,
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                destroy_pipeline(device, untextured);
                return Err(error);
            }
        };

        Ok(MeshPipelineSet {
            untextured: MeshPipeline { handle: untextured },
            textured: MeshPipeline { handle: textured },
        })
    }

    /// Creates descriptors that let scene shaders sample the graph-owned shadow map.
    pub(super) fn create_pass_resources(
        &self,
        device: &Device,
        shadow_map_view: vk::ImageView,
    ) -> Result<MeshPassResources, VulkanError> {
        MeshPassResources::create(device, self.pass_set_layout, shadow_map_view)
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

    /// Binds the mesh pipeline and records one indexed mesh draw if the handle is live.
    pub(super) fn bind_and_draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        pipeline_set: MeshPipelineSet,
        materials: &VulkanMaterialStore,
        pass_resources: Option<&MeshPassResources>,
        frame_slot: usize,
        item: &RenderItemPacket,
        extent: vk::Extent2D,
    ) -> Result<(), VulkanError> {
        if !item.flags.visible {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                "mesh draw skipped because the item is not visible"
            );
            return Ok(());
        }

        let Some(mesh) = self.meshes.get(&item.mesh) else {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                "mesh draw skipped because the Vulkan mesh is missing"
            );
            return Ok(());
        };
        let Some(material_descriptor_set) = materials.descriptor_set_for(item.material) else {
            tracing::trace!(
                mesh = item.mesh.raw(),
                material = item.material.raw(),
                "mesh draw skipped because the Vulkan material is missing"
            );
            return Ok(());
        };
        let frame_descriptor_set = self.frame_descriptor_sets.get(frame_slot).copied().ok_or(
            VulkanError::FrameSlotIndexOutOfRange {
                index: frame_slot,
                count: self.frame_descriptor_sets.len(),
            },
        )?;
        let pipeline = pipeline_set.choose(materials.has_base_color_texture(item.material));

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
        let vertex_buffers = [mesh.vertex_buffer.handle()];
        let offsets = [0_u64];

        tracing::trace!(
            mesh = item.mesh.raw(),
            material = item.material.raw(),
            indices = mesh.index_count,
            width = extent.width,
            height = extent.height,
            "recording Vulkan mesh draw"
        );

        // Safety: the command buffer is recording inside a compatible render pass. The pipeline
        // was created for that pass, and mesh buffers are owned by the renderer until frame end.
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                shader_interface::FRAME_SET,
                &[frame_descriptor_set],
                &[],
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                shader_interface::MATERIAL_SET,
                &[material_descriptor_set],
                &[],
            );
            if let Some(pass_resources) = pass_resources {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    shader_interface::PASS_SET,
                    &[pass_resources.descriptor_set()],
                    &[],
                );
            }
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(
                command_buffer,
                mesh.index_buffer.handle(),
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
        }

        Ok(())
    }

    /// Destroys one swapchain-owned mesh pipeline pair.
    pub(super) fn destroy_pipeline_set(&self, device: &Device, pipeline_set: MeshPipelineSet) {
        destroy_pipeline(device, pipeline_set.untextured.handle);
        destroy_pipeline(device, pipeline_set.textured.handle);
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
                indices = uploaded.index_count,
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
    index_buffer: GpuBuffer,
    index_count: u32,
}

impl MeshPipelineSet {
    /// Selects the shader variant that matches the material descriptor contract.
    fn choose(self, textured: bool) -> MeshPipeline {
        if textured {
            self.textured
        } else {
            self.untextured
        }
    }
}

impl MeshPassResources {
    /// Creates one descriptor set for scene-pass sampled graph resources.
    fn create(
        device: &Device,
        pass_set_layout: vk::DescriptorSetLayout,
        shadow_map_view: vk::ImageView,
    ) -> Result<Self, VulkanError> {
        let sampler = create_pass_sampler(device)?;
        let descriptor_pool = match create_pass_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_sampler(device, sampler);
                return Err(error);
            }
        };
        let descriptor_set =
            match allocate_pass_descriptor_set(device, descriptor_pool, pass_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    destroy_descriptor_pool(device, descriptor_pool);
                    destroy_sampler(device, sampler);
                    return Err(error);
                }
            };

        update_pass_descriptor_set(device, descriptor_set, sampler, shadow_map_view);
        tracing::info!("created Vulkan mesh pass descriptors");
        Ok(Self {
            descriptor_pool,
            descriptor_set,
            sampler,
        })
    }

    /// Returns the descriptor set bound at `set = 2` for the scene mesh shaders.
    fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    /// Destroys scene-pass descriptor resources before graph target image views are released.
    pub(super) fn destroy(self, device: &Device) {
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.sampler);
    }
}

impl VulkanMesh {
    /// Creates host-visible vertex and index buffers for one renderer mesh geometry.
    fn upload(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        geometry: &MeshGeometry,
    ) -> Result<Self, VulkanError> {
        let vertex_buffer = create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            geometry.vertices(),
        )?;
        let index_buffer = match create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::INDEX_BUFFER,
            geometry.indices(),
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                vertex_buffer.destroy(device);
                return Err(error);
            }
        };

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: geometry.index_count() as u32,
        })
    }

    /// Destroys the uploaded vertex and index buffers for one mesh.
    fn destroy(self, device: &Device) {
        self.index_buffer.destroy(device);
        self.vertex_buffer.destroy(device);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MeshFrameUniform {
    pub(super) view_proj: [f32; 16],
    pub(super) shadow_view_proj: [f32; 16],
    pub(super) camera_pos: [f32; 4],
    pub(super) light_dir: [f32; 4],
    pub(super) light_color: [f32; 4],
    pub(super) ambient_color: [f32; 4],
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
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(shader_interface::PASS_SHADOW_MAP_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
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
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

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
fn create_pass_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
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

/// Creates the descriptor pool for the scene mesh pass resource set.
fn create_pass_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
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

/// Writes the shadow-map view into the scene mesh pass descriptor set.
fn update_pass_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    shadow_map_view: vk::ImageView,
) {
    let image_info = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(shadow_map_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(shader_interface::PASS_SHADOW_MAP_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info)];

    // Safety: descriptor set, sampler, and image views belong to this device and remain alive.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Returns a stable initial uniform before the first extracted frame writes camera data.
fn identity_frame_uniform() -> MeshFrameUniform {
    MeshFrameUniform {
        view_proj: identity_mat4(),
        shadow_view_proj: identity_mat4(),
        camera_pos: [0.0, 0.0, 0.0, 1.0],
        light_dir: [0.4, -1.0, 0.3, 0.0],
        light_color: [1.0, 0.97, 0.9, 1.0],
        ambient_color: DEFAULT_AMBIENT_COLOR,
    }
}

/// Returns an identity matrix for mesh uniforms before the first frame writes a camera.
fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Creates the graphics pipeline that renders indexed mesh packets into the swapchain pass.
fn create_mesh_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader_bytes: &[u8],
    fragment_shader_bytes: &[u8],
    color_output: bool,
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
        color_output,
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
    color_output: bool,
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
    let vertex_attributes = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(MeshVertex, position) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(MeshVertex, normal) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(MeshVertex, uv) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(offset_of!(MeshVertex, color) as u32),
    ];
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
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let color_blend_attachments = if color_output {
        vec![
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                ),
        ]
    } else {
        Vec::new()
    };
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
