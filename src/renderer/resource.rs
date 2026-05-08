use std::{
    collections::HashMap,
    fs, io, mem,
    path::{Path, PathBuf},
};

use ash::{Instance, vk};

use super::{MAX_FRAMES_IN_FLIGHT, Material, MaterialId, MeshId, ModelId, ModelVertex};

#[derive(Debug, Clone)]
pub struct CpuMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn cube() -> Self {
        Self {
            vertices: CUBE_VERTICES.to_vec(),
            indices: CUBE_INDICES.to_vec(),
        }
    }

    pub fn load_obj(path: impl AsRef<Path>) -> io::Result<Self> {
        let model = CpuModel::load_obj(path)?;
        Ok(model
            .primitives
            .into_iter()
            .next()
            .map(|primitive| primitive.mesh)
            .unwrap_or_else(|| Self {
                vertices: Vec::new(),
                indices: Vec::new(),
            }))
    }

    pub fn from_obj_str(source: &str, path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let mut positions = Vec::<[f32; 3]>::new();
        let mut normals = Vec::<[f32; 3]>::new();
        let mut uvs = Vec::<[f32; 2]>::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for (line_index, line) in source.lines().enumerate() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.split_whitespace();
            let Some(tag) = parts.next() else { continue };

            match tag {
                "v" => positions.push(parse_vec3(parts, &path, line_index)?),
                "vn" => normals.push(parse_vec3(parts, &path, line_index)?),
                "vt" => {
                    let uv = parse_vec2(parts, &path, line_index)?;
                    uvs.push([uv[0], 1.0 - uv[1]]);
                }
                "f" => {
                    let face = parts
                        .map(|part| parse_face_vertex(part, &positions, &uvs, &normals))
                        .collect::<io::Result<Vec<_>>>()?;

                    if face.len() < 3 {
                        return invalid_obj(&path, line_index, "face has fewer than 3 vertices");
                    }

                    for i in 1..face.len() - 1 {
                        for vertex in [face[0], face[i], face[i + 1]] {
                            vertices.push(vertex);
                            indices.push((vertices.len() - 1) as u32);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Self { vertices, indices })
    }
}

#[derive(Debug, Clone)]
pub struct CpuPrimitive {
    pub mesh: CpuMesh,
    pub material: Material,
}

#[derive(Debug, Clone)]
pub struct CpuModel {
    pub primitives: Vec<CpuPrimitive>,
}

impl CpuModel {
    pub fn cube() -> Self {
        Self {
            primitives: vec![CpuPrimitive {
                mesh: CpuMesh::cube(),
                material: Material {
                    base_color: [0.18, 0.48, 1.0, 1.0],
                },
            }],
        }
    }

    pub fn load_obj(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        Self::from_obj_str(&fs::read_to_string(path)?, path)
    }

    pub fn from_obj_str(source: &str, path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let base_dir = path.parent().unwrap_or(Path::new("."));
        let mut positions = Vec::<[f32; 3]>::new();
        let mut normals = Vec::<[f32; 3]>::new();
        let mut uvs = Vec::<[f32; 2]>::new();
        let mut materials = HashMap::<String, Material>::new();
        let mut builders = HashMap::<String, MeshBuilder>::new();
        let mut current = String::from("default");
        materials.insert(current.clone(), Material::default());

        for (line_index, line) in source.lines().enumerate() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.split_whitespace();
            let Some(tag) = parts.next() else { continue };

            match tag {
                "mtllib" => {
                    for name in parts {
                        load_mtl(base_dir.join(name), &mut materials)?;
                    }
                }
                "usemtl" => current = parts.next().unwrap_or("default").to_string(),
                "v" => positions.push(parse_vec3(parts, &path, line_index)?),
                "vn" => normals.push(parse_vec3(parts, &path, line_index)?),
                "vt" => {
                    let uv = parse_vec2(parts, &path, line_index)?;
                    uvs.push([uv[0], 1.0 - uv[1]]);
                }
                "f" => {
                    let face = parts
                        .map(|part| parse_face_vertex(part, &positions, &uvs, &normals))
                        .collect::<io::Result<Vec<_>>>()?;

                    if face.len() < 3 {
                        return invalid_obj(&path, line_index, "face has fewer than 3 vertices");
                    }

                    let builder = builders.entry(current.clone()).or_default();
                    for i in 1..face.len() - 1 {
                        builder.triangle(face[0], face[i], face[i + 1]);
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            primitives: builders
                .into_iter()
                .filter_map(|(name, builder)| {
                    (!builder.vertices.is_empty()).then(|| CpuPrimitive {
                        mesh: builder.finish(),
                        material: materials.get(&name).copied().unwrap_or_default(),
                    })
                })
                .collect(),
        })
    }
}

#[derive(Default)]
struct MeshBuilder {
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn triangle(&mut self, a: ModelVertex, b: ModelVertex, c: ModelVertex) {
        for vertex in [a, b, c] {
            self.vertices.push(vertex);
            self.indices.push((self.vertices.len() - 1) as u32);
        }
    }

    fn finish(self) -> CpuMesh {
        CpuMesh {
            vertices: self.vertices,
            indices: self.indices,
        }
    }
}

pub(super) struct MeshBuffers {
    pub vertex: GpuBuffer,
    pub index: GpuBuffer,
    pub index_count: u32,
}

impl MeshBuffers {
    pub fn from_mesh(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: &CpuMesh,
    ) -> Self {
        Self {
            vertex: GpuBuffer::device_local(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &mesh.vertices,
            ),
            index: GpuBuffer::device_local(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.vertex.destroy(device);
            self.index.destroy(device);
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct GpuPrimitive {
    pub mesh: MeshId,
    pub material: MaterialId,
}

#[derive(Debug, Clone)]
pub(super) struct GpuModel {
    pub primitives: Vec<GpuPrimitive>,
}

pub(super) struct GpuAssets {
    meshes: Vec<MeshBuffers>,
    materials: Vec<Material>,
    models: Vec<GpuModel>,
}

impl GpuAssets {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Self {
        let mut assets = Self {
            meshes: Vec::new(),
            materials: vec![Material::default()],
            models: Vec::new(),
        };
        assets.upload_model(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuModel::cube(),
        );
        assets
    }

    pub fn upload_mesh(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: &CpuMesh,
    ) -> MeshId {
        self.meshes.push(MeshBuffers::from_mesh(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            mesh,
        ));
        MeshId(self.meshes.len() - 1)
    }

    pub fn upload_material(&mut self, material: Material) -> MaterialId {
        self.materials.push(material);
        MaterialId(self.materials.len() - 1)
    }

    pub fn upload_model(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        model: &CpuModel,
    ) -> ModelId {
        let primitives = model
            .primitives
            .iter()
            .map(|primitive| GpuPrimitive {
                mesh: self.upload_mesh(
                    instance,
                    device,
                    physical_device,
                    command_pool,
                    queue,
                    &primitive.mesh,
                ),
                material: self.upload_material(primitive.material),
            })
            .collect();

        self.models.push(GpuModel { primitives });
        ModelId(self.models.len() - 1)
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshBuffers> {
        self.meshes.get(id.0)
    }

    pub fn material(&self, id: MaterialId) -> Material {
        self.materials.get(id.0).copied().unwrap_or_default()
    }

    pub fn model(&self, id: ModelId) -> Option<&GpuModel> {
        self.models.get(id.0)
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for mesh in &mut self.meshes {
            unsafe { mesh.destroy(device) };
        }
    }
}

pub(super) struct GpuBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl GpuBuffer {
    pub fn host_uniform<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        value: &T,
    ) -> Self {
        let mut buffer = Self::new(
            instance,
            device,
            physical_device,
            mem::size_of::<T>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { buffer.write(device, value) };
        buffer
    }

    pub fn device_local<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Self {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        let mut staging = Self::new(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { staging.write_slice(device, data) };

        let buffer = Self::new(
            instance,
            device,
            physical_device,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        copy_buffer(
            device,
            command_pool,
            queue,
            staging.buffer,
            buffer.buffer,
            size,
        );

        unsafe { staging.destroy(device) };
        buffer
    }

    fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&info, None)
                .expect("renderer: failed to create buffer")
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            properties,
        );

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("renderer: failed to allocate buffer memory")
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("renderer: failed to bind buffer memory")
        };

        Self { buffer, memory }
    }

    pub unsafe fn write<T: Copy>(&mut self, device: &ash::Device, data: &T) {
        let size = mem::size_of::<T>() as vk::DeviceSize;
        let mapped = unsafe {
            device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("renderer: failed to map buffer memory")
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                (data as *const T).cast::<u8>(),
                mapped.cast::<u8>(),
                size as usize,
            );
            device.unmap_memory(self.memory);
        }
    }

    unsafe fn write_slice<T: Copy>(&mut self, device: &ash::Device, data: &[T]) {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        let mapped = unsafe {
            device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("renderer: failed to map buffer memory")
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                size as usize,
            );
            device.unmap_memory(self.memory);
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe { device.destroy_buffer(self.buffer, None) };
            self.buffer = vk::Buffer::null();
        }

        if self.memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
        }
    }
}

pub(super) struct SceneBindings {
    pub layout: vk::DescriptorSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
    pool: vk::DescriptorPool,
    buffers: Vec<GpuBuffer>,
    range: vk::DeviceSize,
}

impl SceneBindings {
    pub fn new<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        initial: &T,
    ) -> Self {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("renderer: failed to create scene descriptor layout")
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        };
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("renderer: failed to create scene descriptor pool")
        };

        let layouts = vec![layout; MAX_FRAMES_IN_FLIGHT];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc)
                .expect("renderer: failed to allocate scene descriptor sets")
        };

        let range = mem::size_of::<T>() as vk::DeviceSize;
        let buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| GpuBuffer::host_uniform(instance, device, physical_device, initial))
            .collect::<Vec<_>>();

        for (set, buffer) in sets.iter().zip(&buffers) {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .range(range);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        Self {
            layout,
            sets,
            pool,
            buffers,
            range,
        }
    }

    pub fn update<T: Copy>(&mut self, device: &ash::Device, frame_index: usize, value: &T) {
        debug_assert_eq!(self.range, mem::size_of::<T>() as vk::DeviceSize);
        unsafe { self.buffers[frame_index].write(device, value) };
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for buffer in &mut self.buffers {
            unsafe { buffer.destroy(device) };
        }

        if self.pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.pool, None) };
            self.pool = vk::DescriptorPool::null();
        }

        if self.layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
            self.layout = vk::DescriptorSetLayout::null();
        }
    }
}

pub(super) struct DepthTarget {
    pub view: vk::ImageView,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl DepthTarget {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("renderer: failed to create depth image")
        };

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("renderer: failed to allocate depth memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("renderer: failed to bind depth memory")
        };

        transition_depth_image(device, command_pool, queue, image);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("renderer: failed to create depth view")
        };

        Self {
            view,
            image,
            memory,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }

        if self.image != vk::Image::null() {
            unsafe { device.destroy_image(self.image, None) };
            self.image = vk::Image::null();
        }

        if self.memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
        }
    }
}

fn copy_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let region = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(command_buffer, src, dst, std::slice::from_ref(&region)) };
    });
}

fn transition_depth_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
    });
}

fn submit_once<F: FnOnce(vk::CommandBuffer)>(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    record: F,
) {
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc)
            .expect("renderer: failed to allocate transient command buffer")[0]
    };

    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin)
            .expect("renderer: failed to begin transient command buffer")
    };

    record(command_buffer);

    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("renderer: failed to end transient command buffer")
    };

    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

    unsafe {
        device
            .queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())
            .expect("renderer: failed to submit transient command buffer");
        device
            .queue_wait_idle(queue)
            .expect("renderer: failed to wait transient command buffer");
        device.free_command_buffers(command_pool, &command_buffers);
    }
}

fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    (0..memory.memory_type_count)
        .find(|&index| {
            let supported = (type_filter & (1_u32 << index)) != 0;
            let flags = memory.memory_types[index as usize].property_flags;
            supported && flags.contains(properties)
        })
        .expect("renderer: failed to find suitable memory type")
}

fn load_mtl(path: PathBuf, materials: &mut HashMap<String, Material>) -> io::Result<()> {
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    let mut current = None::<String>;

    for line in source.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else { continue };

        match tag {
            "newmtl" => {
                let name = parts.next().unwrap_or("default").to_string();
                materials.entry(name.clone()).or_default();
                current = Some(name);
            }
            "Kd" => {
                if let Some(material) = current.as_ref().and_then(|name| materials.get_mut(name)) {
                    material.base_color[0] =
                        parts.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    material.base_color[1] =
                        parts.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    material.base_color[2] =
                        parts.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                }
            }
            "d" => {
                if let Some(material) = current.as_ref().and_then(|name| materials.get_mut(name)) {
                    material.base_color[3] =
                        parts.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                }
            }
            "Tr" => {
                if let Some(material) = current.as_ref().and_then(|name| materials.get_mut(name)) {
                    material.base_color[3] =
                        1.0 - parts.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn parse_vec3<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    path: &Path,
    line: usize,
) -> io::Result<[f32; 3]> {
    Ok([
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
    ])
}

fn parse_vec2<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    path: &Path,
    line: usize,
) -> io::Result<[f32; 2]> {
    Ok([
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
    ])
}

fn parse_f32(value: Option<&str>, path: &Path, line: usize) -> io::Result<f32> {
    value
        .ok_or_else(|| obj_error(path, line, "missing float"))?
        .parse()
        .map_err(|_| obj_error(path, line, "invalid float"))
}

fn parse_face_vertex(
    part: &str,
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
    normals: &[[f32; 3]],
) -> io::Result<ModelVertex> {
    let mut ids = part.split('/');
    let position = positions[obj_index(ids.next(), positions.len())?];
    let uv = ids
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| obj_index(Some(value), uvs.len()).map(|index| uvs[index]))
        .transpose()?
        .unwrap_or([0.0; 2]);
    let normal = ids
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| obj_index(Some(value), normals.len()).map(|index| normals[index]))
        .transpose()?
        .unwrap_or([0.0, 1.0, 0.0]);

    Ok(ModelVertex {
        position,
        normal,
        uv,
    })
}

fn obj_index(value: Option<&str>, len: usize) -> io::Result<usize> {
    let index = value
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing obj index"))?
        .parse::<isize>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid obj index"))?;

    let resolved = if index > 0 {
        index - 1
    } else if index < 0 {
        len as isize + index
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "obj indices are 1-based",
        ));
    };

    if resolved < 0 || resolved >= len as isize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "obj index out of bounds",
        ));
    }

    Ok(resolved as usize)
}

fn invalid_obj<T>(path: &Path, line: usize, message: &'static str) -> io::Result<T> {
    Err(obj_error(path, line, message))
}

fn obj_error(path: &Path, line: usize, message: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}:{}: {message}", path.display(), line + 1),
    )
}

const S: f32 = 0.75;

const CUBE_VERTICES: [ModelVertex; 24] = [
    ModelVertex {
        position: [-S, -S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [-S, -S, S],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, S],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 0.0],
    },
];

const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17, 18,
    18, 19, 16, 20, 21, 22, 22, 23, 20,
];
