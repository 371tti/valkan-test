use std::mem;

use ash::{Instance, vk};

use crate::renderer::{MAX_FRAMES_IN_FLIGHT, Material, MaterialId, MeshId, ModelId, TextureId};

use super::cpu::{CpuMesh, CpuModel, CpuTexture, TextureFilter, TextureSampler, TextureWrap};

fn mesh_center(mesh: &CpuMesh) -> [f32; 3] {
    let Some(first) = mesh.vertices.first() else {
        return [0.0; 3];
    };
    let (min, max) = mesh.vertices.iter().fold(
        (first.position, first.position),
        |(mut min, mut max), vertex| {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }

            (min, max)
        },
    );

    [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ]
}

fn mesh_radius(mesh: &CpuMesh) -> f32 {
    let center = mesh_center(mesh);

    mesh.vertices
        .iter()
        .map(|vertex| {
            let dx = vertex.position[0] - center[0];
            let dy = vertex.position[1] - center[1];
            let dz = vertex.position[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f32::max)
}

pub(in crate::renderer) struct MeshBuffers {
    pub vertex: GpuBuffer,
    pub index: GpuBuffer,
    pub index_count: u32,
    pub center: [f32; 3],
    pub radius: f32,
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
            center: mesh_center(mesh),
            radius: mesh_radius(mesh),
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
pub(in crate::renderer) struct GpuPrimitive {
    pub mesh: MeshId,
    pub material: MaterialId,
}

#[derive(Debug, Clone)]
pub(in crate::renderer) struct GpuModel {
    pub primitives: Vec<GpuPrimitive>,
}

pub(in crate::renderer) struct GpuTexture {
    view: vk::ImageView,
    image: vk::Image,
    memory: vk::DeviceMemory,
    sampler: vk::Sampler,
}

impl GpuTexture {
    fn from_texture(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture: &CpuTexture,
    ) -> Self {
        let size = texture.pixels.len() as vk::DeviceSize;
        let mut staging = GpuBuffer::new(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { staging.write_slice(device, &texture.pixels) };

        let (image, memory) = create_texture_image(
            instance,
            device,
            physical_device,
            texture.width,
            texture.height,
            texture.srgb,
        );
        transition_texture_image(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging.buffer,
            image,
            texture.width,
            texture.height,
        );
        transition_texture_image(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe { staging.destroy(device) };

        let view = create_texture_view(device, image, texture.srgb);
        let sampler = create_texture_sampler(device, texture.sampler);

        Self {
            view,
            image,
            memory,
            sampler,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.sampler != vk::Sampler::null() {
            unsafe { device.destroy_sampler(self.sampler, None) };
            self.sampler = vk::Sampler::null();
        }

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

struct GpuMaterial {
    material: Material,
    texture_set: vk::DescriptorSet,
}

pub(in crate::renderer) struct GpuAssets {
    meshes: Vec<MeshBuffers>,
    materials: Vec<GpuMaterial>,
    textures: Vec<GpuTexture>,
    models: Vec<GpuModel>,
    texture_layout: vk::DescriptorSetLayout,
    texture_pool: vk::DescriptorPool,
}

impl GpuAssets {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Self {
        let texture_layout = create_texture_set_layout(device);
        let texture_pool = create_texture_descriptor_pool(device);
        let mut assets = Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            models: Vec::new(),
            texture_layout,
            texture_pool,
        };
        assets.upload_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuTexture::white(),
        );
        assets.upload_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuTexture::flat_normal(),
        );
        assets.upload_material(device, Material::default());
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

    pub fn upload_material(&mut self, device: &ash::Device, material: Material) -> MaterialId {
        let texture_set = allocate_texture_set(
            device,
            self.texture_layout,
            self.texture_pool,
            &self.textures,
            material,
        );
        self.materials.push(GpuMaterial {
            material,
            texture_set,
        });
        MaterialId(self.materials.len() - 1)
    }

    pub fn upload_texture(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture: &CpuTexture,
    ) -> TextureId {
        self.textures.push(GpuTexture::from_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            texture,
        ));
        TextureId(self.textures.len() - 1)
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
        let textures = model
            .textures
            .iter()
            .map(|texture| {
                self.upload_texture(
                    instance,
                    device,
                    physical_device,
                    command_pool,
                    queue,
                    texture,
                )
            })
            .collect::<Vec<_>>();

        let primitives = model
            .primitives
            .iter()
            .map(|primitive| {
                let mut material = primitive.material;
                material.base_color_texture = material
                    .base_color_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.metallic_roughness_texture = material
                    .metallic_roughness_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.normal_texture = material
                    .normal_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.occlusion_texture = material
                    .occlusion_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.emissive_texture = material
                    .emissive_texture
                    .and_then(|texture| textures.get(texture.0).copied());

                GpuPrimitive {
                    mesh: self.upload_mesh(
                        instance,
                        device,
                        physical_device,
                        command_pool,
                        queue,
                        &primitive.mesh,
                    ),
                    material: self.upload_material(device, material),
                }
            })
            .collect();

        self.models.push(GpuModel { primitives });
        ModelId(self.models.len() - 1)
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshBuffers> {
        self.meshes.get(id.0)
    }

    pub fn material(&self, id: MaterialId) -> Material {
        self.materials
            .get(id.0)
            .map(|material| material.material)
            .unwrap_or_default()
    }

    pub fn model(&self, id: ModelId) -> Option<&GpuModel> {
        self.models.get(id.0)
    }

    pub fn material_texture_set(&self, id: MaterialId) -> Option<vk::DescriptorSet> {
        self.materials
            .get(id.0)
            .map(|material| material.texture_set)
    }

    pub fn texture_set_layout(&self) -> vk::DescriptorSetLayout {
        self.texture_layout
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for mesh in &mut self.meshes {
            unsafe { mesh.destroy(device) };
        }

        for texture in &mut self.textures {
            unsafe { texture.destroy(device) };
        }

        if self.texture_pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.texture_pool, None) };
            self.texture_pool = vk::DescriptorPool::null();
        }

        if self.texture_layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.texture_layout, None) };
            self.texture_layout = vk::DescriptorSetLayout::null();
        }
    }
}

pub(in crate::renderer) struct GpuBuffer {
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

pub(in crate::renderer) struct SceneBindings {
    pub layout: vk::DescriptorSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
    pool: vk::DescriptorPool,
    buffers: Vec<GpuBuffer>,
    range: vk::DeviceSize,
    owns_layout: bool,
}

impl SceneBindings {
    pub fn new<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        initial: &T,
        cube_reflection: vk::DescriptorImageInfo,
        planar_reflection: vk::DescriptorImageInfo,
    ) -> Self {
        let layout = create_scene_set_layout(device);
        Self::with_layout(
            instance,
            device,
            physical_device,
            initial,
            cube_reflection,
            planar_reflection,
            layout,
            MAX_FRAMES_IN_FLIGHT,
            true,
        )
    }

    pub fn with_layout<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        initial: &T,
        cube_reflection: vk::DescriptorImageInfo,
        planar_reflection: vk::DescriptorImageInfo,
        layout: vk::DescriptorSetLayout,
        count: usize,
        owns_layout: bool,
    ) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (count * 2) as u32,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count as u32)
            .pool_sizes(&pool_sizes);

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("renderer: failed to create scene descriptor pool")
        };

        let layouts = vec![layout; count];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc)
                .expect("renderer: failed to allocate scene descriptor sets")
        };

        let range = mem::size_of::<T>() as vk::DeviceSize;
        let buffers = (0..count)
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
            let reflection_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&cube_reflection));
            let planar_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&planar_reflection));

            unsafe { device.update_descriptor_sets(&[write, reflection_write, planar_write], &[]) };
        }

        Self {
            layout,
            sets,
            pool,
            buffers,
            range,
            owns_layout,
        }
    }

    pub fn update<T: Copy>(&mut self, device: &ash::Device, frame_index: usize, value: &T) {
        debug_assert_eq!(self.range, mem::size_of::<T>() as vk::DeviceSize);
        unsafe { self.buffers[frame_index].write(device, value) };
    }

    pub fn update_reflections(
        &self,
        device: &ash::Device,
        cube_reflection: vk::DescriptorImageInfo,
        planar_reflection: vk::DescriptorImageInfo,
    ) {
        let mut writes = Vec::with_capacity(self.sets.len() * 2);
        for set in &self.sets {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&cube_reflection)),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&planar_reflection)),
            );
        }

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for buffer in &mut self.buffers {
            unsafe { buffer.destroy(device) };
        }

        if self.pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.pool, None) };
            self.pool = vk::DescriptorPool::null();
        }

        if self.owns_layout && self.layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
            self.layout = vk::DescriptorSetLayout::null();
        }
    }
}

fn create_scene_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("renderer: failed to create scene descriptor layout")
    }
}

pub(in crate::renderer) struct DepthTarget {
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

pub(in crate::renderer) struct ReflectionProbe {
    pub view: vk::ImageView,
    pub face_views: Vec<vk::ImageView>,
    pub sampler: vk::Sampler,
    pub depth: DepthTarget,
    pub extent: vk::Extent2D,
    image: vk::Image,
    memory: vk::DeviceMemory,
    layout: vk::ImageLayout,
}

impl ReflectionProbe {
    pub const FACE_COUNT: usize = 6;

    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        format: vk::Format,
        size: u32,
    ) -> Self {
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };
        let image_info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(Self::FACE_COUNT as u32)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("renderer: failed to create reflection probe image")
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
                .expect("renderer: failed to allocate reflection probe memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("renderer: failed to bind reflection probe memory")
        };

        let view = create_reflection_probe_view(
            device,
            image,
            format,
            vk::ImageViewType::CUBE,
            0,
            Self::FACE_COUNT as u32,
        );
        let face_views = (0..Self::FACE_COUNT)
            .map(|face| {
                create_reflection_probe_view(
                    device,
                    image,
                    format,
                    vk::ImageViewType::TYPE_2D,
                    face as u32,
                    1,
                )
            })
            .collect();
        let sampler = create_texture_sampler(
            device,
            TextureSampler {
                mag_filter: TextureFilter::Linear,
                min_filter: TextureFilter::Linear,
                wrap_s: TextureWrap::ClampToEdge,
                wrap_t: TextureWrap::ClampToEdge,
            },
        );
        let depth = DepthTarget::new(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            extent,
            vk::Format::D32_SFLOAT,
        );
        initialize_reflection_probe_image(device, command_pool, queue, image);

        Self {
            view,
            face_views,
            sampler,
            depth,
            extent,
            image,
            memory,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }

    pub fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout
    }

    pub fn set_layout(&mut self, layout: vk::ImageLayout) {
        self.layout = layout;
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe { self.depth.destroy(device) };

        for view in self.face_views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }

        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }

        if self.sampler != vk::Sampler::null() {
            unsafe { device.destroy_sampler(self.sampler, None) };
            self.sampler = vk::Sampler::null();
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

pub(in crate::renderer) struct PlanarReflectionTarget {
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub depth: DepthTarget,
    pub extent: vk::Extent2D,
    image: vk::Image,
    memory: vk::DeviceMemory,
    layout: vk::ImageLayout,
}

impl PlanarReflectionTarget {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let extent = vk::Extent2D {
            width: extent.width.max(1),
            height: extent.height.max(1),
        };
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
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("renderer: failed to create planar reflection image")
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
                .expect("renderer: failed to allocate planar reflection memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("renderer: failed to bind planar reflection memory")
        };

        let view = create_planar_reflection_view(device, image, format);
        let sampler = create_texture_sampler(
            device,
            TextureSampler {
                mag_filter: TextureFilter::Linear,
                min_filter: TextureFilter::Linear,
                wrap_s: TextureWrap::ClampToEdge,
                wrap_t: TextureWrap::ClampToEdge,
            },
        );
        let depth = DepthTarget::new(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            extent,
            vk::Format::D32_SFLOAT,
        );
        initialize_planar_reflection_image(device, command_pool, queue, image);

        Self {
            view,
            sampler,
            depth,
            extent,
            image,
            memory,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }

    pub fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout
    }

    pub fn set_layout(&mut self, layout: vk::ImageLayout) {
        self.layout = layout;
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe { self.depth.destroy(device) };

        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }

        if self.sampler != vk::Sampler::null() {
            unsafe { device.destroy_sampler(self.sampler, None) };
            self.sampler = vk::Sampler::null();
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

fn initialize_planar_reflection_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        planar_reflection_barrier(
            device,
            command_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );

        unsafe {
            device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.025, 0.035, 0.05, 1.0],
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
        }

        planar_reflection_barrier(
            device,
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    });
}

fn initialize_reflection_probe_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        reflection_probe_barrier(
            device,
            command_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );

        unsafe {
            device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.03, 0.045, 0.065, 1.0],
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: ReflectionProbe::FACE_COUNT as u32,
                }],
            );
        }

        reflection_probe_barrier(
            device,
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    });
}

fn reflection_probe_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: ReflectionProbe::FACE_COUNT as u32,
        });
    let dependency =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
}

fn planar_reflection_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
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

    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
}

fn create_reflection_probe_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    view_type: vk::ImageViewType,
    base_array_layer: u32,
    layer_count: u32,
) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer,
            layer_count,
        });

    unsafe {
        device
            .create_image_view(&info, None)
            .expect("renderer: failed to create reflection probe view")
    }
}

fn create_planar_reflection_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device
            .create_image_view(&info, None)
            .expect("renderer: failed to create planar reflection view")
    }
}

fn create_texture_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = (0..5)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .collect::<Vec<_>>();

    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&info, None)
            .expect("renderer: failed to create texture descriptor layout")
    }
}

fn create_texture_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
    const MAX_MATERIALS: u32 = 1024;
    const TEXTURES_PER_MATERIAL: u32 = 5;

    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: MAX_MATERIALS * TEXTURES_PER_MATERIAL,
    };
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MAX_MATERIALS)
        .pool_sizes(std::slice::from_ref(&pool_size));

    unsafe {
        device
            .create_descriptor_pool(&info, None)
            .expect("renderer: failed to create texture descriptor pool")
    }
}

fn allocate_texture_set(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    textures: &[GpuTexture],
    material: Material,
) -> vk::DescriptorSet {
    let layouts = [layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe {
        device
            .allocate_descriptor_sets(&alloc)
            .expect("renderer: failed to allocate texture descriptor set")[0]
    };
    let texture_ids = [
        material.base_color_texture.unwrap_or(TextureId::DEFAULT),
        material
            .metallic_roughness_texture
            .unwrap_or(TextureId::DEFAULT),
        material.normal_texture.unwrap_or(TextureId::NORMAL),
        material.occlusion_texture.unwrap_or(TextureId::DEFAULT),
        material.emissive_texture.unwrap_or(TextureId::DEFAULT),
    ];
    let image_infos = texture_ids
        .into_iter()
        .enumerate()
        .map(|(index, texture)| {
            let fallback = if index == 2 {
                TextureId::NORMAL
            } else {
                TextureId::DEFAULT
            };
            let texture = textures
                .get(texture.0)
                .or_else(|| textures.get(fallback.0))
                .expect("renderer: missing default texture");

            vk::DescriptorImageInfo::default()
                .sampler(texture.sampler)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();
    let writes = image_infos
        .iter()
        .enumerate()
        .map(|(binding, image_info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(image_info))
        })
        .collect::<Vec<_>>();

    unsafe { device.update_descriptor_sets(&writes, &[]) };
    set
}

fn create_texture_image(
    instance: &Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    srgb: bool,
) -> (vk::Image, vk::DeviceMemory) {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(texture_format(srgb))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("renderer: failed to create texture image")
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
            .expect("renderer: failed to allocate texture memory")
    };

    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("renderer: failed to bind texture memory")
    };

    (image, memory)
}

fn create_texture_view(device: &ash::Device, image: vk::Image, srgb: bool) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(texture_format(srgb))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device
            .create_image_view(&info, None)
            .expect("renderer: failed to create texture view")
    }
}

fn texture_format(srgb: bool) -> vk::Format {
    if srgb {
        vk::Format::R8G8B8A8_SRGB
    } else {
        vk::Format::R8G8B8A8_UNORM
    }
}

fn create_texture_sampler(device: &ash::Device, sampler: TextureSampler) -> vk::Sampler {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(texture_filter(sampler.mag_filter))
        .min_filter(texture_filter(sampler.min_filter))
        .address_mode_u(texture_wrap(sampler.wrap_s))
        .address_mode_v(texture_wrap(sampler.wrap_t))
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(0.0);

    unsafe {
        device
            .create_sampler(&info, None)
            .expect("renderer: failed to create texture sampler")
    }
}

fn texture_filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

fn texture_wrap(wrap: TextureWrap) -> vk::SamplerAddressMode {
    match wrap {
        TextureWrap::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        TextureWrap::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        TextureWrap::Repeat => vk::SamplerAddressMode::REPEAT,
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

fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            )
        };
    });
}

fn transition_texture_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        ),
        _ => panic!("unsupported texture layout transition"),
    };

    submit_once(device, command_pool, queue, |command_buffer| {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
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

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
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
