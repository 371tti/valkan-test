use std::{collections::BTreeMap, mem::size_of};

use ash::vk::Handle;
use ash::{Device, vk};

use crate::{
    protocol::{
        AssetHandle, MaterialAlphaMode, MaterialDescriptor, MaterialHandle, MaterialTextureSlot,
        TextureDescriptor, TextureFormat, TextureHandle,
    },
    renderer::pipeline::shader_interface,
};

use super::{
    VulkanError,
    buffer::{GpuBuffer, create_buffer_with_data, find_memory_type},
};

const MATERIAL_DESCRIPTOR_CAPACITY: u32 = 1024;

pub(super) struct VulkanMaterialStore {
    textures: BTreeMap<TextureHandle, VulkanTexture>,
    materials: BTreeMap<MaterialHandle, VulkanMaterial>,
    descriptor_pool: vk::DescriptorPool,
    material_set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
}

impl VulkanMaterialStore {
    /// Creates Vulkan descriptor resources used by uploaded material records.
    pub(super) fn create(device: &Device) -> Result<Self, VulkanError> {
        let material_set_layout = create_material_set_layout(device)?;
        let descriptor_pool = match create_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_descriptor_set_layout(device, material_set_layout);
                return Err(error);
            }
        };
        let sampler = match create_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_descriptor_pool(device, descriptor_pool);
                destroy_descriptor_set_layout(device, material_set_layout);
                return Err(error);
            }
        };
        tracing::info!("created Vulkan material resources");

        Ok(Self {
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            descriptor_pool,
            material_set_layout,
            sampler,
        })
    }

    /// Uploads imported RGBA texture payloads into sampled Vulkan images.
    pub(super) fn upload_imported_textures(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        records: &[(TextureHandle, TextureDescriptor)],
    ) -> Result<(), VulkanError> {
        for (handle, descriptor) in records {
            let texture = VulkanTexture::upload(
                device,
                memory_properties,
                queue_family_index,
                queue,
                descriptor,
            )?;
            tracing::trace!(
                texture = handle.raw(),
                width = descriptor.width(),
                height = descriptor.height(),
                format = descriptor.format().name(),
                "uploaded Vulkan texture image"
            );
            if let Some(old) = self.textures.insert(*handle, texture) {
                old.destroy(device);
            }
        }

        Ok(())
    }

    /// Uploads material parameter buffers and texture descriptor sets for imported materials.
    pub(super) fn upload_imported_materials(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        records: &[(MaterialHandle, MaterialDescriptor)],
    ) -> Result<(), VulkanError> {
        for (handle, descriptor) in records {
            let material = VulkanMaterial::upload(
                device,
                memory_properties,
                self.descriptor_pool,
                self.material_set_layout,
                self.sampler,
                descriptor,
                &self.textures,
            )?;
            tracing::trace!(
                material = handle.raw(),
                alpha_mode = descriptor.alpha_mode().name(),
                texture_count = descriptor.textures().len(),
                descriptor_set = material.descriptor_set().as_raw(),
                "uploaded Vulkan material descriptors"
            );
            if let Some(old) = self.materials.insert(*handle, material) {
                old.destroy(device);
            }
        }

        Ok(())
    }

    /// Returns the material descriptor set layout used by mesh pipeline layouts.
    pub(super) fn material_set_layout(&self) -> vk::DescriptorSetLayout {
        self.material_set_layout
    }

    /// Returns the descriptor set for a live material handle.
    pub(super) fn descriptor_set_for(&self, material: MaterialHandle) -> Option<vk::DescriptorSet> {
        self.materials
            .get(&material)
            .map(VulkanMaterial::descriptor_set)
    }

    /// Returns whether the material can be drawn with the texture-sampling pipeline variant.
    pub(super) fn has_base_color_texture(&self, material: MaterialHandle) -> bool {
        self.materials
            .get(&material)
            .is_some_and(VulkanMaterial::has_base_color_texture)
    }

    /// Destroys backend material and texture resources whose protocol handles have retired.
    pub(super) fn destroy_retired(&mut self, device: &Device, retired: &[AssetHandle]) {
        for asset in retired {
            match *asset {
                AssetHandle::Material(material) => self.destroy_material(device, material),
                AssetHandle::Texture(texture) => self.destroy_texture(device, texture),
                AssetHandle::Scene(_) | AssetHandle::Mesh(_) => {}
            }
        }
    }

    /// Destroys every uploaded material, texture image, and descriptor owner.
    pub(super) fn destroy(self, device: &Device) {
        for material in self.materials.into_values() {
            material.destroy(device);
        }
        for texture in self.textures.into_values() {
            texture.destroy(device);
        }
        destroy_sampler(device, self.sampler);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_descriptor_set_layout(device, self.material_set_layout);
    }

    /// Destroys one material parameter buffer when its handle retires.
    fn destroy_material(&mut self, device: &Device, material: MaterialHandle) {
        if let Some(uploaded) = self.materials.remove(&material) {
            tracing::trace!(
                material = material.raw(),
                "destroying retired Vulkan material"
            );
            uploaded.destroy(device);
        }
    }

    /// Destroys one sampled image when its texture handle retires.
    fn destroy_texture(&mut self, device: &Device, texture: TextureHandle) {
        if let Some(uploaded) = self.textures.remove(&texture) {
            tracing::trace!(texture = texture.raw(), "destroying retired Vulkan texture");
            uploaded.destroy(device);
        }
    }
}

impl Default for VulkanMaterialStore {
    /// Creates an inert store for tests that do not construct Vulkan resources.
    fn default() -> Self {
        Self {
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            material_set_layout: vk::DescriptorSetLayout::null(),
            sampler: vk::Sampler::null(),
        }
    }
}

struct VulkanMaterial {
    params: GpuBuffer,
    descriptor_set: vk::DescriptorSet,
    has_base_color_texture: bool,
}

impl VulkanMaterial {
    /// Creates one material parameter buffer and writes descriptors for explicit textures only.
    fn upload(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        descriptor_pool: vk::DescriptorPool,
        material_set_layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
        descriptor: &MaterialDescriptor,
        textures: &BTreeMap<TextureHandle, VulkanTexture>,
    ) -> Result<Self, VulkanError> {
        let base_color = descriptor
            .texture(MaterialTextureSlot::BaseColor)
            .and_then(|handle| textures.get(&handle));
        let params = create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &[MaterialParams::from_descriptor(
                descriptor,
                base_color.is_some(),
            )],
        )?;
        let descriptor_set =
            match allocate_material_descriptor_set(device, descriptor_pool, material_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    params.destroy(device);
                    return Err(error);
                }
            };

        let has_base_color_texture = base_color.is_some();
        update_material_descriptor_set(device, descriptor_set, &params, sampler, base_color);
        Ok(Self {
            params,
            descriptor_set,
            has_base_color_texture,
        })
    }

    /// Returns the descriptor set written for this material record.
    fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    /// Returns whether this material wrote a base-color texture descriptor.
    fn has_base_color_texture(&self) -> bool {
        self.has_base_color_texture
    }

    /// Destroys the material parameter buffer after GPU use has retired.
    fn destroy(self, device: &Device) {
        self.params.destroy(device);
    }
}

struct VulkanTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl VulkanTexture {
    /// Creates a sampled image and copies the imported texture payload into it.
    fn upload(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        descriptor: &TextureDescriptor,
    ) -> Result<Self, VulkanError> {
        let format = texture_format(descriptor.format());
        let staging = create_buffer_with_data(
            device,
            memory_properties,
            vk::BufferUsageFlags::TRANSFER_SRC,
            descriptor.pixels(),
        )?;
        let image =
            match create_texture_image(device, descriptor.width(), descriptor.height(), format) {
                Ok(image) => image,
                Err(error) => {
                    staging.destroy(device);
                    return Err(error);
                }
            };
        let memory = match allocate_image_memory(device, memory_properties, image) {
            Ok(memory) => memory,
            Err(error) => {
                destroy_image(device, image);
                staging.destroy(device);
                return Err(error);
            }
        };
        if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
            free_memory(device, memory);
            destroy_image(device, image);
            staging.destroy(device);
            return Err(VulkanError::Vk(error));
        }

        let upload_result = upload_texture_pixels(
            device,
            queue_family_index,
            queue,
            staging.handle(),
            image,
            descriptor.width(),
            descriptor.height(),
        );
        staging.destroy(device);
        upload_result?;

        let view = match create_texture_image_view(device, image, format) {
            Ok(view) => view,
            Err(error) => {
                free_memory(device, memory);
                destroy_image(device, image);
                return Err(error);
            }
        };

        Ok(Self {
            image,
            memory,
            view,
        })
    }

    /// Destroys the sampled texture image, view, and memory allocation.
    fn destroy(self, device: &Device) {
        destroy_image_view(device, self.view);
        free_memory(device, self.memory);
        destroy_image(device, self.image);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterialParams {
    alpha_mode: u32,
    alpha_cutoff: f32,
    has_base_color: u32,
    _pad: u32,
}

impl MaterialParams {
    /// Converts protocol material facts into the stable shader parameter layout.
    fn from_descriptor(descriptor: &MaterialDescriptor, has_base_color: bool) -> Self {
        Self {
            alpha_mode: alpha_mode_code(descriptor.alpha_mode()),
            alpha_cutoff: descriptor.alpha_cutoff_milli() as f32 / 1000.0,
            has_base_color: u32::from(has_base_color),
            _pad: 0,
        }
    }
}

/// Returns the Vulkan format used by one protocol texture payload.
fn texture_format(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
    }
}

/// Converts alpha modes into stable shader-side integer tags.
fn alpha_mode_code(mode: MaterialAlphaMode) -> u32 {
    match mode {
        MaterialAlphaMode::Opaque => 0,
        MaterialAlphaMode::Cutout => 1,
        MaterialAlphaMode::Transparent => 2,
    }
}

/// Creates the descriptor layout shared by uploaded material records.
fn create_material_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(shader_interface::MATERIAL_PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    for slot in [
        MaterialTextureSlot::BaseColor,
        MaterialTextureSlot::Normal,
        MaterialTextureSlot::MetallicRoughness,
        MaterialTextureSlot::Occlusion,
        MaterialTextureSlot::Emissive,
    ] {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(shader_interface::material_texture_binding(slot))
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
    }
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the bounded descriptor pool used for imported material records.
fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MATERIAL_DESCRIPTOR_CAPACITY),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MATERIAL_DESCRIPTOR_CAPACITY * 5),
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MATERIAL_DESCRIPTOR_CAPACITY)
        .pool_sizes(&pool_sizes);

    // Safety: pool sizes are local values and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the sampler shared by imported material texture descriptors.
fn create_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .max_lod(1.0);

    // Safety: sampler creation uses only local scalar values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates one descriptor set for a material record.
fn allocate_material_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    material_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VulkanError> {
    let layouts = [material_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and set layout are alive for this allocation.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map(|mut sets| sets.remove(0))
        .map_err(VulkanError::Vk)
}

/// Writes parameter and explicit texture descriptors for one material set.
fn update_material_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    params: &GpuBuffer,
    sampler: vk::Sampler,
    base_color: Option<&VulkanTexture>,
) {
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(params.handle())
        .offset(0)
        .range(size_of::<MaterialParams>() as vk::DeviceSize)];
    let image_info = base_color.map(|texture| {
        vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    });
    let mut writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(shader_interface::MATERIAL_PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info),
    ];
    if let Some(image_info) = image_info.as_ref() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_BASE_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(image_info)),
        );
    }

    // Safety: descriptor set and resources are alive and the write slices outlive the call.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Creates the optimal tiled image that receives an imported texture upload.
fn create_texture_image(
    device: &Device,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<vk::Image, VulkanError> {
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
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
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    // Safety: image create info contains only local values and no custom allocator is used.
    unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates device-local memory compatible with a texture image.
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

/// Runs one blocking upload command buffer on the renderer queue.
fn upload_texture_pixels(
    device: &Device,
    queue_family_index: u32,
    queue: vk::Queue,
    staging: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<(), VulkanError> {
    submit_immediate_commands(device, queue_family_index, queue, |command_buffer| {
        transition_texture_image(
            device,
            command_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        copy_buffer_to_image(device, command_buffer, staging, image, width, height);
        transition_texture_image(
            device,
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
    })
}

/// Creates, records, submits, and frees one short-lived command buffer.
fn submit_immediate_commands(
    device: &Device,
    queue_family_index: u32,
    queue: vk::Queue,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<(), VulkanError> {
    let command_pool = create_immediate_command_pool(device, queue_family_index)?;
    let command_buffer = match allocate_immediate_command_buffer(device, command_pool) {
        Ok(command_buffer) => command_buffer,
        Err(error) => {
            destroy_command_pool(device, command_pool);
            return Err(error);
        }
    };
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // Safety: the command buffer was allocated from this pool and is recorded once.
    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
        record(command_buffer);
        device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &command_buffers);
    }
    destroy_command_pool(device, command_pool);

    Ok(())
}

/// Creates a transient pool for one upload command.
fn create_immediate_command_pool(
    device: &Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, VulkanError> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);

    // Safety: the queue family index belongs to this logical device.
    unsafe { device.create_command_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates the primary command buffer used by one upload command.
fn allocate_immediate_command_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, VulkanError> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    // Safety: the command pool belongs to this device and is alive for allocation.
    unsafe { device.allocate_command_buffers(&allocate_info) }
        .map(|mut buffers| buffers.remove(0))
        .map_err(VulkanError::Vk)
}

/// Records one texture image layout transition for upload.
fn transition_texture_image(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(texture_subresource_range());
    let barriers = [barrier];

    // Safety: command buffer is recording and the image belongs to the renderer device.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }
}

/// Records the staging buffer copy into the texture image.
fn copy_buffer_to_image(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    let subresource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);
    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    let regions = [region];

    // Safety: the destination image is in TRANSFER_DST layout and both resources are alive.
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }
}

/// Creates a sampled image view for one uploaded texture.
fn create_texture_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, VulkanError> {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(texture_subresource_range());

    // Safety: the image is a color texture created by this device.
    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Returns the single-mip color range used by imported texture images.
fn texture_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

/// Destroys one Vulkan sampler.
fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler != vk::Sampler::null() {
        // Safety: the sampler was created by this device and is destroyed after descriptor use.
        unsafe {
            device.destroy_sampler(sampler, None);
        }
    }
}

/// Destroys one descriptor pool and all sets allocated from it.
fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool != vk::DescriptorPool::null() {
        // Safety: the pool is destroyed after all descriptor users are idle.
        unsafe {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}

/// Destroys one descriptor set layout.
fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        // Safety: descriptor set layouts are destroyed after dependent pools and pipelines.
        unsafe {
            device.destroy_descriptor_set_layout(layout, None);
        }
    }
}

/// Destroys one temporary command pool.
fn destroy_command_pool(device: &Device, command_pool: vk::CommandPool) {
    if command_pool != vk::CommandPool::null() {
        // Safety: all buffers from this pool were freed or are implicitly freed here.
        unsafe {
            device.destroy_command_pool(command_pool, None);
        }
    }
}

/// Destroys one image view.
fn destroy_image_view(device: &Device, image_view: vk::ImageView) {
    if image_view != vk::ImageView::null() {
        // Safety: the image view was created by this device and is destroyed exactly once.
        unsafe {
            device.destroy_image_view(image_view, None);
        }
    }
}

/// Destroys one raw image handle.
fn destroy_image(device: &Device, image: vk::Image) {
    if image != vk::Image::null() {
        // Safety: the image was created by this device and is destroyed after GPU idle.
        unsafe {
            device.destroy_image(image, None);
        }
    }
}

/// Frees one image memory allocation.
fn free_memory(device: &Device, memory: vk::DeviceMemory) {
    if memory != vk::DeviceMemory::null() {
        // Safety: the allocation belongs to this device and is no longer bound to live work.
        unsafe {
            device.free_memory(memory, None);
        }
    }
}
