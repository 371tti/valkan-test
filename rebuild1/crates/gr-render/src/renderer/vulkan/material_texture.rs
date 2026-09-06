use ash::{Device, vk};

use crate::protocol::{MaterialTextureSlot, TextureDescriptor, TextureFormat};

use super::{
    VulkanError,
    buffer::{create_buffer_with_data, find_memory_type},
    immediate::{submit_immediate_commands, transition_image},
};

pub(super) struct VulkanTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl VulkanTexture {
    /// Creates a sampled image and copies the imported texture payload into it.
    pub(super) fn upload(
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
            destroy_image(device, image);
            free_memory(device, memory);
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
        if let Err(error) = upload_result {
            destroy_image(device, image);
            free_memory(device, memory);
            return Err(error);
        }

        let view = match create_texture_image_view(device, image, format) {
            Ok(view) => view,
            Err(error) => {
                destroy_image(device, image);
                free_memory(device, memory);
                return Err(error);
            }
        };

        Ok(Self {
            image,
            memory,
            view,
        })
    }

    /// Returns the sampled image view used by material descriptors.
    pub(super) fn image_view(&self) -> vk::ImageView {
        self.view
    }

    /// Destroys the sampled texture image, view, and memory allocation.
    pub(super) fn destroy(self, device: &Device) {
        destroy_image_view(device, self.view);
        destroy_image(device, self.image);
        free_memory(device, self.memory);
    }
}

pub(super) struct DefaultMaterialTextures {
    base_color: VulkanTexture,
    normal: VulkanTexture,
    metallic_roughness: VulkanTexture,
    occlusion: VulkanTexture,
    emissive: VulkanTexture,
}

impl DefaultMaterialTextures {
    /// Uploads the explicit glTF defaults used only when a material omits an optional map.
    pub(super) fn create(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
    ) -> Result<Self, VulkanError> {
        let mut build = DefaultMaterialTextureBuild::new(device);
        build.base_color = Some(Self::upload_default(
            device,
            memory_properties,
            queue_family_index,
            queue,
            TextureDescriptor::solid_rgba8_srgb([255, 255, 255, 255]),
        )?);
        build.normal = Some(Self::upload_default(
            device,
            memory_properties,
            queue_family_index,
            queue,
            TextureDescriptor::solid_rgba8_linear([128, 128, 255, 255]),
        )?);
        build.metallic_roughness = Some(Self::upload_default(
            device,
            memory_properties,
            queue_family_index,
            queue,
            TextureDescriptor::solid_rgba8_linear([255, 255, 255, 255]),
        )?);
        build.occlusion = Some(Self::upload_default(
            device,
            memory_properties,
            queue_family_index,
            queue,
            TextureDescriptor::solid_rgba8_linear([255, 255, 255, 255]),
        )?);
        build.emissive = Some(Self::upload_default(
            device,
            memory_properties,
            queue_family_index,
            queue,
            TextureDescriptor::solid_rgba8_srgb([255, 255, 255, 255]),
        )?);
        Ok(build.finish())
    }

    /// Returns the default image for a material slot according to glTF factor semantics.
    pub(super) fn texture(&self, slot: MaterialTextureSlot) -> &VulkanTexture {
        match slot {
            MaterialTextureSlot::BaseColor => &self.base_color,
            MaterialTextureSlot::Normal => &self.normal,
            MaterialTextureSlot::MetallicRoughness => &self.metallic_roughness,
            MaterialTextureSlot::Occlusion => &self.occlusion,
            MaterialTextureSlot::Emissive => &self.emissive,
        }
    }

    /// Destroys default texture images after all material descriptor users are gone.
    pub(super) fn destroy(self, device: &Device) {
        self.emissive.destroy(device);
        self.occlusion.destroy(device);
        self.metallic_roughness.destroy(device);
        self.normal.destroy(device);
        self.base_color.destroy(device);
    }

    /// Uploads one default map through the same path as imported textures.
    fn upload_default(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        descriptor: TextureDescriptor,
    ) -> Result<VulkanTexture, VulkanError> {
        VulkanTexture::upload(
            device,
            memory_properties,
            queue_family_index,
            queue,
            &descriptor,
        )
    }
}

struct DefaultMaterialTextureBuild<'a> {
    device: &'a Device,
    base_color: Option<VulkanTexture>,
    normal: Option<VulkanTexture>,
    metallic_roughness: Option<VulkanTexture>,
    occlusion: Option<VulkanTexture>,
    emissive: Option<VulkanTexture>,
}

impl<'a> DefaultMaterialTextureBuild<'a> {
    /// Tracks partially-created default textures so error cleanup stays local to this module.
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            base_color: None,
            normal: None,
            metallic_roughness: None,
            occlusion: None,
            emissive: None,
        }
    }

    /// Finishes the build and hands ownership of every uploaded default texture to the caller.
    fn finish(mut self) -> DefaultMaterialTextures {
        DefaultMaterialTextures {
            base_color: take_created(&mut self.base_color, "material default base color"),
            normal: take_created(&mut self.normal, "material default normal"),
            metallic_roughness: take_created(
                &mut self.metallic_roughness,
                "material default metallic roughness",
            ),
            occlusion: take_created(&mut self.occlusion, "material default occlusion"),
            emissive: take_created(&mut self.emissive, "material default emissive"),
        }
    }
}

impl Drop for DefaultMaterialTextureBuild<'_> {
    /// Cleans up any default textures that were uploaded before a later step failed.
    fn drop(&mut self) {
        if let Some(texture) = self.emissive.take() {
            texture.destroy(self.device);
        }
        if let Some(texture) = self.occlusion.take() {
            texture.destroy(self.device);
        }
        if let Some(texture) = self.metallic_roughness.take() {
            texture.destroy(self.device);
        }
        if let Some(texture) = self.normal.take() {
            texture.destroy(self.device);
        }
        if let Some(texture) = self.base_color.take() {
            texture.destroy(self.device);
        }
    }
}

fn take_created<T>(slot: &mut Option<T>, name: &'static str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("{name} was never created"))
}

/// Returns the Vulkan format used by one protocol texture payload.
fn texture_format(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Rgba8Linear => vk::Format::R8G8B8A8_UNORM,
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
        transition_image(
            device,
            command_buffer,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        copy_buffer_to_image(device, command_buffer, staging, image, width, height);
        transition_image(
            device,
            command_buffer,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
    })
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
