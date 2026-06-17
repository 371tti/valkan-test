use ash::{Device, vk};

use crate::protocol::NonZeroExtent;

use super::{
    VulkanError,
    buffer::find_memory_type,
    immediate::{submit_immediate_commands, transition_image},
};

pub(super) struct ColorTarget {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) view: vk::ImageView,
    pub(super) format: vk::Format,
}

pub(super) struct DepthTarget {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) view: vk::ImageView,
    pub(super) format: vk::Format,
}

/// Clears 1x1 fallback shadow images to full light and makes them shader-readable.
pub(super) fn initialize_shadow_sampler_fallback_images(
    device: &Device,
    queue_family_index: u32,
    queue: vk::Queue,
    moment_image: vk::Image,
    transmittance_image: vk::Image,
) -> Result<(), VulkanError> {
    submit_immediate_commands(device, queue_family_index, queue, |command_buffer| {
        transition_image(
            device,
            command_buffer,
            moment_image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        transition_image(
            device,
            command_buffer,
            transmittance_image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        clear_shadow_fallback_images(device, command_buffer, moment_image, transmittance_image);
        transition_image(
            device,
            command_buffer,
            moment_image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        transition_image(
            device,
            command_buffer,
            transmittance_image,
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

/// Creates a graph-owned color target image, memory allocation, and view.
pub(super) fn create_color_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<ColorTarget, VulkanError> {
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width(),
            height: extent.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)?;
    let memory = match allocate_image_memory(device, memory_properties, image) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_image(device, image);
            return Err(error);
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        free_memory(device, memory);
        destroy_image(device, image);
        return Err(VulkanError::Vk(error));
    }
    let view = match create_color_image_view(device, image, format) {
        Ok(view) => view,
        Err(error) => {
            free_memory(device, memory);
            destroy_image(device, image);
            return Err(error);
        }
    };

    tracing::trace!(
        width = extent.width(),
        height = extent.height(),
        format = ?format,
        usage = ?usage,
        "created Vulkan color target"
    );
    Ok(ColorTarget {
        image,
        memory,
        view,
        format,
    })
}

/// Creates the depth image, memory, and view shared by the graph scene pass.
pub(super) fn create_depth_target(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<DepthTarget, VulkanError> {
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width(),
            height: extent.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)?;
    let memory = match allocate_image_memory(device, memory_properties, image) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_image(device, image);
            return Err(error);
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        free_memory(device, memory);
        destroy_image(device, image);
        return Err(VulkanError::Vk(error));
    }
    let view = match create_depth_image_view(device, image, format) {
        Ok(view) => view,
        Err(error) => {
            free_memory(device, memory);
            destroy_image(device, image);
            return Err(error);
        }
    };

    tracing::trace!(
        width = extent.width(),
        height = extent.height(),
        format = ?format,
        usage = ?usage,
        "created Vulkan depth target"
    );
    Ok(DepthTarget {
        image,
        memory,
        view,
        format,
    })
}

/// Destroys a graph-owned color target after the device is idle.
pub(super) fn destroy_color_target(device: &Device, color: ColorTarget) {
    destroy_image_view(device, color.view);
    free_memory(device, color.memory);
    destroy_image(device, color.image);
}

/// Destroys a depth target after all framebuffers that reference it are gone.
pub(super) fn destroy_depth_target(device: &Device, depth: DepthTarget) {
    destroy_image_view(device, depth.view);
    free_memory(device, depth.memory);
    destroy_image(device, depth.image);
}

/// Destroys one image view.
pub(super) fn destroy_image_view(device: &Device, image_view: vk::ImageView) {
    if image_view == vk::ImageView::null() {
        return;
    }

    unsafe {
        device.destroy_image_view(image_view, None);
    }
}

/// Writes full-light values into dummy shadow maps in transfer-destination layout.
fn clear_shadow_fallback_images(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    moment_image: vk::Image,
    transmittance_image: vk::Image,
) {
    let color = vk::ClearColorValue {
        float32: [1.0, 1.0, 1.0, 1.0],
    };
    let color_range = [image_subresource_range(vk::ImageAspectFlags::COLOR)];

    unsafe {
        device.cmd_clear_color_image(
            command_buffer,
            moment_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &color,
            &color_range,
        );
        device.cmd_clear_color_image(
            command_buffer,
            transmittance_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &color,
            &color_range,
        );
    }
}

fn image_subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn create_color_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, VulkanError> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

fn allocate_image_memory(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
) -> Result<vk::DeviceMemory, VulkanError> {
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);

    unsafe { device.allocate_memory(&allocate_info, None) }.map_err(VulkanError::Vk)
}

fn create_depth_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, VulkanError> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    unsafe { device.create_image_view(&create_info, None) }.map_err(VulkanError::Vk)
}

fn destroy_image(device: &Device, image: vk::Image) {
    if image == vk::Image::null() {
        return;
    }

    unsafe {
        device.destroy_image(image, None);
    }
}

fn free_memory(device: &Device, memory: vk::DeviceMemory) {
    if memory == vk::DeviceMemory::null() {
        return;
    }

    unsafe {
        device.free_memory(memory, None);
    }
}
