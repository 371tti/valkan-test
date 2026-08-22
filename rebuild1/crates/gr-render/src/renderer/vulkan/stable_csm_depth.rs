//! Vulkan depth arrays for the four stable CSM directional cascades.

use ash::{Device, vk};

use crate::protocol::NonZeroExtent;

use super::{VulkanError, buffer::find_memory_type, immediate::submit_immediate_commands};

pub(super) const STABLE_CSM_DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;

pub(super) struct StableCsmDepthArray {
    pub(super) image: vk::Image,
    pub(super) memory: vk::DeviceMemory,
    pub(super) sampled_view: vk::ImageView,
    pub(super) layer_views: Vec<vk::ImageView>,
    pub(super) layer_count: u32,
}

impl StableCsmDepthArray {
    pub(super) fn create(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        extent: NonZeroExtent,
        layer_count: u32,
    ) -> Result<Self, VulkanError> {
        let layer_count = layer_count.max(1);
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(STABLE_CSM_DEPTH_FORMAT)
            .extent(vk::Extent3D {
                width: extent.width(),
                height: extent.height(),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layer_count)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&create_info, None) }.map_err(VulkanError::Vk)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = match find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(error);
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(VulkanError::Vk(error));
            }
        };
        if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(VulkanError::Vk(error));
        }
        let sampled_view = match create_view(
            device,
            image,
            vk::ImageViewType::TYPE_2D_ARRAY,
            0,
            layer_count,
        ) {
            Ok(view) => view,
            Err(error) => {
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_image(image, None);
                }
                return Err(error);
            }
        };
        let mut layer_views = Vec::with_capacity(layer_count as usize);
        for layer in 0..layer_count {
            match create_view(device, image, vk::ImageViewType::TYPE_2D, layer, 1) {
                Ok(view) => layer_views.push(view),
                Err(error) => {
                    unsafe {
                        for view in layer_views.drain(..) {
                            device.destroy_image_view(view, None);
                        }
                        device.destroy_image_view(sampled_view, None);
                        device.free_memory(memory, None);
                        device.destroy_image(image, None);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            image,
            memory,
            sampled_view,
            layer_views,
            layer_count,
        })
    }

    pub(super) fn initialize_shader_read(
        &self,
        device: &Device,
        queue_family_index: u32,
        queue: vk::Queue,
    ) -> Result<(), VulkanError> {
        submit_immediate_commands(device, queue_family_index, queue, |command_buffer| {
            let range = self.full_range();
            image_barrier(
                device,
                command_buffer,
                self.image,
                range,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            unsafe {
                device.cmd_clear_depth_stencil_image(
                    command_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    &[range],
                );
            }
            image_barrier(
                device,
                command_buffer,
                self.image,
                range,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
        })
    }

    pub(super) fn transition_layer_to_attachment(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        layer: u32,
    ) {
        image_barrier(
            device,
            command_buffer,
            self.image,
            self.layer_range(layer),
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    }

    pub(super) fn transition_layer_to_shader_read(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        layer: u32,
    ) {
        image_barrier(
            device,
            command_buffer,
            self.image,
            self.layer_range(layer),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
    }

    pub(super) fn destroy(self, device: &Device) {
        unsafe {
            for view in self.layer_views {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(self.sampled_view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }

    fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(self.layer_count)
    }
    fn layer_range(&self, layer: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(layer.min(self.layer_count - 1))
            .layer_count(1)
    }
}

fn create_view(
    device: &Device,
    image: vk::Image,
    view_type: vk::ImageViewType,
    base_layer: u32,
    layer_count: u32,
) -> Result<vk::ImageView, VulkanError> {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(base_layer)
        .layer_count(layer_count);
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(STABLE_CSM_DEPTH_FORMAT)
        .subresource_range(range);
    unsafe { device.create_image_view(&info, None) }.map_err(VulkanError::Vk)
}

#[allow(clippy::too_many_arguments)]
fn image_barrier(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
    source_access: vk::AccessFlags,
    destination_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(source_access)
        .dst_access_mask(destination_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );
    }
}
