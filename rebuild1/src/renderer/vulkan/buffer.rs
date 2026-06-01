use std::mem::size_of_val;

use ash::{Device, Instance, vk};

use super::VulkanError;

pub(super) struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl GpuBuffer {
    /// Returns the raw Vulkan buffer handle for command buffer binding.
    pub(super) fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Destroys the buffer and its bound memory allocation.
    pub(super) fn destroy(self, device: &Device) {
        // Safety: buffers are destroyed after all submitted work using them is idle.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }

    /// Maps host-visible memory for one bounded read and unmaps it before returning.
    pub(super) fn read_bytes<R>(
        &self,
        device: &Device,
        size: vk::DeviceSize,
        read: impl FnOnce(&[u8]) -> R,
    ) -> Result<R, VulkanError> {
        // Safety: this buffer's allocation is host visible and coherent, and callers pass a byte
        // count that fits the allocation made for this buffer.
        unsafe {
            let mapped = device.map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())?;
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize);
            let value = read(bytes);
            device.unmap_memory(self.memory);
            Ok(value)
        }
    }
}

/// Reads memory type flags for the selected physical device.
pub(super) fn memory_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceMemoryProperties {
    // Safety: the physical device was selected from this Vulkan instance.
    unsafe { instance.get_physical_device_memory_properties(physical_device) }
}

/// Creates one host-visible buffer and fills it with typed data.
pub(super) fn create_buffer_with_data<T: Copy>(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    values: &[T],
) -> Result<GpuBuffer, VulkanError> {
    let size = size_of_val(values) as vk::DeviceSize;
    let buffer = create_buffer(device, size, usage)?;
    let memory = match allocate_buffer_memory(device, memory_properties, buffer) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_buffer(device, buffer);
            return Err(error);
        }
    };

    // Safety: the buffer and allocation were created by this device and the allocation satisfies
    // the memory requirements returned for the buffer.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.map_err(VulkanError::Vk)?;
    let gpu_buffer = GpuBuffer { buffer, memory };
    write_buffer_slice(device, &gpu_buffer, values)?;
    Ok(gpu_buffer)
}

/// Creates one host-visible coherent buffer with explicit usage and byte size.
pub(super) fn create_host_buffer(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    size: vk::DeviceSize,
) -> Result<GpuBuffer, VulkanError> {
    let buffer = create_buffer(device, size, usage)?;
    let memory = match allocate_buffer_memory(device, memory_properties, buffer) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_buffer(device, buffer);
            return Err(error);
        }
    };

    // Safety: the selected allocation type satisfies the buffer requirements.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.map_err(VulkanError::Vk)?;
    Ok(GpuBuffer { buffer, memory })
}

/// Copies one typed value into a host-visible coherent buffer allocation.
pub(super) fn write_buffer_value<T: Copy>(
    device: &Device,
    buffer: &GpuBuffer,
    value: &T,
) -> Result<(), VulkanError> {
    write_buffer_slice(device, buffer, std::slice::from_ref(value))
}

/// Destroys a group of GPU buffers.
pub(super) fn destroy_buffers(device: &Device, buffers: Vec<GpuBuffer>) {
    for buffer in buffers {
        buffer.destroy(device);
    }
}

/// Creates a Vulkan buffer with the requested byte size and usage.
fn create_buffer(
    device: &Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer, VulkanError> {
    let create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // Safety: the create info contains only local values and no custom allocator is used.
    unsafe { device.create_buffer(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates host-visible memory compatible with one Vulkan buffer.
fn allocate_buffer_memory(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer: vk::Buffer,
) -> Result<vk::DeviceMemory, VulkanError> {
    // Safety: the buffer was created by this device.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);

    // Safety: the memory type index was selected from this physical device's memory properties.
    unsafe { device.allocate_memory(&allocate_info, None) }.map_err(VulkanError::Vk)
}

/// Finds a memory type supported by a resource and matching required properties.
pub(super) fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, VulkanError> {
    for index in 0..memory_properties.memory_type_count {
        let supported = (type_bits & (1 << index)) != 0;
        let flags = memory_properties.memory_types[index as usize].property_flags;

        if supported && flags.contains(required) {
            return Ok(index);
        }
    }

    Err(VulkanError::MemoryTypeUnavailable)
}

/// Copies one typed slice into a host-visible coherent buffer allocation.
fn write_buffer_slice<T: Copy>(
    device: &Device,
    buffer: &GpuBuffer,
    values: &[T],
) -> Result<(), VulkanError> {
    let size = size_of_val(values) as vk::DeviceSize;

    // Safety: the allocation is host visible, coherent, and large enough for this typed slice.
    unsafe {
        let mapped = device.map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>(),
            size as usize,
        );
        device.unmap_memory(buffer.memory);
    }

    Ok(())
}

/// Destroys one raw buffer handle without freeing memory.
fn destroy_buffer(device: &Device, buffer: vk::Buffer) {
    if buffer == vk::Buffer::null() {
        return;
    }

    // Safety: the buffer was created by this device and is not used after this point.
    unsafe {
        device.destroy_buffer(buffer, None);
    }
}
