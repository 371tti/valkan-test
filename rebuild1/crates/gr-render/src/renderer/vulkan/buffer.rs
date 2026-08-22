use std::mem::size_of_val;

use ash::{Device, Instance, vk};

use super::{VulkanError, immediate::submit_immediate_commands};

pub(super) struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct PendingDeviceLocalUpload {
    staging: GpuBuffer,
    destination: GpuBuffer,
    size: vk::DeviceSize,
    final_usage: vk::BufferUsageFlags,
}

#[derive(Clone, Copy)]
struct DeferredDeviceLocalCopy {
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    final_usage: vk::BufferUsageFlags,
}

/// Owns staging buffers whose copies will be recorded by a later command buffer.
///
/// Destination buffers are returned separately so descriptor sets can be built before submission.
/// The staging allocations must remain alive until the command buffer's fence has completed.
pub(super) struct DeferredDeviceLocalBufferUploads {
    staging: Vec<GpuBuffer>,
    copies: Vec<DeferredDeviceLocalCopy>,
}

impl DeferredDeviceLocalBufferUploads {
    /// Records all deferred copies and their transfer-to-consumer visibility barriers.
    pub(super) fn record(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        for copy in &self.copies {
            copy_buffer(
                device,
                command_buffer,
                copy.source,
                copy.destination,
                copy.size,
            );
            transition_uploaded_buffer(
                device,
                command_buffer,
                copy.destination,
                copy.size,
                copy.final_usage,
            );
        }
    }

    /// Releases staging allocations after the submission containing `record` has completed.
    pub(super) fn destroy(self, device: &Device) {
        destroy_buffers(device, self.staging);
    }
}

/// Collects device-local uploads for either one immediate setup submit or a deferred caller submit.
pub(super) struct DeviceLocalBufferUploadBatch<'a> {
    device: &'a Device,
    memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
    uploads: Vec<PendingDeviceLocalUpload>,
}

impl<'a> DeviceLocalBufferUploadBatch<'a> {
    pub(super) fn new(
        device: &'a Device,
        memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        Self {
            device,
            memory_properties,
            uploads: Vec::new(),
        }
    }

    /// Copies typed data into a staging allocation and queues one device-local destination.
    pub(super) fn push<T: Copy>(
        &mut self,
        usage: vk::BufferUsageFlags,
        values: &[T],
    ) -> Result<(), VulkanError> {
        let size = size_of_val(values) as vk::DeviceSize;
        let staging = create_buffer_with_data(
            self.device,
            self.memory_properties,
            vk::BufferUsageFlags::TRANSFER_SRC,
            values,
        )?;
        let destination = match create_buffer_with_properties(
            self.device,
            self.memory_properties,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            size,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                staging.destroy(self.device);
                return Err(error);
            }
        };
        self.uploads.push(PendingDeviceLocalUpload {
            staging,
            destination,
            size,
            final_usage: usage,
        });
        Ok(())
    }

    /// Returns destinations and keeps their staging copies deferred for a caller-owned submission.
    pub(super) fn finish_deferred(mut self) -> (Vec<GpuBuffer>, DeferredDeviceLocalBufferUploads) {
        let uploads = std::mem::take(&mut self.uploads);
        let mut destinations = Vec::with_capacity(uploads.len());
        let mut staging = Vec::with_capacity(uploads.len());
        let mut copies = Vec::with_capacity(uploads.len());
        for upload in uploads {
            copies.push(DeferredDeviceLocalCopy {
                source: upload.staging.handle(),
                destination: upload.destination.handle(),
                size: upload.size,
                final_usage: upload.final_usage,
            });
            staging.push(upload.staging);
            destinations.push(upload.destination);
        }
        (
            destinations,
            DeferredDeviceLocalBufferUploads { staging, copies },
        )
    }

    /// Submits all queued copies once and returns destinations in insertion order.
    pub(super) fn finish(
        self,
        queue_family_index: u32,
        queue: vk::Queue,
    ) -> Result<Vec<GpuBuffer>, VulkanError> {
        let device = self.device;
        let (destinations, deferred) = self.finish_deferred();
        if destinations.is_empty() {
            return Ok(Vec::new());
        }
        let upload_result =
            submit_immediate_commands(device, queue_family_index, queue, |command_buffer| {
                deferred.record(device, command_buffer);
            });
        deferred.destroy(device);
        if let Err(error) = upload_result {
            destroy_buffers(device, destinations);
            return Err(error);
        }
        Ok(destinations)
    }
}

impl Drop for DeviceLocalBufferUploadBatch<'_> {
    fn drop(&mut self) {
        for upload in self.uploads.drain(..) {
            upload.staging.destroy(self.device);
            upload.destination.destroy(self.device);
        }
    }
}

impl GpuBuffer {
    /// Returns the raw Vulkan buffer handle for command buffer binding.
    pub(super) fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Destroys the buffer and its bound memory allocation.
    pub(super) fn destroy(self, device: &Device) {
        destroy_buffer_allocation(device, self.buffer, self.memory);
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
    let gpu_buffer = create_buffer_with_properties(
        device,
        memory_properties,
        usage,
        size,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    if let Err(error) = write_buffer_slice(device, &gpu_buffer, values) {
        gpu_buffer.destroy(device);
        return Err(error);
    }
    Ok(gpu_buffer)
}

/// Creates one device-local buffer and uploads typed data through a short-lived staging buffer.
pub(super) fn create_device_local_buffer_with_data<T: Copy>(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    queue: vk::Queue,
    usage: vk::BufferUsageFlags,
    values: &[T],
) -> Result<GpuBuffer, VulkanError> {
    let size = size_of_val(values) as vk::DeviceSize;
    let staging = create_buffer_with_data(
        device,
        memory_properties,
        vk::BufferUsageFlags::TRANSFER_SRC,
        values,
    )?;
    let device_buffer = match create_buffer_with_properties(
        device,
        memory_properties,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        size,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            staging.destroy(device);
            return Err(error);
        }
    };

    let upload_result = upload_buffer_copy(
        device,
        queue_family_index,
        queue,
        staging.handle(),
        device_buffer.handle(),
        size,
        usage,
    );
    staging.destroy(device);
    if let Err(error) = upload_result {
        device_buffer.destroy(device);
        return Err(error);
    }

    Ok(device_buffer)
}

/// Creates one host-visible coherent buffer with explicit usage and byte size.
pub(super) fn create_host_buffer(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    size: vk::DeviceSize,
) -> Result<GpuBuffer, VulkanError> {
    create_buffer_with_properties(
        device,
        memory_properties,
        usage,
        size,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
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

/// Creates one buffer and binds memory with explicit placement requirements.
fn create_buffer_with_properties(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    size: vk::DeviceSize,
    properties: vk::MemoryPropertyFlags,
) -> Result<GpuBuffer, VulkanError> {
    let buffer = create_buffer(device, size, usage)?;
    let memory = match allocate_buffer_memory(device, memory_properties, buffer, properties) {
        Ok(memory) => memory,
        Err(error) => {
            destroy_buffer(device, buffer);
            return Err(error);
        }
    };

    // Safety: the selected allocation type satisfies the buffer requirements.
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        destroy_buffer_allocation(device, buffer, memory);
        return Err(VulkanError::Vk(error));
    }
    Ok(GpuBuffer { buffer, memory })
}

/// Allocates memory compatible with one Vulkan buffer and requested placement.
fn allocate_buffer_memory(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer: vk::Buffer,
    properties: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, VulkanError> {
    // Safety: the buffer was created by this device.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index =
        find_memory_type(memory_properties, requirements.memory_type_bits, properties)?;
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

/// Copies one staging buffer into a device-local buffer and makes it visible to draw commands.
fn upload_buffer_copy(
    device: &Device,
    queue_family_index: u32,
    queue: vk::Queue,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    final_usage: vk::BufferUsageFlags,
) -> Result<(), VulkanError> {
    submit_immediate_commands(device, queue_family_index, queue, |command_buffer| {
        copy_buffer(device, command_buffer, source, destination, size);
        transition_uploaded_buffer(device, command_buffer, destination, size, final_usage);
    })
}

/// Records one full-buffer copy for setup-time uploads.
fn copy_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) {
    let regions = [vk::BufferCopy::default().size(size)];

    // Safety: the command buffer is recording and both buffers are alive on this device.
    unsafe {
        device.cmd_copy_buffer(command_buffer, source, destination, &regions);
    }
}

/// Makes uploaded buffer data visible to the pipeline stages that will consume it.
fn transition_uploaded_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) {
    let (dst_stage, dst_access) = buffer_read_stage_access(usage);
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(size);
    let barriers = [barrier];

    // Safety: the command buffer is recording and the buffer belongs to this device.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
    }
}

/// Returns conservative shader/draw read masks for a freshly uploaded static buffer.
fn buffer_read_stage_access(
    usage: vk::BufferUsageFlags,
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let mut stages = vk::PipelineStageFlags::empty();
    let mut access = vk::AccessFlags::empty();

    if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::INDEX_READ;
    }
    if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
        stages |= vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
        access |= vk::AccessFlags::UNIFORM_READ;
    }
    if stages.is_empty() || access.is_empty() {
        (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ,
        )
    } else {
        (stages, access)
    }
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

/// Releases one buffer and the successful allocation associated with it.
fn destroy_buffer_allocation(device: &Device, buffer: vk::Buffer, memory: vk::DeviceMemory) {
    destroy_buffer(device, buffer);
    // Safety: callers transfer ownership of one successful allocation and release it exactly once.
    unsafe {
        device.free_memory(memory, None);
    }
}
