use ash::{Device, vk};

use super::VulkanError;

/// Records one short-lived command buffer, submits it, and waits for the queue.
///
/// This is for setup-time layout transitions and uploads only; frame rendering stays on the
/// reusable frame command buffers.
pub(super) fn submit_immediate_commands(
    device: &Device,
    queue_family_index: u32,
    queue: vk::Queue,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<(), VulkanError> {
    let command_pool = create_command_pool(device, queue_family_index)?;
    let command_buffer = match allocate_command_buffer(device, command_pool) {
        Ok(command_buffer) => command_buffer,
        Err(error) => {
            destroy_command_pool(device, command_pool);
            return Err(error);
        }
    };
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // Safety: the command buffer was allocated from this transient pool and is recorded once.
    // Keep cleanup outside the fallible sequence so begin/end/submit/wait failures cannot leak the
    // transient command buffer or pool.
    let result: Result<(), vk::Result> = (|| {
        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?;
            record(command_buffer);
            device.end_command_buffer(command_buffer)?;
            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
        }
        Ok(())
    })();
    unsafe {
        device.free_command_buffers(command_pool, &[command_buffer]);
    }
    destroy_command_pool(device, command_pool);

    result.map_err(VulkanError::Vk)
}

/// Records one image layout transition for setup-time images.
pub(super) fn transition_image(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
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
        .subresource_range(subresource_range(aspect));
    let barriers = [barrier];

    // Safety: the command buffer is recording and the image belongs to the same logical device.
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

/// Creates a transient command pool for one setup-time submission.
fn create_command_pool(
    device: &Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, VulkanError> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);

    // Safety: the queue family index belongs to this logical device.
    unsafe { device.create_command_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates the primary command buffer used by one setup-time submission.
fn allocate_command_buffer(
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

/// Destroys one transient command pool after its command buffer is freed.
fn destroy_command_pool(device: &Device, command_pool: vk::CommandPool) {
    if command_pool != vk::CommandPool::null() {
        // Safety: the pool belongs to this device and no command buffers remain allocated.
        unsafe {
            device.destroy_command_pool(command_pool, None);
        }
    }
}

/// Returns a single-mip, single-layer image range for color or depth setup images.
fn subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
