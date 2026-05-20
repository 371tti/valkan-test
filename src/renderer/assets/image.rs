use ash::{Instance, vk};

pub(in crate::renderer) struct GpuImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

pub(in crate::renderer) fn image_2d_info(
    format: vk::Format,
    extent: vk::Extent2D,
    usage: vk::ImageUsageFlags,
) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
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
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
}

pub(in crate::renderer) fn create_device_image(
    instance: &Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    image_info: &vk::ImageCreateInfo<'_>,
    label: &str,
) -> GpuImage {
    let image = unsafe {
        device
            .create_image(image_info, None)
            .unwrap_or_else(|err| panic!("renderer: failed to create {label} image: {err}"))
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
            .unwrap_or_else(|err| panic!("renderer: failed to allocate {label} memory: {err}"))
    };

    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .unwrap_or_else(|err| panic!("renderer: failed to bind {label} memory: {err}"))
    };

    GpuImage { image, memory }
}

pub(in crate::renderer) fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    view_type: vk::ImageViewType,
    aspect_mask: vk::ImageAspectFlags,
    base_array_layer: u32,
    layer_count: u32,
    label: &str,
) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer,
            layer_count,
        });

    unsafe {
        device
            .create_image_view(&info, None)
            .unwrap_or_else(|err| panic!("renderer: failed to create {label} view: {err}"))
    }
}

pub(in crate::renderer) unsafe fn destroy_sampler(device: &ash::Device, sampler: &mut vk::Sampler) {
    if *sampler != vk::Sampler::null() {
        unsafe { device.destroy_sampler(*sampler, None) };
        *sampler = vk::Sampler::null();
    }
}

pub(in crate::renderer) unsafe fn destroy_image_view(
    device: &ash::Device,
    view: &mut vk::ImageView,
) {
    if *view != vk::ImageView::null() {
        unsafe { device.destroy_image_view(*view, None) };
        *view = vk::ImageView::null();
    }
}

pub(in crate::renderer) unsafe fn destroy_image_memory(
    device: &ash::Device,
    image: &mut vk::Image,
    memory: &mut vk::DeviceMemory,
) {
    if *image != vk::Image::null() {
        unsafe { device.destroy_image(*image, None) };
        *image = vk::Image::null();
    }

    if *memory != vk::DeviceMemory::null() {
        unsafe { device.free_memory(*memory, None) };
        *memory = vk::DeviceMemory::null();
    }
}

pub(in crate::renderer) fn find_memory_type(
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
