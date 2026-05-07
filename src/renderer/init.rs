use std::{
    ffi::{CStr, CString},
    os::raw::c_void,
    sync::Arc,
};

use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::{
    APP_NAME, ENGINE_NAME,
    renderer::{MAX_FRAMES_IN_FLIGHT, QueueFamilyIndices, RendererConfig},
};

const WANT_VALIDATION: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

impl super::Renderer {
    pub fn new(window_ref: Arc<Window>) -> Self {
        let entry = unsafe { Entry::load().expect("renderer init: failed to load Vulkan entry") };
        log::debug!("renderer init: Vulkan entry loaded");

        let validation_enabled = WANT_VALIDATION && validation_layer_supported(&entry);

        if WANT_VALIDATION && !validation_enabled {
            log::warn!("renderer init: validation layer is not supported; validation disabled");
        } else if validation_enabled {
            log::debug!("renderer init: validation enabled");
        } else {
            log::debug!("renderer init: validation disabled");
        }

        let instance = create_instance(&entry, &window_ref, validation_enabled);
        log::debug!("renderer init: Vulkan instance created");

        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, &window_ref);
        log::debug!("renderer init: surface created");

        let (physical_device, queue_family_indices) =
            pick_physical_device(&instance, &surface_loader, surface);

        let logical_device =
            create_logical_device(&instance, physical_device, queue_family_indices);
        let graphics_queue =
            unsafe { logical_device.get_device_queue(queue_family_indices.graphics_family, 0) };
        let present_queue =
            unsafe { logical_device.get_device_queue(queue_family_indices.present_family, 0) };
        log::debug!("renderer init: logical device and queues ready");

        let swapchain_loader = swapchain::Device::new(&instance, &logical_device);

        let (
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        ) = create_swapchain(
            &window_ref,
            &instance,
            &logical_device,
            physical_device,
            &surface_loader,
            surface,
            &swapchain_loader,
            queue_family_indices,
            None,
            None,
        );
        log::debug!(
            "renderer init: swapchain created: {}x{}, images: {}",
            swapchain_extent.width,
            swapchain_extent.height,
            swapchain_images.len()
        );

        let command_pool =
            create_command_pool(&logical_device, queue_family_indices.graphics_family);

        let command_buffers =
            create_command_buffers(&logical_device, command_pool, swapchain_images.len() as u32);
        log::debug!("renderer init: command pool and buffers ready");

        let image_available_semaphores = create_semaphores(&logical_device, MAX_FRAMES_IN_FLIGHT);

        let render_finished_semaphores = create_semaphores(&logical_device, swapchain_images.len());

        let in_flight_fences = create_fences(&logical_device, MAX_FRAMES_IN_FLIGHT);
        log::debug!("renderer init: synchronization objects ready");

        let swapchain_image_layouts = vec![vk::ImageLayout::UNDEFINED; swapchain_images.len()];

        let (debug_utils_loader, debug_messenger) = if validation_enabled {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let messenger = create_debug_messenger(&loader);
            log::debug!("renderer init: debug messenger ready");

            (Some(loader), Some(messenger))
        } else {
            (None, None)
        };

        Self {
            window_ref,
            instance,
            surface_loader,
            surface,
            physical_device,
            queue_family_indices,
            logical_device,
            graphics_queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            swapchain_image_layouts,
            config: RendererConfig::default(),
            debug_utils_loader,
            debug_messenger,
            needs_swapchain_rebuild_fast: false,
            needs_swapchain_rebuild_full: false,
        }
    }
}

fn create_instance(entry: &Entry, window: &Window, enable_validation: bool) -> Instance {
    let app_name = CString::new(APP_NAME).unwrap();
    let engine_name = CString::new(ENGINE_NAME).unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let display_handle = window
        .display_handle()
        .expect("renderer init: failed to get display handle");

    let mut extension_names = ash_window::enumerate_required_extensions(display_handle.as_raw())
        .expect("renderer init: failed to enumerate required extensions")
        .to_vec();

    if enable_validation {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    let layer_names = if enable_validation {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut debug_create_info = debug_messenger_create_info();

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names);

    if enable_validation {
        create_info = create_info.push_next(&mut debug_create_info);
    }

    unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("renderer init: failed to create Vulkan instance")
    }
}

fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> vk::SurfaceKHR {
    let display_handle = window
        .display_handle()
        .expect("renderer init: failed to get display handle");

    let window_handle = window
        .window_handle()
        .expect("renderer init: failed to get window handle");

    unsafe {
        ash_window::create_surface(
            entry,
            instance,
            display_handle.as_raw(),
            window_handle.as_raw(),
            None,
        )
        .expect("renderer init: failed to create Vulkan surface")
    }
}

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, QueueFamilyIndices) {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("renderer init: failed to enumerate physical devices")
    };

    devices
        .into_iter()
        .find_map(|device| {
            let indices = find_queue_families(instance, device, surface_loader, surface)?;

            if !device_supports_swapchain(instance, device) {
                return None;
            }

            let support = query_swapchain_support(device, surface_loader, surface);

            if support.formats.is_empty() || support.present_modes.is_empty() {
                return None;
            }

            let props = unsafe { instance.get_physical_device_properties(device) };

            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            log::debug!("renderer init: selected GPU: {:?}", name);

            Some((device, indices))
        })
        .expect("renderer init: failed to find suitable GPU")
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
) -> ash::Device {
    let queue_priority = [1.0_f32];

    let mut unique_queue_families = vec![indices.graphics_family];

    if indices.present_family != indices.graphics_family {
        unique_queue_families.push(indices.present_family);
    }

    let queue_create_infos: Vec<_> = unique_queue_families
        .iter()
        .map(|&queue_family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priority)
        })
        .collect();

    let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

    let features = vk::PhysicalDeviceFeatures::default();

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features);

    unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .expect("renderer init: failed to create logical device")
    }
}

pub fn create_swapchain(
    window: &Window,
    _instance: &Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: &swapchain::Device,
    indices: QueueFamilyIndices,
    preferred_surface_format: Option<vk::SurfaceFormatKHR>,
    preferred_present_mode: Option<vk::PresentModeKHR>,
) -> (
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
    vk::Extent2D,
) {
    let support = query_swapchain_support(physical_device, surface_loader, surface);

    let surface_format = match preferred_surface_format {
        Some(format) if support.formats.contains(&format) => format,
        Some(format) => {
            log::warn!(
                "renderer: preferred surface format {:?} not supported; falling back",
                format
            );
            choose_surface_format(&support.formats)
        }
        None => choose_surface_format(&support.formats),
    };

    let present_mode = match preferred_present_mode {
        Some(mode) if support.present_modes.contains(&mode) => mode,
        Some(mode) => {
            log::warn!(
                "renderer: preferred present mode {:?} not supported; falling back",
                mode
            );
            choose_present_mode(&support.present_modes)
        }
        None => choose_present_mode(&support.present_modes),
    };
    let extent = choose_extent(window, &support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;

    if support.capabilities.max_image_count != 0 {
        image_count = image_count.min(support.capabilities.max_image_count);
    }

    let queue_family_indices = [indices.graphics_family, indices.present_family];

    let mut create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT // このimageをレンダーターゲット/フレームバッファとして定義
            | vk::ImageUsageFlags::TRANSFER_DST, // このイメージは転送先として使用できる(vkCmdClearとかの対象 リサイズでのクリアとかのため)
        )
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    if indices.graphics_family != indices.present_family {
        create_info = create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&queue_family_indices);
    } else {
        create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }

    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .expect("renderer init: failed to create swapchain")
    };

    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("renderer init: failed to get swapchain images")
    };

    let image_views = images
        .iter()
        .map(|&image| create_image_view(device, image, surface_format.format))
        .collect();

    (
        swapchain,
        images,
        image_views,
        surface_format.format,
        extent,
    )
}

fn create_command_pool(device: &ash::Device, graphics_queue_family: u32) -> vk::CommandPool {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    unsafe {
        device
            .create_command_pool(&create_info, None)
            .expect("failed to create command pool")
    }
}

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);

    unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .expect("failed to allocate command buffers")
    }
}

pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> SwapchainSupport {
    unsafe {
        let capabilities = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("renderer init: failed to get surface capabilities");

        let formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("renderer init: failed to get surface formats");

        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .expect("renderer init: failed to get present modes");

        SwapchainSupport {
            capabilities,
            formats,
            present_modes,
        }
    }
}

pub fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0])
}

pub fn choose_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .copied()
        .find(|&mode| mode == vk::PresentModeKHR::FIFO) // V-Sync
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn choose_extent(window: &Window, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    let size = window.inner_size();

    vk::Extent2D {
        width: size.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: size.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format) -> vk::ImageView {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device
            .create_image_view(&create_info, None)
            .expect("renderer init: failed to create image view")
    }
}

fn find_queue_families(
    instance: &Instance,
    device: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<QueueFamilyIndices> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };

    let mut graphics_family = None;
    let mut present_family = None;

    for (index, family) in families.iter().enumerate() {
        let index = index as u32;

        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_family = Some(index);
        }

        let present_supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(device, index, surface)
                .expect("renderer init: failed to get surface support")
        };

        if present_supported {
            present_family = Some(index);
        }

        if graphics_family.is_some() && present_family.is_some() {
            break;
        }
    }

    Some(QueueFamilyIndices {
        graphics_family: graphics_family?,
        present_family: present_family?,
    })
}

fn device_supports_swapchain(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(device)
            .expect("renderer init: failed to enumerate device extensions")
    };

    extensions.iter().any(|ext| {
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };

        name == ash::khr::swapchain::NAME
    })
}

fn validation_layer_supported(entry: &Entry) -> bool {
    let layers = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .expect("renderer init: failed to enumerate instance layers")
    };

    layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == VALIDATION_LAYER
    })
}

fn create_debug_messenger(loader: &debug_utils::Instance) -> vk::DebugUtilsMessengerEXT {
    let create_info = debug_messenger_create_info();

    unsafe {
        loader
            .create_debug_utils_messenger(&create_info, None)
            .expect("renderer init: failed to create debug messenger")
    }
}

fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback))
}

fn create_semaphores(device: &ash::Device, count: usize) -> Vec<vk::Semaphore> {
    let info = vk::SemaphoreCreateInfo::default();

    (0..count)
        .map(|_| unsafe {
            device
                .create_semaphore(&info, None)
                .expect("failed to create semaphore")
        })
        .collect()
}

fn create_fences(device: &ash::Device, count: usize) -> Vec<vk::Fence> {
    let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    (0..count)
        .map(|_| unsafe {
            device
                .create_fence(&info, None)
                .expect("failed to create fence")
        })
        .collect()
}

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("[Vulkan][{:?}] {:?}", ty, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("[Vulkan][{:?}] {:?}", ty, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::debug!("[Vulkan][{:?}] {:?}", ty, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::trace!("[Vulkan][{:?}] {:?}", ty, message);
        }
        _ => {
            log::debug!("[Vulkan][{:?}] {:?}", ty, message);
        }
    }

    vk::FALSE
}
