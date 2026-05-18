use ash::vk;

use super::{
    assets::{GpuBuffer, find_memory_type},
    scene::CameraMetering,
};

const METER_MAX_WIDTH: u32 = 320;
const METER_FRAME_INTERVAL: u32 = 3;

pub(super) struct CameraMeter {
    buffers: Vec<GpuBuffer>,
    copied: Vec<bool>,
    source_extent: vk::Extent2D,
    sample_extent: vk::Extent2D,
    format: vk::Format,
    byte_size: vk::DeviceSize,
    readback_image: vk::Image,
    readback_memory: vk::DeviceMemory,
    readback_layout: vk::ImageLayout,
    blit_filter: vk::Filter,
    latest: CameraMetering,
    enabled: bool,
    use_blit: bool,
    frames_until_sample: u32,
}

pub(super) struct CameraMeterConfig {
    pub image_count: usize,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub transfer_src_supported: bool,
}

impl CameraMeter {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        config: CameraMeterConfig,
    ) -> Self {
        let mut meter = Self {
            buffers: Vec::new(),
            copied: Vec::new(),
            source_extent: config.extent,
            sample_extent: metering_extent(config.extent),
            format: config.format,
            byte_size: 0,
            readback_image: vk::Image::null(),
            readback_memory: vk::DeviceMemory::null(),
            readback_layout: vk::ImageLayout::UNDEFINED,
            blit_filter: vk::Filter::NEAREST,
            latest: CameraMetering::default(),
            enabled: false,
            use_blit: false,
            frames_until_sample: 0,
        };
        meter.recreate(instance, device, physical_device, config);
        meter
    }

    pub fn resize(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        config: CameraMeterConfig,
    ) {
        unsafe { self.destroy(device) };
        self.recreate(instance, device, physical_device, config);
    }

    fn recreate(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        config: CameraMeterConfig,
    ) {
        self.source_extent = config.extent;
        self.sample_extent = metering_extent(config.extent);
        self.format = config.format;
        self.latest = CameraMetering::default();
        self.frames_until_sample = 0;
        self.copied = vec![false; config.image_count];

        let Some(bytes_per_pixel) = bytes_per_pixel(config.format) else {
            log::warn!(
                "active camera disabled: unsupported swapchain format {:?}",
                config.format
            );
            self.enabled = false;
            self.byte_size = 0;
            return;
        };

        self.byte_size = self.sample_extent.width as vk::DeviceSize
            * self.sample_extent.height as vk::DeviceSize
            * bytes_per_pixel as vk::DeviceSize;
        self.enabled =
            config.transfer_src_supported && config.image_count > 0 && self.byte_size > 0;

        if !self.enabled {
            log::warn!("active camera disabled: swapchain images cannot be copied");
            return;
        }

        self.use_blit = format_supports_blit(instance, physical_device, config.format);
        self.blit_filter = if format_supports_linear_blit(instance, physical_device, config.format)
        {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };

        if self.use_blit {
            let (image, memory) = create_meter_image(
                instance,
                device,
                physical_device,
                config.format,
                self.sample_extent,
            );
            self.readback_image = image;
            self.readback_memory = memory;
            self.readback_layout = vk::ImageLayout::UNDEFINED;
        } else {
            log::warn!(
                "active camera: format {:?} cannot be blitted; using cropped readback fallback",
                config.format
            );
        }

        self.buffers = (0..config.image_count)
            .map(|_| {
                GpuBuffer::host_transfer_dst(instance, device, physical_device, self.byte_size)
            })
            .collect();

        log::debug!(
            "active camera: readback {}x{} every {} frames ({:.1} KiB)",
            self.sample_extent.width,
            self.sample_extent.height,
            METER_FRAME_INTERVAL,
            self.byte_size as f32 / 1024.0
        );
    }

    pub fn should_sample(&mut self) -> bool {
        if !self.enabled {
            return false;
        }

        if self.frames_until_sample > 0 {
            self.frames_until_sample -= 1;
            return false;
        }

        self.frames_until_sample = METER_FRAME_INTERVAL.saturating_sub(1);
        true
    }

    pub fn latest(&self) -> CameraMetering {
        self.latest
    }

    pub fn read_image(&mut self, device: &ash::Device, image_index: usize) {
        if !self.enabled {
            self.latest = CameraMetering::default();
            return;
        }
        if !self.copied.get(image_index).copied().unwrap_or(false) {
            return;
        }

        let Some(buffer) = self.buffers.get(image_index) else {
            return;
        };
        self.latest = unsafe {
            buffer.with_mapped_bytes(device, self.byte_size, |bytes| {
                sample_metering(bytes, self.sample_extent, self.format)
            })
        };

        if let Some(copied) = self.copied.get_mut(image_index) {
            *copied = false;
        }
    }

    pub fn record_copy(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        image_index: usize,
    ) {
        if !self.enabled {
            return;
        }

        let Some(buffer) = self.buffers.get(image_index).map(|buffer| buffer.buffer) else {
            return;
        };

        if self.use_blit && self.readback_image != vk::Image::null() {
            self.record_downsample(device, command_buffer, image);
            record_image_to_buffer(
                device,
                command_buffer,
                self.readback_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                self.sample_extent,
                vk::Offset3D { x: 0, y: 0, z: 0 },
            );
        } else {
            record_image_to_buffer(
                device,
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                self.sample_extent,
                centered_offset(self.source_extent, self.sample_extent),
            );
        }

        if let Some(copied) = self.copied.get_mut(image_index) {
            *copied = true;
        }
    }

    fn record_downsample(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
    ) {
        self.transition_readback(
            device,
            command_buffer,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let blit = vk::ImageBlit::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.source_extent.width as i32,
                    y: self.source_extent.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.sample_extent.width as i32,
                    y: self.sample_extent.height as i32,
                    z: 1,
                },
            ]);

        unsafe {
            device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                self.blit_filter,
            );
        }

        self.transition_readback(
            device,
            command_buffer,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
    }

    fn transition_readback(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        new_layout: vk::ImageLayout,
    ) {
        if self.readback_layout == new_layout {
            return;
        }

        let (src_stage, src_access, dst_stage, dst_access) =
            meter_transition(self.readback_layout, new_layout);
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(self.readback_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.readback_image)
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
        self.readback_layout = new_layout;
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for buffer in &mut self.buffers {
            unsafe { buffer.destroy(device) };
        }
        self.buffers.clear();
        self.copied.clear();

        if self.readback_image != vk::Image::null() {
            unsafe { device.destroy_image(self.readback_image, None) };
            self.readback_image = vk::Image::null();
        }
        if self.readback_memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.readback_memory, None) };
            self.readback_memory = vk::DeviceMemory::null();
        }

        self.enabled = false;
        self.use_blit = false;
        self.byte_size = 0;
        self.readback_layout = vk::ImageLayout::UNDEFINED;
        self.latest = CameraMetering::default();
    }
}

fn create_meter_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    extent: vk::Extent2D,
) -> (vk::Image, vk::DeviceMemory) {
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
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("renderer: failed to create active camera meter image")
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
            .expect("renderer: failed to allocate active camera meter image memory")
    };

    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("renderer: failed to bind active camera meter image memory")
    };

    (image, memory)
}

fn record_image_to_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    buffer: vk::Buffer,
    extent: vk::Extent2D,
    image_offset: vk::Offset3D,
) {
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
        .image_offset(image_offset)
        .image_extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            layout,
            buffer,
            std::slice::from_ref(&region),
        );
    }
}

fn meter_transition(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> (
    vk::PipelineStageFlags2,
    vk::AccessFlags2,
    vk::PipelineStageFlags2,
    vk::AccessFlags2,
) {
    match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        ),
        _ => panic!("unsupported active camera meter transition: {old_layout:?} -> {new_layout:?}"),
    }
}

fn metering_extent(source: vk::Extent2D) -> vk::Extent2D {
    let source_width = source.width.max(1);
    let source_height = source.height.max(1);
    let width = source_width.min(METER_MAX_WIDTH).max(1);
    let height = ((width as u64 * source_height as u64 + source_width as u64 / 2)
        / source_width as u64)
        .clamp(1, source_height as u64) as u32;

    vk::Extent2D { width, height }
}

fn centered_offset(source: vk::Extent2D, extent: vk::Extent2D) -> vk::Offset3D {
    vk::Offset3D {
        x: ((source.width.saturating_sub(extent.width)) / 2) as i32,
        y: ((source.height.saturating_sub(extent.height)) / 2) as i32,
        z: 0,
    }
}

fn format_supports_blit(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> bool {
    let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    props
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
}

fn format_supports_linear_blit(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> bool {
    let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    props
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
}

fn sample_metering(bytes: &[u8], extent: vk::Extent2D, format: vk::Format) -> CameraMetering {
    let Some(order) = color_order(format) else {
        return CameraMetering::default();
    };
    if extent.width == 0 || extent.height == 0 || bytes.len() < 4 {
        return CameraMetering::default();
    }

    let width = extent.width as usize;
    let height = extent.height as usize;
    let step_x = (width / 96).max(1);
    let step_y = (height / 54).max(1);

    let mut luma_sum = 0.0;
    let mut center_luma_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut center_weight_sum = 0.0;
    let mut highlight_weight = 0.0;
    let mut color_sum = [0.0; 3];

    for y in (step_y / 2..height).step_by(step_y) {
        for x in (step_x / 2..width).step_by(step_x) {
            let index = (y * width + x) * 4;
            if index + 3 >= bytes.len() {
                continue;
            }

            let color = decode_color(&bytes[index..index + 4], order);
            let luma = luminance(color);
            let nx = ((x as f32 + 0.5) / extent.width as f32) * 2.0 - 1.0;
            let ny = ((y as f32 + 0.5) / extent.height as f32) * 2.0 - 1.0;
            let center = (1.0 - (nx * nx + ny * ny).sqrt()).clamp(0.0, 1.0);
            let weight = 1.0 + center * center * 3.0;

            luma_sum += luma * weight;
            weight_sum += weight;
            for channel in 0..3 {
                color_sum[channel] += color[channel] * weight;
            }

            if nx.abs() < 0.34 && ny.abs() < 0.34 {
                center_luma_sum += luma * weight;
                center_weight_sum += weight;
            }
            if luma > 0.82 {
                highlight_weight += weight;
            }
        }
    }

    if weight_sum <= f32::EPSILON {
        return CameraMetering::default();
    }

    let average_luminance = luma_sum / weight_sum;
    let center_luminance = if center_weight_sum > 0.0 {
        center_luma_sum / center_weight_sum
    } else {
        average_luminance
    };
    let highlight_fraction = highlight_weight / weight_sum;
    let average_color = color_sum.map(|channel| channel / weight_sum);
    let chroma = max_channel(average_color) / average_luminance.max(0.001);
    let confidence = smoothstep(0.02, 0.10, average_luminance)
        * (1.0 - smoothstep(0.42, 0.88, highlight_fraction))
        * smoothstep(1.01, 1.35, chroma);

    CameraMetering {
        valid: true,
        average_luminance,
        center_luminance,
        highlight_fraction,
        average_color,
        white_balance_confidence: confidence.clamp(0.0, 1.0),
    }
}

#[derive(Clone, Copy)]
enum ColorOrder {
    Rgba,
    Bgra,
}

fn bytes_per_pixel(format: vk::Format) -> Option<usize> {
    color_order(format).map(|_| 4)
}

fn color_order(format: vk::Format) -> Option<ColorOrder> {
    match format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(ColorOrder::Rgba),
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(ColorOrder::Bgra),
        _ => None,
    }
}

fn decode_color(pixel: &[u8], order: ColorOrder) -> [f32; 3] {
    let (r, g, b) = match order {
        ColorOrder::Rgba => (pixel[0], pixel[1], pixel[2]),
        ColorOrder::Bgra => (pixel[2], pixel[1], pixel[0]),
    };

    [
        srgb_to_linear(r as f32 / 255.0),
        srgb_to_linear(g as f32 / 255.0),
        srgb_to_linear(b as f32 / 255.0),
    ]
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn luminance(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

fn max_channel(color: [f32; 3]) -> f32 {
    color[0].max(color[1]).max(color[2])
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);

    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_weighted_metering_prefers_the_middle() {
        let extent = vk::Extent2D {
            width: 8,
            height: 8,
        };
        let mut bytes = vec![16; extent.width as usize * extent.height as usize * 4];
        for y in 3..5 {
            for x in 3..5 {
                let index = (y * extent.width as usize + x) * 4;
                bytes[index..index + 4].copy_from_slice(&[240, 240, 240, 255]);
            }
        }

        let metering = sample_metering(&bytes, extent, vk::Format::R8G8B8A8_SRGB);

        assert!(metering.valid);
        assert!(metering.center_luminance > metering.average_luminance);
    }

    #[test]
    fn readback_extent_keeps_aspect_but_caps_width() {
        let extent = metering_extent(vk::Extent2D {
            width: 2400,
            height: 1600,
        });

        assert_eq!(extent.width, METER_MAX_WIDTH);
        assert_eq!(extent.height, 213);
    }
}
