use ash::{Device, vk};

use crate::protocol::{FrameId, FramebufferMetering, FramebufferReadbackOptions, NonZeroExtent};

use super::{
    VulkanError,
    buffer::{GpuBuffer, create_host_buffer, destroy_buffers},
};

pub(super) struct FramebufferReadbackState {
    options: FramebufferReadbackOptions,
    buffers: Vec<GpuBuffer>,
    copied: Vec<Option<FrameId>>,
    extent: NonZeroExtent,
    format: vk::Format,
    byte_size: vk::DeviceSize,
    transfer_src_supported: bool,
    frames_until_sample: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FramebufferReadbackConfig {
    pub(super) image_count: usize,
    pub(super) extent: NonZeroExtent,
    pub(super) format: vk::Format,
    pub(super) transfer_src_supported: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FramebufferReadbackCopy {
    pub(super) buffer: vk::Buffer,
    pub(super) extent: vk::Extent2D,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PreparedFramebufferReadback {
    pub(super) latest: Option<FramebufferReadbackSample>,
    pub(super) copy: Option<FramebufferReadbackCopy>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FramebufferReadbackSample {
    pub(super) frame_id: FrameId,
    pub(super) metering: FramebufferMetering,
}

#[derive(Clone, Copy)]
enum ColorOrder {
    Rgba,
    Bgra,
}

impl Default for FramebufferReadbackState {
    /// Creates an inert readback state until the app enables it through the renderer protocol.
    fn default() -> Self {
        Self {
            options: FramebufferReadbackOptions::default(),
            buffers: Vec::new(),
            copied: Vec::new(),
            extent: NonZeroExtent::new(1, 1).expect("literal readback extent is non-zero"),
            format: vk::Format::UNDEFINED,
            byte_size: 0,
            transfer_src_supported: false,
            frames_until_sample: 0,
        }
    }
}

impl FramebufferReadbackState {
    /// Stores the app-requested readback policy and resets the frame cadence.
    pub(super) fn set_options(&mut self, options: FramebufferReadbackOptions) {
        tracing::info!(
            enabled = options.enabled(),
            interval_frames = options.interval_frames(),
            "configured framebuffer readback"
        );
        self.options = options;
        self.frames_until_sample = 0;
    }

    /// Rebuilds host readback buffers for the current swapchain image set.
    pub(super) fn configure(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        config: FramebufferReadbackConfig,
    ) -> Result<(), VulkanError> {
        self.destroy_buffers(device);
        self.extent = config.extent;
        self.format = config.format;
        self.transfer_src_supported = config.transfer_src_supported;
        self.frames_until_sample = 0;
        self.byte_size = byte_size(config.extent, config.format).unwrap_or(0);

        if !self.can_read(config.image_count) {
            tracing::trace!(
                enabled = self.options.enabled(),
                transfer_src_supported = self.transfer_src_supported,
                image_count = config.image_count,
                byte_size = self.byte_size,
                format = ?self.format,
                "framebuffer readback disabled for this swapchain"
            );
            return Ok(());
        }

        self.buffers = create_readback_buffers(
            device,
            memory_properties,
            config.image_count,
            self.byte_size,
        )?;
        self.copied = vec![None; config.image_count];

        tracing::info!(
            width = config.extent.width(),
            height = config.extent.height(),
            image_count = config.image_count,
            bytes_per_frame = self.byte_size,
            "created framebuffer readback buffers"
        );
        Ok(())
    }

    /// Reads the previous sample for this image and decides whether the new frame should copy.
    pub(super) fn prepare_frame(
        &mut self,
        device: &Device,
        image_index: u32,
    ) -> Result<PreparedFramebufferReadback, VulkanError> {
        let latest = self.read_previous_sample(device, image_index)?;
        let copy = self
            .should_sample()
            .then(|| self.copy_for_image(image_index))
            .transpose()?;

        Ok(PreparedFramebufferReadback { latest, copy })
    }

    /// Marks that command recording has scheduled a copy into this image's host buffer.
    pub(super) fn mark_copy_recorded(&mut self, image_index: u32, frame_id: FrameId) {
        if let Some(copied) = self.copied.get_mut(image_index as usize) {
            *copied = Some(frame_id);
        }
    }

    /// Destroys all readback buffers before device teardown or swapchain replacement.
    pub(super) fn destroy(&mut self, device: &Device) {
        self.destroy_buffers(device);
        self.byte_size = 0;
        self.format = vk::Format::UNDEFINED;
        self.transfer_src_supported = false;
    }

    /// Returns whether this swapchain can satisfy the current app readback request.
    fn can_read(&self, image_count: usize) -> bool {
        self.options.enabled()
            && self.transfer_src_supported
            && image_count > 0
            && color_order(self.format).is_some()
            && self.byte_size > 0
    }

    /// Maps and samples the buffer last written for the acquired swapchain image.
    fn read_previous_sample(
        &mut self,
        device: &Device,
        image_index: u32,
    ) -> Result<Option<FramebufferReadbackSample>, VulkanError> {
        let index = image_index as usize;
        let Some(frame_id) = self.copied.get(index).copied().flatten() else {
            return Ok(None);
        };

        let Some(buffer) = self.buffers.get(index) else {
            return Ok(None);
        };
        let metering = buffer.read_bytes(device, self.byte_size, |bytes| {
            sample_metering(bytes, self.extent, self.format)
        })?;
        if let Some(copied) = self.copied.get_mut(index) {
            *copied = None;
        }

        Ok(metering
            .valid()
            .then_some(FramebufferReadbackSample { frame_id, metering }))
    }

    /// Returns true at the configured interval without allowing unbounded copy work.
    fn should_sample(&mut self) -> bool {
        if !self.can_read(self.buffers.len()) {
            return false;
        }
        if self.frames_until_sample > 0 {
            self.frames_until_sample -= 1;
            return false;
        }

        self.frames_until_sample = self.options.interval_frames().saturating_sub(1);
        true
    }

    /// Returns the copy target for one acquired swapchain image.
    fn copy_for_image(&self, image_index: u32) -> Result<FramebufferReadbackCopy, VulkanError> {
        let index = image_index as usize;
        let buffer = self
            .buffers
            .get(index)
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index,
                count: self.buffers.len(),
            })?;

        Ok(FramebufferReadbackCopy {
            buffer: buffer.handle(),
            extent: vk::Extent2D {
                width: self.extent.width(),
                height: self.extent.height(),
            },
        })
    }

    /// Releases current host buffers and clears per-image copy flags.
    fn destroy_buffers(&mut self, device: &Device) {
        destroy_buffers(device, std::mem::take(&mut self.buffers));
        self.copied.clear();
    }
}

/// Creates one host transfer buffer for each swapchain image.
fn create_readback_buffers(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    count: usize,
    byte_size: vk::DeviceSize,
) -> Result<Vec<GpuBuffer>, VulkanError> {
    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        match create_host_buffer(
            device,
            memory_properties,
            vk::BufferUsageFlags::TRANSFER_DST,
            byte_size,
        ) {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => {
                destroy_buffers(device, buffers);
                return Err(error);
            }
        }
    }

    Ok(buffers)
}

/// Returns the tightly-packed byte size needed for one framebuffer copy.
fn byte_size(extent: NonZeroExtent, format: vk::Format) -> Option<vk::DeviceSize> {
    let bytes_per_pixel = bytes_per_pixel(format)? as vk::DeviceSize;
    Some(extent.width() as vk::DeviceSize * extent.height() as vk::DeviceSize * bytes_per_pixel)
}

/// Records a tightly-packed copy from a transfer-src color image into a host buffer.
pub(super) fn record_image_to_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    copy: FramebufferReadbackCopy,
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
        .image_extent(vk::Extent3D {
            width: copy.extent.width,
            height: copy.extent.height,
            depth: 1,
        });

    // Safety: graph barriers put `image` in TRANSFER_SRC_OPTIMAL before this pass, and `buffer`
    // is a host transfer destination sized for this extent.
    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            copy.buffer,
            std::slice::from_ref(&region),
        );
    }
}

/// Samples copied framebuffer pixels with the old center-weighted AutoCamera metering rule.
fn sample_metering(bytes: &[u8], extent: NonZeroExtent, format: vk::Format) -> FramebufferMetering {
    let Some(order) = color_order(format) else {
        return FramebufferMetering::default();
    };
    if bytes.len() < 4 {
        return FramebufferMetering::default();
    }

    let width = extent.width() as usize;
    let height = extent.height() as usize;
    let step_x = (width / 96).max(1);
    let step_y = (height / 54).max(1);
    let mut totals = MeteringTotals::default();

    for y in (step_y / 2..height).step_by(step_y) {
        for x in (step_x / 2..width).step_by(step_x) {
            let index = (y * width + x) * 4;
            if index + 3 >= bytes.len() {
                continue;
            }
            totals.add_sample(decode_color(&bytes[index..index + 4], order), x, y, extent);
        }
    }

    totals.finish()
}

#[derive(Default)]
struct MeteringTotals {
    luma_sum: f32,
    center_luma_sum: f32,
    weight_sum: f32,
    center_weight_sum: f32,
    highlight_weight: f32,
    color_sum: [f32; 3],
}

impl MeteringTotals {
    /// Adds one display-space sample with center weighting and highlight classification.
    fn add_sample(&mut self, color: [f32; 3], x: usize, y: usize, extent: NonZeroExtent) {
        let luma = luminance(color);
        let nx = ((x as f32 + 0.5) / extent.width() as f32) * 2.0 - 1.0;
        let ny = ((y as f32 + 0.5) / extent.height() as f32) * 2.0 - 1.0;
        let center = (1.0 - (nx * nx + ny * ny).sqrt()).clamp(0.0, 1.0);
        let center2 = center * center;
        let center4 = center2 * center2;
        let in_spot = nx.abs() < 0.30 && ny.abs() < 0.30;
        let weight = 0.35 + center2 * 2.0 + center4 * 6.0 + if in_spot { 3.0 } else { 0.0 };

        self.luma_sum += luma * weight;
        self.weight_sum += weight;
        for (sum, channel) in self.color_sum.iter_mut().zip(color) {
            *sum += channel * weight;
        }
        if nx.abs() < 0.34 && ny.abs() < 0.34 {
            self.center_luma_sum += luma * weight;
            self.center_weight_sum += weight;
        }
        if luma > 0.82 {
            self.highlight_weight += weight;
        }
    }

    /// Converts accumulated weighted samples into a protocol metering packet.
    fn finish(self) -> FramebufferMetering {
        if self.weight_sum <= f32::EPSILON {
            return FramebufferMetering::default();
        }

        let average_luminance = self.luma_sum / self.weight_sum;
        let center_luminance = if self.center_weight_sum > 0.0 {
            self.center_luma_sum / self.center_weight_sum
        } else {
            average_luminance
        };
        let raw_highlight_fraction = self.highlight_weight / self.weight_sum;
        let average_color = self.color_sum.map(|channel| channel / self.weight_sum);
        let center_dark_priority =
            ((average_luminance - center_luminance) / average_luminance.max(0.002)).clamp(0.0, 1.0);
        let highlight_fraction =
            raw_highlight_fraction * smooth_mix(1.0, 0.45, center_dark_priority);
        let chroma = max_channel(average_color) / average_luminance.max(0.001);
        let confidence = smoothstep(0.02, 0.10, average_luminance)
            * (1.0 - smoothstep(0.42, 0.88, highlight_fraction))
            * smoothstep(1.01, 1.35, chroma);

        FramebufferMetering::new(
            average_luminance,
            center_luminance,
            highlight_fraction,
            average_color,
            confidence,
        )
    }
}

/// Returns the byte width of formats the app-side metering protocol understands.
fn bytes_per_pixel(format: vk::Format) -> Option<usize> {
    color_order(format).map(|_| 4)
}

/// Returns how four-byte pixels map into RGB channels for supported framebuffer formats.
fn color_order(format: vk::Format) -> Option<ColorOrder> {
    match format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(ColorOrder::Rgba),
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(ColorOrder::Bgra),
        _ => None,
    }
}

/// Decodes one sRGB framebuffer pixel into linear RGB for metering math.
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

/// Converts sRGB channel values to linear light before luminance calculations.
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Returns Rec. 709 luminance for linear RGB values.
fn luminance(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

/// Returns the largest RGB channel used to identify useful chroma signal.
fn max_channel(color: [f32; 3]) -> f32 {
    color[0].max(color[1]).max(color[2])
}

/// Returns a Hermite smoothstep value for metering confidence.
fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);

    t * t * (3.0 - 2.0 * t)
}

/// Linearly interpolates two scalar metering values.
fn smooth_mix(a: f32, b: f32, weight: f32) -> f32 {
    a + (b - a) * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framebuffer_metering_prefers_center_luminance() {
        let extent = NonZeroExtent::new(8, 8).expect("literal extent is non-zero");
        let mut bytes = vec![16; extent.width() as usize * extent.height() as usize * 4];
        for y in 3..5 {
            for x in 3..5 {
                let index = (y * extent.width() as usize + x) * 4;
                bytes[index..index + 4].copy_from_slice(&[240, 240, 240, 255]);
            }
        }

        let metering = sample_metering(&bytes, extent, vk::Format::R8G8B8A8_SRGB);

        assert!(metering.valid());
        assert!(metering.center_luminance() > metering.average_luminance());
    }

    #[test]
    fn framebuffer_metering_protects_dark_center_subject() {
        let extent = NonZeroExtent::new(16, 16).expect("literal extent is non-zero");
        let mut bytes = vec![245; extent.width() as usize * extent.height() as usize * 4];
        for y in 5..11 {
            for x in 5..11 {
                let index = (y * extent.width() as usize + x) * 4;
                bytes[index..index + 4].copy_from_slice(&[28, 28, 28, 255]);
            }
        }

        let metering = sample_metering(&bytes, extent, vk::Format::R8G8B8A8_SRGB);

        assert!(metering.valid());
        assert!(metering.center_luminance() < metering.average_luminance() * 0.20);
        assert!(metering.highlight_fraction() < 0.30);
    }
}
