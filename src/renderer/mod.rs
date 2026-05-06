use std::sync::Arc;

use ash::{
    Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::window::Window;

use crate::renderer::init::{create_command_buffers, create_swapchain};

/// CPUがGPU完了を待たずに先行して準備できるフレーム数(1..4程度が一般的)
const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    window_ref: Arc<Window>,

    entry: ash::Entry,
    instance: Instance,

    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,

    logical_device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    /// Swapchain image が描画可能になったことを **GPU** に通知するせまふぉ
    image_available_semaphores: Vec<vk::Semaphore>,
    /// 描画が終わったことを **Present(Window)** に通知するせまふぉ
    render_finished_semaphores: Vec<vk::Semaphore>,
    /// CPUがGPUに送ったコマンド(フレーム処理)の終了の確認/待ちに使うせまふぉ
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    swapchain_image_layouts: Vec<vk::ImageLayout>,

    debug_utils_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    framebuffer_resized: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Renderer {
    pub fn draw(&mut self) {
        unsafe {
            let fence = self.in_flight_fences[self.current_frame];

            self.logical_device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fence");

            if self.framebuffer_resized {
                self.framebuffer_resized = false;
                self.recreate_swapchain();
                return;
            }

            let image_available = self.image_available_semaphores[self.current_frame];

            let (image_index, suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                    return;
                }

                Err(err) => {
                    panic!("failed to acquire swapchain image: {err:?}");
                }
            };

            self.logical_device
                .reset_fences(&[fence])
                .expect("failed to reset fence");

            let command_buffer = self.command_buffers[image_index as usize];

            self.logical_device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            self.record_clear_command_buffer(command_buffer, image_index as usize);

            let render_finished = self.render_finished_semaphores[image_index as usize];

            let wait_semaphores = [image_available];
            let signal_semaphores = [render_finished];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.logical_device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .expect("failed to submit draw command buffer");

            let swapchains = [self.swapchain];
            let image_indices = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let present_suboptimal = match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                Ok(suboptimal) => suboptimal,

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.framebuffer_resized = true;
                    false
                }

                Err(err) => {
                    panic!("failed to present swapchain image: {err:?}");
                }
            };

            if suboptimal || present_suboptimal || self.framebuffer_resized {
                self.framebuffer_resized = false;
                self.recreate_swapchain();
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.framebuffer_resized = true;
    }

    fn recreate_swapchain(&mut self) {
        let size = self.window_ref.inner_size();

        if size.width == 0 || size.height == 0 {
            return;
        }

        unsafe {
            self.logical_device
                .device_wait_idle()
                .expect("failed to wait device idle");

            self.cleanup_swapchain();
        }

        let (
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        ) = create_swapchain(
            &self.window_ref,
            &self.instance,
            &self.logical_device,
            self.physical_device,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            self.queue_family_indices,
        );

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;

        self.swapchain_image_layouts =
            vec![vk::ImageLayout::UNDEFINED; self.swapchain_images.len()];

        self.command_buffers = create_command_buffers(
            &self.logical_device,
            self.command_pool,
            self.swapchain_images.len() as u32,
        );

        log::info!(
            "recreated swapchain: {}x{}",
            self.swapchain_extent.width,
            self.swapchain_extent.height
        );
    }

    fn record_clear_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();

            self.logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            let image = self.swapchain_images[image_index];

            self.transition_image_layout(
                command_buffer,
                image,
                self.swapchain_image_layouts[image_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let clear_color = vk::ClearColorValue {
                float32: [1.0, 0.0, 1.0, 1.0],
            };

            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            self.logical_device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[range],
            );

            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.swapchain_image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;

            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("failed to end command buffer");
        }
    }

    fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),

                (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),

                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                ),

                _ => panic!("unsupported layout transition"),
            };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.logical_device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    unsafe fn cleanup_swapchain(&mut self) {
        if !self.command_buffers.is_empty() {
            unsafe {
                self.logical_device
                    .free_command_buffers(self.command_pool, &self.command_buffers)
            };

            self.command_buffers.clear();
        }

        for image_view in self.swapchain_image_views.drain(..) {
            unsafe { self.logical_device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };

        self.swapchain_images.clear();
        self.swapchain_image_layouts.clear();
    }
}

pub mod r#drop;
pub mod init;
