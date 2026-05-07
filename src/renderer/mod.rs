use std::sync::Arc;

use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::window::Window;

use crate::renderer::init::{create_command_buffers, create_semaphores, create_swapchain};

/// CPUがGPU完了を待たずに先行して準備できるフレーム数(1..4程度が一般的)
const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct SwapchainState {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
    command_buffers: Vec<vk::CommandBuffer>,
    image_layouts: Vec<vk::ImageLayout>,
    images_in_flight: Vec<vk::Fence>,
    render_finished_semaphores: Vec<vk::Semaphore>,
}

struct SyncState {
    image_available_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

impl SyncState {
    fn advance_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();
    }
}

pub struct Renderer {
    _entry: Entry,
    window_ref: Arc<Window>,

    config: RendererConfig,
    instance: Instance,

    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,

    logical_device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: SwapchainState,

    command_pool: vk::CommandPool,
    sync: SyncState,

    debug_utils_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    needs_swapchain_rebuild_fast: bool,
    needs_swapchain_rebuild_full: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RendererConfig {
    preferred_present_mode: Option<vk::PresentModeKHR>,
    preferred_surface_format: Option<vk::SurfaceFormatKHR>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Renderer {
    /// 描画 高速パス: スケジュール済み再作成をここで実行してからreturn
    pub fn draw(&mut self) {
        unsafe {
            // スケジュール済みSwapchain再作成を実行
            if self.needs_swapchain_rebuild_full {
                self.needs_swapchain_rebuild_full = false;
                self.recreate_swapchain_full();
                return;
            }

            if self.needs_swapchain_rebuild_fast {
                self.needs_swapchain_rebuild_fast = false;
                self.recreate_swapchain_fast();
                return;
            }

            let frame_index = self.sync.current_frame;
            let fence = self.sync.in_flight_fences[frame_index];

            self.logical_device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fence");

            let image_available = self.sync.image_available_semaphores[frame_index];

            let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!(
                        "renderer: acquire_next_image returned OUT_OF_DATE_KHR, scheduling fast rebuild"
                    );
                    self.needs_swapchain_rebuild_fast = true;
                    return;
                }

                Err(err) => {
                    log::error!(
                        "renderer: failed to acquire swapchain image: {err:?}, scheduling full rebuild"
                    );
                    self.needs_swapchain_rebuild_full = true;
                    return;
                }
            };

            if self.swapchain.images_in_flight[image_index as usize] != vk::Fence::null() {
                self.logical_device
                    .wait_for_fences(
                        &[self.swapchain.images_in_flight[image_index as usize]],
                        true,
                        u64::MAX,
                    )
                    .expect("failed to wait for image fence");
            }

            self.swapchain.images_in_flight[image_index as usize] = fence;

            self.logical_device
                .reset_fences(&[fence])
                .expect("failed to reset fence");

            let command_buffer = self.swapchain.command_buffers[image_index as usize];

            self.logical_device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            self.record_clear_command_buffer(command_buffer, image_index as usize);

            let render_finished = self.swapchain.render_finished_semaphores[image_index as usize];

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

            let swapchains = [self.swapchain.swapchain];
            let image_indices = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                Ok(_suboptimal) => {}

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!(
                        "renderer: queue_present returned OUT_OF_DATE_KHR, scheduling fast rebuild"
                    );
                    self.needs_swapchain_rebuild_fast = true;
                }

                Err(err) => {
                    log::error!("renderer: queue_present failed: {err:?}, scheduling full rebuild");
                    self.needs_swapchain_rebuild_full = true;
                }
            }

            self.sync.advance_frame();
        }
    }

    /// ウィンドウリサイズ: 高速再作成をスケジュール
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.needs_swapchain_rebuild_fast = true;
    }

    /// PresentMode設定変更: 完全再作成をスケジュール
    pub fn set_present_mode(&mut self, mode: vk::PresentModeKHR) {
        self.config.preferred_present_mode = Some(mode);
        log::debug!("renderer: set_present_mode scheduled");
        self.needs_swapchain_rebuild_full = true;
    }

    /// SurfaceFormat設定変更: 完全再作成をスケジュール
    pub fn set_surface_format(&mut self, format: vk::SurfaceFormatKHR) {
        self.config.preferred_surface_format = Some(format);
        log::debug!("renderer: set_surface_format scheduled");
        self.needs_swapchain_rebuild_full = true;
    }

    /// 高速再作成（ウィンドウリサイズ用）: 現在のサイズで即座に再作成
    fn recreate_swapchain_fast(&mut self) {
        let size = self.window_ref.inner_size();

        if size.width == 0 || size.height == 0 {
            log::warn!("renderer: fast rebuild skipped (zero window size)");
            return;
        }

        self.wait_for_swapchain_idle();

        unsafe {
            self.cleanup_swapchain();
        }

        // 現在のサポート情報をそのまま使用（高速）
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
            self.config.preferred_surface_format,
            self.config.preferred_present_mode,
        );

        let image_count = swapchain_images.len();

        self.swapchain = SwapchainState {
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views,
            format: swapchain_format,
            extent: swapchain_extent,
            command_buffers: create_command_buffers(
                &self.logical_device,
                self.command_pool,
                image_count as u32,
            ),
            image_layouts: vec![vk::ImageLayout::UNDEFINED; image_count],
            images_in_flight: vec![vk::Fence::null(); image_count],
            render_finished_semaphores: create_semaphores(&self.logical_device, image_count),
        };

        log::trace!(
            "fast recreated swapchain: {}x{}, format: {:?}, present_mode: {:?}",
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.swapchain.format,
            self.config.preferred_present_mode,
        );
    }

    /// 完全再作成（query_swapchain_support を再度呼ぶ）: 設定変更やエラー復帰用
    fn recreate_swapchain_full(&mut self) {
        let size = self.window_ref.inner_size();

        if size.width == 0 || size.height == 0 {
            log::warn!("renderer: full rebuild skipped (zero window size)");
            return;
        }

        self.wait_for_swapchain_idle();

        unsafe {
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
            self.config.preferred_surface_format,
            self.config.preferred_present_mode,
        );

        let image_count = swapchain_images.len();

        self.swapchain = SwapchainState {
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views,
            format: swapchain_format,
            extent: swapchain_extent,
            command_buffers: create_command_buffers(
                &self.logical_device,
                self.command_pool,
                image_count as u32,
            ),
            image_layouts: vec![vk::ImageLayout::UNDEFINED; image_count],
            images_in_flight: vec![vk::Fence::null(); image_count],
            render_finished_semaphores: create_semaphores(&self.logical_device, image_count),
        };

        log::trace!(
            "full recreated swapchain: {}x{}, format: {:?}, present_mode: {:?}",
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.swapchain.format,
            self.config.preferred_present_mode,
        );
    }

    fn record_clear_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            let image = self.swapchain.images[image_index];

            self.transition_image_layout(
                command_buffer,
                image,
                self.swapchain.image_layouts[image_index],
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

            self.swapchain.image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;

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

    fn wait_for_swapchain_idle(&self) {
        unsafe {
            self.logical_device
                .wait_for_fences(&self.sync.in_flight_fences, true, u64::MAX)
                .expect("failed to wait for in-flight fences");

            self.logical_device
                .queue_wait_idle(self.present_queue)
                .expect("failed to wait present queue idle");
        }
    }

    unsafe fn cleanup_swapchain(&mut self) {
        if !self.swapchain.command_buffers.is_empty() {
            unsafe {
                self.logical_device
                    .free_command_buffers(self.command_pool, &self.swapchain.command_buffers)
            };

            self.swapchain.command_buffers.clear();
        }

        for semaphore in self.swapchain.render_finished_semaphores.drain(..) {
            unsafe { self.logical_device.destroy_semaphore(semaphore, None) };
        }

        for image_view in self.swapchain.image_views.drain(..) {
            unsafe { self.logical_device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None)
        };

        self.swapchain.images.clear();
        self.swapchain.image_layouts.clear();
        self.swapchain.images_in_flight.clear();
    }
}

pub mod r#drop;
pub mod init;
