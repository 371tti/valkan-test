use ash::vk;

use super::{
    REFLECTION_PROBE_SIZE, Renderer,
    assets::{PlanarReflectionTarget, ReflectionProbe, SceneRenderTarget},
    init::{
        SwapchainCreateContext, SwapchainStateContext, create_swapchain, create_swapchain_state,
    },
};

impl Renderer {
    /// Window resize: schedule a swapchain rebuild on the next drawable frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.schedule_rebuild();
    }

    pub(super) fn recreate_swapchain(&mut self) {
        let size = self.window_ref.inner_size();

        if size.width == 0 || size.height == 0 {
            log::debug!("renderer: rebuild postponed (zero window size)");
            return;
        }

        let old_format = self.swapchain.format;

        self.wait_for_swapchain_idle();

        unsafe {
            self.cleanup_swapchain();
        }

        let swapchain = create_swapchain(SwapchainCreateContext {
            window: &self.window_ref,
            device: &self.logical_device,
            physical_device: self.physical_device,
            surface_loader: &self.surface_loader,
            surface: self.surface,
            swapchain_loader: &self.swapchain_loader,
            indices: self.queue_family_indices,
        });

        if swapchain.format != old_format {
            unsafe { self.reflection_probe.destroy(&self.logical_device) };
            unsafe { self.fallback_reflection_probe.destroy(&self.logical_device) };
            unsafe { self.planar_reflection.destroy(&self.logical_device) };
            unsafe {
                self.fallback_planar_reflection
                    .destroy(&self.logical_device)
            };
            self.reflection_probe = ReflectionProbe::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain.format,
                REFLECTION_PROBE_SIZE,
            );
            self.fallback_reflection_probe = ReflectionProbe::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain.format,
                1,
            );
            self.planar_reflection = PlanarReflectionTarget::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain.format,
                self.planar_reflection.extent,
            );
            self.fallback_planar_reflection = PlanarReflectionTarget::new(
                &self.instance,
                &self.logical_device,
                self.physical_device,
                self.command_pool,
                self.graphics_queue,
                swapchain.format,
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            );
            self.update_scene_image_descriptors();
            self.rebuild_pipelines(swapchain.format)
                .expect("renderer: failed to rebuild pipelines for swapchain format");
        }

        self.swapchain = create_swapchain_state(SwapchainStateContext {
            instance: &self.instance,
            device: &self.logical_device,
            physical_device: self.physical_device,
            command_pool: self.command_pool,
            graphics_queue: self.graphics_queue,
            swapchain,
        });
        self.camera_meter.resize(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            super::metering::CameraMeterConfig {
                image_count: self.swapchain.images.len(),
                extent: self.swapchain.extent,
                format: self.swapchain.format,
                transfer_src_supported: self.swapchain.transfer_src_supported,
            },
        );
        unsafe {
            self.scene_target.destroy(&self.logical_device);
        }
        self.scene_target = SceneRenderTarget::new(
            &self.instance,
            &self.logical_device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            self.swapchain.format,
            self.swapchain.extent,
        );
        self.update_scene_image_descriptors();
        log::trace!(
            "recreated swapchain: {}x{}, format: {:?}",
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.swapchain.format,
        );
    }

    pub(super) fn schedule_rebuild(&mut self) {
        self.needs_swapchain_rebuild = true;
    }

    pub(super) fn wait_for_swapchain_idle(&self) {
        let fences: Vec<vk::Fence> = self
            .sync
            .frames
            .iter()
            .map(|frame| frame.in_flight_fence)
            .collect();

        unsafe {
            self.logical_device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("failed to wait for in-flight fences");

            self.logical_device
                .queue_wait_idle(self.present_queue)
                .expect("failed to wait present queue idle");
        }
    }

    pub(super) unsafe fn cleanup_swapchain(&mut self) {
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

        unsafe { self.swapchain.depth.destroy(&self.logical_device) };

        for image_view in self.swapchain.image_views.drain(..) {
            unsafe { self.logical_device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None)
        };

        self.swapchain.images.clear();
        self.swapchain.images_in_flight.clear();
        self.swapchain.image_layouts.clear();
    }
}
