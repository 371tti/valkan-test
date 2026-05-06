impl Drop for super::Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.logical_device.device_wait_idle();

            for semaphore in self.image_available_semaphores.drain(..) {
                self.logical_device.destroy_semaphore(semaphore, None);
            }

            for semaphore in self.render_finished_semaphores.drain(..) {
                self.logical_device.destroy_semaphore(semaphore, None);
            }

            for fence in self.in_flight_fences.drain(..) {
                self.logical_device.destroy_fence(fence, None);
            }

            self.logical_device.destroy_command_pool(self.command_pool, None);

            for image_view in self.swapchain_image_views.drain(..) {
                self.logical_device.destroy_image_view(image_view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.logical_device.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);

            if let (Some(loader), Some(messenger)) =
                (&self.debug_utils_loader, self.debug_messenger)
            {
                loader.destroy_debug_utils_messenger(messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}
