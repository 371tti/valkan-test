impl Drop for super::Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.logical_device.device_wait_idle();

            for frame in self.sync.frames.drain(..) {
                self.logical_device
                    .destroy_semaphore(frame.image_available, None);
                self.logical_device
                    .destroy_fence(frame.in_flight_fence, None);
            }

            self.cleanup_swapchain();

            self.logical_device.destroy_pipeline(self.pipeline, None);
            self.logical_device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.logical_device.destroy_command_pool(self.command_pool, None);

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
