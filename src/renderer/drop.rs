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

            self.assets.destroy(&self.logical_device);

            for slot in &mut self.pipelines {
                slot.pipeline.destroy(&self.logical_device);
            }

            self.scene_bindings.destroy(&self.logical_device);
            self.logical_device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            self.logical_device
                .destroy_command_pool(self.command_pool, None);

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
