use ash::vk;

use super::{
    ReflectionProbe, RenderScene, Renderer, math::probe_binding_index, uniforms::SceneUniform,
};

impl Renderer {
    /// Draws one frame and performs any scheduled swapchain rebuild first.
    pub fn draw(&mut self, scene: &RenderScene) {
        if !self.has_drawable_extent() {
            return;
        }

        unsafe {
            if self.needs_swapchain_rebuild {
                self.needs_swapchain_rebuild = false;
                self.recreate_swapchain();
                return;
            }

            self.reload_pipeline_if_changed();

            let frame_index = self.sync.current_frame;
            let fence = self.sync.frames[frame_index].in_flight_fence;

            self.logical_device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("failed to wait for fence");

            self.ensure_reflection_targets(scene);

            let shadow = self.prepare_shadow(scene);
            let reflections = self.prepare_reflections(scene);
            self.shadow_scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::shadow(scene, shadow),
            );
            if reflections.probe.enabled {
                let face = self.reflection_probe_face_cursor % ReflectionProbe::FACE_COUNT;
                let binding_index = probe_binding_index(frame_index, face);
                self.probe_scene_bindings.update(
                    &self.logical_device,
                    binding_index,
                    &SceneUniform::reflection_probe_face(scene, reflections, shadow, face),
                );
            }
            self.planar_scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::planar_reflection(
                    scene,
                    self.planar_reflection.extent,
                    reflections,
                    shadow,
                ),
            );

            self.scene_bindings.update(
                &self.logical_device,
                frame_index,
                &SceneUniform::new(scene, self.swapchain.extent, reflections, shadow),
            );

            let image_available = self.sync.frames[frame_index].image_available;

            let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,

                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("renderer: acquire_next_image returned OUT_OF_DATE_KHR");
                    self.schedule_rebuild();
                    return;
                }

                Err(err) => {
                    log::error!("renderer: failed to acquire swapchain image: {err:?}");
                    self.schedule_rebuild();
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
            self.camera_meter
                .read_image(&self.logical_device, image_index as usize);
            self.swapchain.images_in_flight[image_index as usize] = fence;

            self.logical_device
                .reset_fences(&[fence])
                .expect("failed to reset fence");

            let command_buffer = self.swapchain.command_buffers[image_index as usize];

            self.logical_device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            self.record_draw_command_buffer(
                command_buffer,
                image_index as usize,
                frame_index,
                scene,
                reflections,
            );

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
                    log::debug!("renderer: queue_present returned OUT_OF_DATE_KHR");
                    self.schedule_rebuild();
                }

                Err(err) => {
                    log::error!("renderer: queue_present failed: {err:?}");
                    self.schedule_rebuild();
                }
            }

            self.sync.advance_frame();
        }
    }
}
