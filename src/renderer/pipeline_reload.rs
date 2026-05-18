use ash::vk;

use super::{PipelineError, PipelineSlot, Renderer};

impl Renderer {
    pub(super) fn reload_pipeline_if_changed(&mut self) {
        for index in 0..self.pipelines.len() {
            let changed = {
                let slot = &mut self.pipelines[index];
                slot.hot_reload.changed(&slot.desc.shaders)
            };

            match changed {
                Ok(Some(stamp)) => {
                    self.wait_for_swapchain_idle();

                    match rebuild_slot(
                        &self.logical_device,
                        self.pipeline_cache,
                        self.swapchain.format,
                        &mut self.pipelines[index],
                    ) {
                        Ok(()) => {
                            self.pipelines[index].hot_reload.accept(stamp);
                            log::debug!(
                                "renderer: hot reloaded shader '{}'",
                                self.pipelines[index].desc.shaders.name
                            );
                        }
                        Err(err) => log::warn!("renderer: shader hot reload failed: {err}"),
                    }
                }
                Ok(None) => {}
                Err(err) => log::warn!("renderer: shader hot reload check failed: {err}"),
            }
        }

        let changed = self
            .shadow_pipeline
            .hot_reload
            .changed(&self.shadow_pipeline.desc.shaders);
        match changed {
            Ok(Some(stamp)) => {
                self.wait_for_swapchain_idle();
                match rebuild_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.shadow_pipeline,
                ) {
                    Ok(()) => {
                        self.shadow_pipeline.hot_reload.accept(stamp);
                        log::debug!("renderer: hot reloaded shader 'shadow_map'");
                    }
                    Err(err) => log::warn!("renderer: shadow shader hot reload failed: {err}"),
                }
            }
            Ok(None) => {}
            Err(err) => log::warn!("renderer: shadow shader hot reload check failed: {err}"),
        }

        let changed = self
            .post_pipeline
            .hot_reload
            .changed(&self.post_pipeline.desc.shaders);
        match changed {
            Ok(Some(stamp)) => {
                self.wait_for_swapchain_idle();
                match rebuild_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.post_pipeline,
                ) {
                    Ok(()) => {
                        self.post_pipeline.hot_reload.accept(stamp);
                        log::debug!("renderer: hot reloaded shader 'postprocess'");
                    }
                    Err(err) => log::warn!("renderer: postprocess shader hot reload failed: {err}"),
                }
            }
            Ok(None) => {}
            Err(err) => log::warn!("renderer: postprocess shader hot reload check failed: {err}"),
        }
    }

    pub(super) fn rebuild_pipelines(&mut self, format: vk::Format) -> Result<(), PipelineError> {
        for index in 0..self.pipelines.len() {
            rebuild_slot(
                &self.logical_device,
                self.pipeline_cache,
                format,
                &mut self.pipelines[index],
            )?;
        }
        rebuild_slot(
            &self.logical_device,
            self.pipeline_cache,
            format,
            &mut self.shadow_pipeline,
        )?;
        rebuild_slot(
            &self.logical_device,
            self.pipeline_cache,
            format,
            &mut self.post_pipeline,
        )?;

        Ok(())
    }
}

fn rebuild_slot(
    device: &ash::Device,
    cache: vk::PipelineCache,
    format: vk::Format,
    slot: &mut PipelineSlot,
) -> Result<(), PipelineError> {
    let pipeline = slot.desc.build(device, cache, format)?;

    unsafe {
        slot.pipeline.destroy(device);
    }

    slot.pipeline = pipeline;
    Ok(())
}
