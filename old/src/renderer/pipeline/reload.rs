use std::time::SystemTime;

use ash::vk;

use super::{PipelineError, PipelineSlot, Renderer, assets::ShadowMap};

impl Renderer {
    pub(super) fn reload_pipeline_if_changed(&mut self) {
        let mut changes = Vec::new();

        for index in 0..self.pipelines.len() {
            collect_pipeline_change(
                &mut changes,
                PipelineTarget::Model(index),
                &mut self.pipelines[index],
                None,
            );
        }
        collect_pipeline_change(
            &mut changes,
            PipelineTarget::OpaqueShadow,
            &mut self.opaque_shadow_pipeline,
            Some("opaque_shadow_map"),
        );
        collect_pipeline_change(
            &mut changes,
            PipelineTarget::CutoutShadow,
            &mut self.cutout_shadow_pipeline,
            Some("cutout_shadow_map"),
        );
        collect_pipeline_change(
            &mut changes,
            PipelineTarget::TransparentShadow,
            &mut self.transparent_shadow_pipeline,
            Some("transparent_shadow_opacity"),
        );
        collect_pipeline_change(
            &mut changes,
            PipelineTarget::Post,
            &mut self.post_pipeline,
            Some("postprocess"),
        );

        for change in changes {
            self.wait_for_swapchain_idle();
            let result = match change.target {
                PipelineTarget::Model(index) => rebuild_changed_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.pipelines[index],
                    change.stamp,
                ),
                PipelineTarget::OpaqueShadow => rebuild_changed_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.opaque_shadow_pipeline,
                    change.stamp,
                ),
                PipelineTarget::CutoutShadow => rebuild_changed_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.cutout_shadow_pipeline,
                    change.stamp,
                ),
                PipelineTarget::TransparentShadow => rebuild_changed_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    ShadowMap::OPACITY_FORMAT,
                    &mut self.transparent_shadow_pipeline,
                    change.stamp,
                ),
                PipelineTarget::Post => rebuild_changed_slot(
                    &self.logical_device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    &mut self.post_pipeline,
                    change.stamp,
                ),
            };

            match result {
                Ok(()) => log::debug!("renderer: hot reloaded shader '{}'", change.name),
                Err(err) => log::warn!(
                    "renderer: shader '{}' hot reload failed: {err}",
                    change.name
                ),
            }
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
            &mut self.opaque_shadow_pipeline,
        )?;
        rebuild_slot(
            &self.logical_device,
            self.pipeline_cache,
            format,
            &mut self.cutout_shadow_pipeline,
        )?;
        rebuild_slot(
            &self.logical_device,
            self.pipeline_cache,
            ShadowMap::OPACITY_FORMAT,
            &mut self.transparent_shadow_pipeline,
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

#[derive(Clone, Copy)]
enum PipelineTarget {
    Model(usize),
    OpaqueShadow,
    CutoutShadow,
    TransparentShadow,
    Post,
}

struct PipelineChange {
    target: PipelineTarget,
    stamp: Option<SystemTime>,
    name: String,
}

fn collect_pipeline_change(
    changes: &mut Vec<PipelineChange>,
    target: PipelineTarget,
    slot: &mut PipelineSlot,
    name: Option<&str>,
) {
    let name = name.unwrap_or(slot.desc.shaders.name).to_owned();

    match slot.hot_reload.changed(&slot.desc.shaders) {
        Ok(Some(stamp)) => changes.push(PipelineChange {
            target,
            stamp,
            name,
        }),
        Ok(None) => {}
        Err(err) => log::warn!("renderer: shader '{name}' hot reload check failed: {err}"),
    }
}

fn rebuild_changed_slot(
    device: &ash::Device,
    cache: vk::PipelineCache,
    format: vk::Format,
    slot: &mut PipelineSlot,
    stamp: Option<SystemTime>,
) -> Result<(), PipelineError> {
    rebuild_slot(device, cache, format, slot)?;
    slot.hot_reload.accept(stamp);
    Ok(())
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
