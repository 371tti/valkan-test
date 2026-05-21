use ash::vk;

use super::{
    RenderObject, RenderScene, Renderer,
    math::{ShadowProjection, for_each_render_object, shadow_projection},
    rendering::{clear_depth_attachment, render_area, set_viewport_and_scissor},
    uniforms::{ObjectPush, bytes_of},
};
use crate::renderer::MaterialId;

#[derive(Clone, Copy)]
pub(super) struct PreparedShadow {
    pub view_proj: [f32; 16],
    pub resolution: f32,
    pub bias: f32,
    pub strength: f32,
}

impl Renderer {
    pub(super) fn prepare_shadow(&self, scene: &RenderScene) -> PreparedShadow {
        let resolution = self.shadow_map.extent.width as f32;
        let projection = shadow_projection(scene, &self.assets, resolution);

        PreparedShadow {
            view_proj: projection.view_proj,
            resolution,
            bias: shadow_bias(projection, resolution),
            strength: 1.0,
        }
    }

    pub(super) fn record_shadow_map(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
    ) {
        self.transition_depth_image_layout(
            command_buffer,
            self.shadow_map.image(),
            self.shadow_map.layout(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        unsafe {
            set_viewport_and_scissor(&self.logical_device, command_buffer, self.shadow_map.extent);
            let depth_attachment = clear_depth_attachment(
                self.shadow_map.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::STORE,
            );
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area(self.shadow_map.extent))
                .layer_count(1)
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            self.logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline.pipeline.handle,
            );
            self.logical_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline.pipeline.layout,
                0,
                std::slice::from_ref(&self.shadow_scene_bindings.sets[frame_index]),
                &[],
            );

            let mut bound_material = None;
            for_each_render_object(scene, &self.assets, |object| {
                draw_shadow_object(self, command_buffer, object, &mut bound_material);
            });

            self.logical_device.cmd_end_rendering(command_buffer);
        }

        self.transition_depth_image_layout(
            command_buffer,
            self.shadow_map.image(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        );
        self.shadow_map
            .set_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }
}

fn shadow_bias(projection: ShadowProjection, resolution: f32) -> f32 {
    let texel_world = projection.radius * 2.0 / resolution.max(1.0);
    let world_bias = (texel_world * 1.2).clamp(0.006, 0.25);

    (world_bias / projection.depth_range.max(0.001)).clamp(0.00001, 0.001)
}

fn draw_shadow_object(
    renderer: &Renderer,
    command_buffer: vk::CommandBuffer,
    object: RenderObject,
    bound_material: &mut Option<MaterialId>,
) {
    let material = renderer.assets.material(object.material);
    if !material.casts_shadow() {
        return;
    }
    let Some(mesh) = renderer.assets.mesh(object.mesh) else {
        return;
    };

    unsafe {
        if *bound_material != Some(object.material)
            && let Some(texture_set) = renderer.assets.material_texture_set(object.material)
        {
            renderer.logical_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                renderer.shadow_pipeline.pipeline.layout,
                1,
                std::slice::from_ref(&texture_set),
                &[],
            );
            *bound_material = Some(object.material);
        }

        let push = ObjectPush::new(object, material);
        renderer.logical_device.cmd_push_constants(
            command_buffer,
            renderer.shadow_pipeline.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes_of(&push),
        );

        renderer.logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[mesh.vertex.buffer],
            &[0],
        );
        renderer.logical_device.cmd_bind_index_buffer(
            command_buffer,
            mesh.index.buffer,
            0,
            vk::IndexType::UINT32,
        );
        renderer
            .logical_device
            .cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
    }
}
