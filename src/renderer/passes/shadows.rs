use ash::vk;

use super::{
    RenderObject, RenderScene, Renderer, SHADOW_CASCADE_COUNT, SHADOW_CASCADE_GRID,
    math::{
        ShadowProjection, cascaded_shadow_projection_for_aspect, distance_squared,
        for_each_render_object, transform_point, transformed_radius,
    },
    rendering::{clear_depth_attachment, render_area},
    uniforms::{ObjectPush, bytes_of},
};
use crate::renderer::MaterialId;

#[derive(Clone, Copy)]
pub(super) struct ShadowCascade {
    pub view_proj: [f32; 16],
    pub atlas: [f32; 4],
    pub split_depth: f32,
    pub bias: f32,
    pub radius: f32,
    pub center: [f32; 3],
}

impl ShadowCascade {
    fn disabled() -> Self {
        Self {
            view_proj: [0.0; 16],
            atlas: [0.0; 4],
            split_depth: 0.0,
            bias: 0.0,
            radius: 0.0,
            center: [0.0; 3],
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PreparedShadow {
    pub cascades: [ShadowCascade; SHADOW_CASCADE_COUNT],
    pub camera_pos: [f32; 3],
    pub camera_forward: [f32; 3],
    pub atlas_size: f32,
    pub cascade_count: u32,
    pub strength: f32,
}

impl Renderer {
    pub(super) fn prepare_shadow(&self, scene: &RenderScene) -> PreparedShadow {
        let atlas_size = self.shadow_map.extent.width as f32;
        let aspect =
            self.swapchain.extent.width as f32 / self.swapchain.extent.height.max(1) as f32;
        let cascade_size = atlas_size / SHADOW_CASCADE_GRID.max(1) as f32;
        let projections =
            cascaded_shadow_projection_for_aspect(scene, &self.assets, aspect, atlas_size);
        let mut cascades = [ShadowCascade::disabled(); SHADOW_CASCADE_COUNT];

        for (index, cascade) in cascades.iter_mut().enumerate() {
            let projection = projections[index];
            *cascade = ShadowCascade {
                view_proj: projection.view_proj,
                atlas: cascade_atlas(index),
                split_depth: projection.split_depth,
                bias: shadow_bias(projection, cascade_size),
                radius: projection.radius,
                center: projection.center,
            };
        }

        PreparedShadow {
            cascades,
            camera_pos: scene.camera.eye,
            camera_forward: camera_forward(scene),
            atlas_size,
            cascade_count: SHADOW_CASCADE_COUNT as u32,
            strength: 1.0,
        }
    }

    pub(super) fn record_shadow_map(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
        shadow: PreparedShadow,
        cascade_updates: [bool; SHADOW_CASCADE_COUNT],
    ) {
        if !cascade_updates.iter().any(|&update| update) {
            return;
        }

        self.transition_depth_image_layout(
            command_buffer,
            self.shadow_map.image(),
            self.shadow_map.layout(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        unsafe {
            for cascade_index in 0..shadow.cascade_count as usize {
                if !cascade_updates[cascade_index] {
                    continue;
                }

                let cascade = shadow.cascades[cascade_index];
                let rect = cascade_rect(cascade_index, self.shadow_map.extent);
                let depth_attachment = clear_depth_attachment(
                    self.shadow_map.view,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::AttachmentStoreOp::STORE,
                );
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(rect)
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                self.logical_device
                    .cmd_begin_rendering(command_buffer, &rendering_info);
                set_cascade_viewport_and_scissor(&self.logical_device, command_buffer, rect);
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
                    if shadow_caster_intersects_cascade(self, object, cascade) {
                        draw_shadow_object(
                            self,
                            command_buffer,
                            object,
                            cascade_index,
                            &mut bound_material,
                        );
                    }
                });

                self.logical_device.cmd_end_rendering(command_buffer);
            }
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

fn camera_forward(scene: &RenderScene) -> [f32; 3] {
    let direction = [
        scene.camera.target[0] - scene.camera.eye[0],
        scene.camera.target[1] - scene.camera.eye[1],
        scene.camera.target[2] - scene.camera.eye[2],
    ];
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt()
            .max(f32::EPSILON);

    [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ]
}

fn cascade_atlas(index: usize) -> [f32; 4] {
    let grid = SHADOW_CASCADE_GRID.max(1) as f32;
    let scale = 1.0 / grid;
    let col = (index as u32 % SHADOW_CASCADE_GRID) as f32;
    let row = (index as u32 / SHADOW_CASCADE_GRID) as f32;

    [col * scale, row * scale, scale, scale]
}

fn cascade_rect(index: usize, extent: vk::Extent2D) -> vk::Rect2D {
    let grid = SHADOW_CASCADE_GRID.max(1);
    let cascade_width = (extent.width / grid).max(1);
    let cascade_height = (extent.height / grid).max(1);
    let col = index as u32 % grid;
    let row = index as u32 / grid;

    vk::Rect2D {
        offset: vk::Offset2D {
            x: (col * cascade_width) as i32,
            y: (row * cascade_height) as i32,
        },
        extent: vk::Extent2D {
            width: cascade_width,
            height: cascade_height,
        },
    }
}

fn set_cascade_viewport_and_scissor(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    rect: vk::Rect2D,
) {
    let viewport = vk::Viewport {
        x: rect.offset.x as f32,
        y: rect.offset.y as f32,
        width: rect.extent.width as f32,
        height: rect.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    unsafe {
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[rect]);
    }
}

fn shadow_bias(projection: ShadowProjection, resolution: f32) -> f32 {
    let texel_world = projection.radius * 2.0 / resolution.max(1.0);
    let world_bias = (texel_world * 1.1).clamp(0.006, 0.22);

    (world_bias / projection.depth_range.max(0.001)).clamp(0.00001, 0.001)
}

fn shadow_caster_intersects_cascade(
    renderer: &Renderer,
    object: RenderObject,
    cascade: ShadowCascade,
) -> bool {
    let Some(mesh) = renderer.assets.mesh(object.mesh) else {
        return false;
    };
    let center = transform_point(object.transform.matrix(), mesh.center);
    let radius = transformed_radius(object.transform, mesh.radius).max(0.01);
    let limit = cascade.radius * 1.22 + radius + 4.0;

    distance_squared(center, cascade.center) <= limit * limit
}

fn draw_shadow_object(
    renderer: &Renderer,
    command_buffer: vk::CommandBuffer,
    object: RenderObject,
    cascade_index: usize,
    bound_material: &mut Option<MaterialId>,
) {
    let material = renderer.assets.material(object.material);
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

        let mut push = ObjectPush::new(object, material);
        push.emissive_color[3] = cascade_index as f32;
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
