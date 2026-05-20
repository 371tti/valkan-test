use std::cmp::Ordering;

use ash::vk;

use super::{
    Camera, PipelineSlot, ReflectionProbe, RenderObject, RenderScene, Renderer, SceneBindings,
    math::{
        distance_squared, for_each_visible_render_object, reflection_probe_face_axes,
        transform_point,
    },
    rendering::{
        clear_color_attachment, clear_depth_attachment, render_area, set_viewport_and_scissor,
    },
    uniforms::{ObjectPush, bytes_of},
};
use crate::renderer::reflections::PreparedReflections;
use crate::renderer::{MaterialId, PipelineId, RenderDebugMode};

#[derive(Clone, Copy)]
struct TransparentDraw {
    object: RenderObject,
    distance2: f32,
}

struct DrawContext<'a> {
    device: &'a ash::Device,
    pipelines: &'a [PipelineSlot],
    assets: &'a super::assets::GpuAssets,
    scene_bindings: &'a SceneBindings,
    command_buffer: vk::CommandBuffer,
    frame_index: usize,
    debug_mode: RenderDebugMode,
}

#[derive(Default)]
struct DrawBindings {
    pipeline: Option<PipelineId>,
    material: Option<MaterialId>,
}

fn draw_object(context: &DrawContext<'_>, object: RenderObject, bindings: &mut DrawBindings) {
    let Some(mesh) = context.assets.mesh(object.mesh) else {
        return;
    };
    let material = context.assets.material(object.material);
    let pipeline = material_pipeline(object.pipeline, material, context.debug_mode);
    let Some(slot) = context.pipelines.get(pipeline.0) else {
        return;
    };

    unsafe {
        if bindings.pipeline != Some(pipeline) {
            context.device.cmd_bind_pipeline(
                context.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.handle,
            );
            context.device.cmd_bind_descriptor_sets(
                context.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.layout,
                0,
                std::slice::from_ref(&context.scene_bindings.sets[context.frame_index]),
                &[],
            );
            bindings.pipeline = Some(pipeline);
            bindings.material = None;
        }

        if bindings.material != Some(object.material)
            && let Some(texture_set) = context.assets.material_texture_set(object.material)
        {
            context.device.cmd_bind_descriptor_sets(
                context.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.layout,
                1,
                std::slice::from_ref(&texture_set),
                &[],
            );
            bindings.material = Some(object.material);
        }

        let push = ObjectPush::new(object, material);
        context.device.cmd_push_constants(
            context.command_buffer,
            slot.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes_of(&push),
        );

        context.device.cmd_bind_vertex_buffers(
            context.command_buffer,
            0,
            &[mesh.vertex.buffer],
            &[0],
        );
        context.device.cmd_bind_index_buffer(
            context.command_buffer,
            mesh.index.buffer,
            0,
            vk::IndexType::UINT32,
        );
        context
            .device
            .cmd_draw_indexed(context.command_buffer, mesh.index_count, 1, 0, 0, 0);
    }
}

fn material_pipeline(
    requested: PipelineId,
    material: crate::renderer::Material,
    debug_mode: RenderDebugMode,
) -> PipelineId {
    if debug_mode == RenderDebugMode::Wireframe {
        return PipelineId::LIT_MESH_WIREFRAME;
    }
    if requested != PipelineId::LIT_MESH {
        return requested;
    }

    match (material.is_translucent(), material.double_sided) {
        (true, true) => PipelineId::LIT_MESH_TRANSPARENT_DOUBLE_SIDED,
        (true, false) => PipelineId::LIT_MESH_TRANSPARENT,
        (false, true) => PipelineId::LIT_MESH_DOUBLE_SIDED,
        (false, false) => PipelineId::LIT_MESH,
    }
}

fn collect_transparent_draw(
    assets: &super::assets::GpuAssets,
    transparent: &mut Vec<TransparentDraw>,
    camera_eye: [f32; 3],
    object: RenderObject,
) -> bool {
    if !assets.material(object.material).is_translucent() {
        return false;
    }

    let center = assets
        .mesh(object.mesh)
        .map(|mesh| transform_point(object.transform.matrix(), mesh.center))
        .unwrap_or(object.transform.translation);
    transparent.push(TransparentDraw {
        object,
        distance2: distance_squared(camera_eye, center),
    });
    true
}

fn reflection_probe_camera(
    scene: &RenderScene,
    reflections: PreparedReflections,
    face: usize,
) -> Camera {
    let (direction, up) = reflection_probe_face_axes(face);
    let center = reflections.probe.center;

    Camera {
        eye: center,
        target: [
            center[0] + direction[0],
            center[1] + direction[1],
            center[2] + direction[2],
        ],
        up,
        fov_y: 90.0_f32.to_radians(),
        near: 0.05,
        far: scene.camera.far.max(5000.0),
    }
}

impl Renderer {
    fn record_scene_target(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
    ) {
        self.transition_color_image_layout(
            command_buffer,
            self.scene_target.image(),
            self.scene_target.color_layout(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );
        self.scene_target
            .set_color_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        self.transition_depth_image_layout(
            command_buffer,
            self.scene_target.depth.image(),
            self.scene_target.depth_layout(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );
        self.scene_target
            .set_depth_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        unsafe {
            let color_attachment = clear_color_attachment(
                self.scene_target.view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                [0.0, 0.0, 0.0, 1.0],
            );
            let depth_attachment = clear_depth_attachment(
                self.scene_target.depth.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::STORE,
            );
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area(self.scene_target.extent))
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            set_viewport_and_scissor(
                &self.logical_device,
                command_buffer,
                self.scene_target.extent,
            );
            self.draw_scene_geometry(
                command_buffer,
                &self.scene_bindings,
                frame_index,
                scene,
                scene.camera,
                self.scene_target.extent,
                scene.debug_mode,
            );
            self.logical_device.cmd_end_rendering(command_buffer);
        }

        self.transition_color_image_layout(
            command_buffer,
            self.scene_target.image(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        );
        self.scene_target
            .set_color_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.transition_depth_image_layout(
            command_buffer,
            self.scene_target.depth.image(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        );
        self.scene_target
            .set_depth_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    fn record_postprocess(&self, command_buffer: vk::CommandBuffer, frame_index: usize) {
        unsafe {
            self.logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.post_pipeline.pipeline.handle,
            );
            self.logical_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.post_pipeline.pipeline.layout,
                0,
                std::slice::from_ref(&self.scene_bindings.sets[frame_index]),
                &[],
            );
            self.logical_device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }

    fn finish_swapchain_image(&mut self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let image = self.swapchain.images[image_index];

        if self.camera_meter.should_sample() {
            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            self.camera_meter
                .record_copy(&self.logical_device, command_buffer, image, image_index);
            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        } else {
            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        }

        self.swapchain.image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    fn record_reflection_probe(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
        reflections: PreparedReflections,
    ) {
        if !reflections.probe.enabled {
            return;
        }

        let face = self.reflection_probe_face_cursor % ReflectionProbe::FACE_COUNT;
        let binding_index = frame_index * ReflectionProbe::FACE_COUNT + face;

        self.record_reflection_probe_face(command_buffer, binding_index, face, scene, reflections);
        self.reflection_probe
            .set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.reflection_probe_face_cursor =
            (self.reflection_probe_face_cursor + 1) % ReflectionProbe::FACE_COUNT;
    }

    fn record_planar_reflection(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
        reflections: PreparedReflections,
    ) {
        if !reflections.planar.enabled {
            return;
        }

        self.transition_color_image_layout(
            command_buffer,
            self.planar_reflection.image(),
            self.planar_reflection.layout(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );

        unsafe {
            set_viewport_and_scissor(
                &self.logical_device,
                command_buffer,
                self.planar_reflection.extent,
            );
            let color_attachment = clear_color_attachment(
                self.planar_reflection.view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                [0.0, 0.0, 0.0, 1.0],
            );
            let depth_attachment = clear_depth_attachment(
                self.planar_reflection.depth.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::DONT_CARE,
            );
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area(self.planar_reflection.extent))
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.draw_scene_geometry(
                command_buffer,
                &self.planar_scene_bindings,
                frame_index,
                scene,
                reflections.planar.camera,
                self.planar_reflection.extent,
                RenderDebugMode::Default,
            );
            self.logical_device.cmd_end_rendering(command_buffer);
        }

        self.transition_color_image_layout(
            command_buffer,
            self.planar_reflection.image(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        );
        self.planar_reflection
            .set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    fn record_reflection_probe_face(
        &self,
        command_buffer: vk::CommandBuffer,
        binding_index: usize,
        face: usize,
        scene: &RenderScene,
        reflections: PreparedReflections,
    ) {
        let probe = &self.reflection_probe;
        let face = face.min(ReflectionProbe::FACE_COUNT - 1);
        let face_view = probe.face_views[face];

        self.transition_color_image_layout(
            command_buffer,
            probe.image(),
            probe.layout(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ReflectionProbe::FACE_COUNT as u32,
        );

        unsafe {
            set_viewport_and_scissor(&self.logical_device, command_buffer, probe.extent);
            let color_attachment = clear_color_attachment(
                face_view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                [0.0, 0.0, 0.0, 1.0],
            );
            let depth_attachment = clear_depth_attachment(
                probe.depth.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::DONT_CARE,
            );
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area(probe.extent))
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.draw_scene_geometry(
                command_buffer,
                &self.probe_scene_bindings,
                binding_index,
                scene,
                reflection_probe_camera(scene, reflections, face),
                probe.extent,
                RenderDebugMode::Default,
            );
            self.logical_device.cmd_end_rendering(command_buffer);
        }

        self.transition_color_image_layout(
            command_buffer,
            probe.image(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ReflectionProbe::FACE_COUNT as u32,
        );
    }

    fn draw_scene_geometry(
        &self,
        command_buffer: vk::CommandBuffer,
        scene_bindings: &SceneBindings,
        binding_index: usize,
        scene: &RenderScene,
        camera: Camera,
        extent: vk::Extent2D,
        debug_mode: RenderDebugMode,
    ) {
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let context = DrawContext {
            device: &self.logical_device,
            pipelines: &self.pipelines,
            assets: &self.assets,
            scene_bindings,
            command_buffer,
            frame_index: binding_index,
            debug_mode,
        };
        let mut bindings = DrawBindings::default();
        let mut transparent = Vec::new();

        for_each_visible_render_object(scene, &self.assets, camera, aspect, |object| {
            if !collect_transparent_draw(&self.assets, &mut transparent, camera.eye, object) {
                draw_object(&context, object, &mut bindings);
            }
        });

        transparent.sort_by(|a, b| {
            b.distance2
                .partial_cmp(&a.distance2)
                .unwrap_or(Ordering::Equal)
        });

        for item in transparent {
            draw_object(&context, item.object, &mut bindings);
        }
    }

    pub(super) fn record_draw_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_index: usize,
        scene: &RenderScene,
        reflections: PreparedReflections,
        shadow: super::shadows::PreparedShadow,
        pass_updates: super::PassUpdates,
    ) {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            if pass_updates.shadow_map() {
                self.record_shadow_map(
                    command_buffer,
                    frame_index,
                    scene,
                    shadow,
                    pass_updates.shadow_cascades,
                );
            }
            if pass_updates.reflection_probe {
                self.record_reflection_probe(command_buffer, frame_index, scene, reflections);
            }
            if pass_updates.planar_reflection {
                self.record_planar_reflection(command_buffer, frame_index, scene, reflections);
            }
            self.record_scene_target(command_buffer, frame_index, scene);
            let image = self.swapchain.images[image_index];
            let old_layout = self.swapchain.image_layouts[image_index];

            self.transition_image_layout(
                command_buffer,
                image,
                old_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let color_attachment = clear_color_attachment(
                self.swapchain.image_views[image_index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                [0.0, 0.0, 0.0, 1.0],
            );
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area(self.swapchain.extent))
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            set_viewport_and_scissor(&self.logical_device, command_buffer, self.swapchain.extent);
            self.record_postprocess(command_buffer, frame_index);
            self.logical_device.cmd_end_rendering(command_buffer);

            self.finish_swapchain_image(command_buffer, image_index);

            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("failed to end command buffer");
        }
    }
}
