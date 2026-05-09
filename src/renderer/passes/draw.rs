use std::cmp::Ordering;

use ash::vk;

use super::{
    PipelineSlot, ReflectionProbe, RenderObject, RenderScene, Renderer, SceneBindings,
    math::{model_object, transform_point},
    uniforms::{ObjectPush, bytes_of, has_texture},
};
use crate::renderer::reflections::PreparedReflections;
use crate::renderer::{MaterialId, PipelineId};

#[derive(Clone, Copy)]
struct TransparentDraw {
    object: RenderObject,
    distance2: f32,
}

fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];

    dx * dx + dy * dy + dz * dz
}

fn draw_object(
    device: &ash::Device,
    pipelines: &[PipelineSlot],
    assets: &super::assets::GpuAssets,
    scene_bindings: &SceneBindings,
    command_buffer: vk::CommandBuffer,
    frame_index: usize,
    object: RenderObject,
    bound_pipeline: &mut Option<PipelineId>,
    bound_material: &mut Option<MaterialId>,
) {
    let Some(mesh) = assets.mesh(object.mesh) else {
        return;
    };
    let material = assets.material(object.material);
    let pipeline = if object.pipeline == PipelineId::LIT_MESH && material.is_translucent() {
        PipelineId::LIT_MESH_TRANSPARENT
    } else {
        object.pipeline
    };
    let Some(slot) = pipelines.get(pipeline.0) else {
        return;
    };

    unsafe {
        if *bound_pipeline != Some(pipeline) {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.handle,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.layout,
                0,
                std::slice::from_ref(&scene_bindings.sets[frame_index]),
                &[],
            );
            *bound_pipeline = Some(pipeline);
            *bound_material = None;
        }

        if *bound_material != Some(object.material) {
            if let Some(texture_set) = assets.material_texture_set(object.material) {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    slot.pipeline.layout,
                    1,
                    std::slice::from_ref(&texture_set),
                    &[],
                );
                *bound_material = Some(object.material);
            }
        }

        let push = ObjectPush {
            model: object.transform.matrix(),
            base_color: material.base_color,
            emissive_color: [
                material.emissive_color[0],
                material.emissive_color[1],
                material.emissive_color[2],
                0.0,
            ],
            material: [
                material.metallic,
                material.roughness,
                material.specular,
                material.ambient_occlusion,
            ],
            texture_flags: [
                has_texture(material.base_color_texture),
                has_texture(material.metallic_roughness_texture),
                has_texture(material.normal_texture),
                has_texture(material.occlusion_texture),
            ],
            texture_info: [
                has_texture(material.emissive_texture),
                material.normal_scale,
                material.occlusion_strength,
                material.alpha_cutoff,
            ],
        };
        device.cmd_push_constants(
            command_buffer,
            slot.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes_of(&push),
        );

        device.cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertex.buffer], &[0]);
        device.cmd_bind_index_buffer(command_buffer, mesh.index.buffer, 0, vk::IndexType::UINT32);
        device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
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

impl Renderer {
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

        let binding_base = frame_index * ReflectionProbe::FACE_COUNT;

        self.record_reflection_probe_faces(
            command_buffer,
            self.reflection_probe.image(),
            self.reflection_probe.layout(),
            self.reflection_probe.extent,
            &self.reflection_probe.face_views,
            self.reflection_probe.depth.view,
            &self.probe_scene_bindings,
            binding_base,
            scene,
            reflections.probe.center,
        );
        self.reflection_probe
            .set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
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
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.planar_reflection.extent.width as f32,
                height: self.planar_reflection.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.planar_reflection.extent,
            };

            self.logical_device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.logical_device
                .cmd_set_scissor(command_buffer, 0, &[scissor]);

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.planar_reflection.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.025, 0.035, 0.05, 1.0],
                    },
                });
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.planar_reflection.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.planar_reflection.extent,
                })
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
                reflections.planar.camera.eye,
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

    fn record_reflection_probe_faces(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        extent: vk::Extent2D,
        face_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        scene_bindings: &SceneBindings,
        binding_base: usize,
        scene: &RenderScene,
        probe_center: [f32; 3],
    ) {
        self.transition_color_image_layout(
            command_buffer,
            image,
            old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ReflectionProbe::FACE_COUNT as u32,
        );

        unsafe {
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };

            self.logical_device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.logical_device
                .cmd_set_scissor(command_buffer, 0, &[scissor]);

            for face in 0..ReflectionProbe::FACE_COUNT {
                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(face_views[face])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.04, 0.06, 0.09, 1.0],
                        },
                    });
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachment);

                self.logical_device
                    .cmd_begin_rendering(command_buffer, &rendering_info);
                self.draw_scene_geometry(
                    command_buffer,
                    scene_bindings,
                    binding_base + face,
                    scene,
                    probe_center,
                );
                self.logical_device.cmd_end_rendering(command_buffer);
            }
        }

        self.transition_color_image_layout(
            command_buffer,
            image,
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
        camera_eye: [f32; 3],
    ) {
        let mut bound_pipeline = None;
        let mut bound_material = None;
        let mut transparent = Vec::new();

        for object in &scene.objects {
            if !collect_transparent_draw(&self.assets, &mut transparent, camera_eye, *object) {
                draw_object(
                    &self.logical_device,
                    &self.pipelines,
                    &self.assets,
                    scene_bindings,
                    command_buffer,
                    binding_index,
                    *object,
                    &mut bound_pipeline,
                    &mut bound_material,
                );
            }
        }

        for model in &scene.models {
            let Some(gpu_model) = self.assets.model(model.model) else {
                continue;
            };

            for primitive in &gpu_model.primitives {
                let object = model_object(model.transform, model.pipeline, primitive);
                if !collect_transparent_draw(&self.assets, &mut transparent, camera_eye, object) {
                    draw_object(
                        &self.logical_device,
                        &self.pipelines,
                        &self.assets,
                        scene_bindings,
                        command_buffer,
                        binding_index,
                        object,
                        &mut bound_pipeline,
                        &mut bound_material,
                    );
                }
            }
        }

        transparent.sort_by(|a, b| {
            b.distance2
                .partial_cmp(&a.distance2)
                .unwrap_or(Ordering::Equal)
        });

        for item in transparent {
            draw_object(
                &self.logical_device,
                &self.pipelines,
                &self.assets,
                scene_bindings,
                command_buffer,
                binding_index,
                item.object,
                &mut bound_pipeline,
                &mut bound_material,
            );
        }
    }

    pub(super) fn record_draw_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_index: usize,
        scene: &RenderScene,
        reflections: PreparedReflections,
    ) {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin command buffer");

            self.record_reflection_probe(command_buffer, frame_index, scene, reflections);
            self.record_planar_reflection(command_buffer, frame_index, scene, reflections);

            let image = self.swapchain.images[image_index];
            let old_layout = self.swapchain.image_layouts[image_index];

            self.transition_image_layout(
                command_buffer,
                image,
                old_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.image_views[image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                });

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.extent.width as f32,
                height: self.swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            };

            self.logical_device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.logical_device
                .cmd_set_scissor(command_buffer, 0, &[scissor]);

            self.draw_scene_geometry(
                command_buffer,
                &self.scene_bindings,
                frame_index,
                scene,
                scene.camera.eye,
            );
            self.logical_device.cmd_end_rendering(command_buffer);

            self.transition_image_layout(
                command_buffer,
                image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.swapchain.image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;

            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("failed to end command buffer");
        }
    }
}
