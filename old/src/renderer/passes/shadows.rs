use ash::vk;

use super::{
    PipelineSlot, RenderObject, RenderScene, Renderer,
    math::{
        ShadowProjection, for_each_render_object, shadow_projection, transform_point,
        transformed_radius,
    },
    rendering::{
        clear_color_attachment, clear_depth_attachment, load_depth_attachment, render_area,
        set_viewport_and_scissor,
    },
    uniforms::{ObjectPush, bytes_of},
};
use crate::renderer::{Material, MaterialId, MeshId};

const SHADOW_FLAG_DEPTH: u32 = 1;
const SHADOW_FLAG_TRANSPARENT: u32 = 2;
const SHADOW_SCISSOR_PADDING: i32 = 8;

#[derive(Clone, Copy)]
pub(super) struct PreparedShadow {
    pub view_proj: [f32; 16],
    pub view: [f32; 16],
    pub resolution: f32,
    pub radius: f32,
    pub near: f32,
    pub far: f32,
    pub bias: f32,
    pub strength: f32,
    pub flags: u32,
}

impl Renderer {
    pub(super) fn prepare_shadow(&self, scene: &RenderScene) -> PreparedShadow {
        let resolution = self.shadow_map.extent.width as f32;
        let aspect =
            self.swapchain.extent.width as f32 / self.swapchain.extent.height.max(1) as f32;
        let projection = shadow_projection(scene, &self.assets, aspect, resolution);

        let mut shadow = PreparedShadow {
            view_proj: projection.view_proj,
            view: projection.view,
            resolution,
            radius: projection.radius,
            near: projection.near,
            far: projection.far,
            bias: shadow_bias(projection, resolution),
            strength: 1.0,
            flags: 0,
        };
        shadow.flags = shadow_flags_for_scene(self, scene, shadow);
        shadow.strength = if shadow.flags == 0 { 0.0 } else { 1.0 };

        shadow
    }

    pub(super) fn record_shadow_map(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        scene: &RenderScene,
        shadow: PreparedShadow,
    ) {
        let batches = collect_shadow_casters(self, scene, shadow);
        if batches.is_empty() {
            return;
        }

        self.transition_depth_image_layout(
            command_buffer,
            self.shadow_map.image(),
            self.shadow_map.layout(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        unsafe {
            set_viewport_and_scissor(&self.logical_device, command_buffer, self.shadow_map.extent);
            let clear_depth = clear_depth_attachment(
                self.shadow_map.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::STORE,
            );
            let depth_rendering = vk::RenderingInfo::default()
                .render_area(render_area(self.shadow_map.extent))
                .layer_count(1)
                .depth_attachment(&clear_depth);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &depth_rendering);
            set_shadow_scissor(&self.logical_device, command_buffer, batches.depth_scissor);
            draw_shadow_pass(
                self,
                command_buffer,
                frame_index,
                &self.opaque_shadow_pipeline,
                ShadowPassKind::Opaque,
                &batches.opaque,
            );
            draw_shadow_pass(
                self,
                command_buffer,
                frame_index,
                &self.cutout_shadow_pipeline,
                ShadowPassKind::Cutout,
                &batches.cutout,
            );
            self.logical_device.cmd_end_rendering(command_buffer);

            if batches.transparent.is_empty() {
                self.transition_depth_image_layout(
                    command_buffer,
                    self.shadow_map.image(),
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                );
                self.shadow_map
                    .set_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                return;
            }

            self.transition_color_image_layout(
                command_buffer,
                self.shadow_map.opacity_image(),
                self.shadow_map.opacity_layout(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                1,
            );
            let opacity_attachment = clear_color_attachment(
                self.shadow_map.opacity_view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                [0.0, 0.0, 0.0, 0.0],
            );
            let opacity_attachments = [opacity_attachment];
            let load_depth = load_depth_attachment(
                self.shadow_map.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AttachmentStoreOp::STORE,
            );
            let opacity_rendering = vk::RenderingInfo::default()
                .render_area(render_area(self.shadow_map.extent))
                .layer_count(1)
                .color_attachments(&opacity_attachments)
                .depth_attachment(&load_depth);

            self.logical_device
                .cmd_begin_rendering(command_buffer, &opacity_rendering);
            set_shadow_scissor(
                &self.logical_device,
                command_buffer,
                batches.transparent_scissor,
            );
            draw_shadow_pass(
                self,
                command_buffer,
                frame_index,
                &self.transparent_shadow_pipeline,
                ShadowPassKind::Transparent,
                &batches.transparent,
            );
            self.logical_device.cmd_end_rendering(command_buffer);
        }

        self.transition_color_image_layout(
            command_buffer,
            self.shadow_map.opacity_image(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        );
        self.shadow_map
            .set_opacity_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShadowPassKind {
    Opaque,
    Cutout,
    Transparent,
}

#[derive(Clone, Copy)]
struct ShadowDrawItem {
    object: RenderObject,
    material: Material,
}

struct ShadowCasterBatches {
    opaque: Vec<ShadowDrawItem>,
    cutout: Vec<ShadowDrawItem>,
    transparent: Vec<ShadowDrawItem>,
    depth_scissor: vk::Rect2D,
    transparent_scissor: vk::Rect2D,
}

impl ShadowCasterBatches {
    fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.cutout.is_empty() && self.transparent.is_empty()
    }
}

struct ShadowTexelBounds {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    extent: vk::Extent2D,
}

impl ShadowTexelBounds {
    fn empty(extent: vk::Extent2D) -> Self {
        Self {
            min_x: extent.width as i32,
            min_y: extent.height as i32,
            max_x: 0,
            max_y: 0,
            extent,
        }
    }

    fn include(&mut self, shadow: PreparedShadow, light_center: [f32; 3], radius: f32) {
        let radius = radius + shadow.radius * 0.008;
        let inv_diameter = 0.5 / shadow.radius.max(0.001);
        let min_u = (light_center[0] - radius) * inv_diameter + 0.5;
        let max_u = (light_center[0] + radius) * inv_diameter + 0.5;
        let min_v = -(light_center[1] + radius) * inv_diameter + 0.5;
        let max_v = -(light_center[1] - radius) * inv_diameter + 0.5;
        let width = self.extent.width as f32;
        let height = self.extent.height as f32;
        let min_x = (min_u.clamp(0.0, 1.0) * width).floor() as i32 - SHADOW_SCISSOR_PADDING;
        let min_y = (min_v.clamp(0.0, 1.0) * height).floor() as i32 - SHADOW_SCISSOR_PADDING;
        let max_x = (max_u.clamp(0.0, 1.0) * width).ceil() as i32 + SHADOW_SCISSOR_PADDING;
        let max_y = (max_v.clamp(0.0, 1.0) * height).ceil() as i32 + SHADOW_SCISSOR_PADDING;

        self.min_x = self.min_x.min(min_x.max(0));
        self.min_y = self.min_y.min(min_y.max(0));
        self.max_x = self.max_x.max(max_x.min(self.extent.width as i32));
        self.max_y = self.max_y.max(max_y.min(self.extent.height as i32));
    }

    fn rect(&self) -> vk::Rect2D {
        if self.min_x >= self.max_x || self.min_y >= self.max_y {
            return render_area(self.extent);
        }

        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.min_x,
                y: self.min_y,
            },
            extent: vk::Extent2D {
                width: (self.max_x - self.min_x).max(1) as u32,
                height: (self.max_y - self.min_y).max(1) as u32,
            },
        }
    }
}

fn shadow_bias(projection: ShadowProjection, resolution: f32) -> f32 {
    let texel_world = projection
        .texel_world
        .max(projection.radius * 2.0 / resolution.max(1.0));
    let world_bias = (texel_world * 1.65).clamp(0.004, 0.18);

    (world_bias / projection.depth_range.max(0.001)).clamp(0.00001, 0.001)
}

fn shadow_flags_for_scene(renderer: &Renderer, scene: &RenderScene, shadow: PreparedShadow) -> u32 {
    let mut flags = 0;

    for_each_render_object(scene, &renderer.assets, |object| {
        if flags == (SHADOW_FLAG_DEPTH | SHADOW_FLAG_TRANSPARENT) {
            return;
        }
        let material = renderer.assets.material(object.material);
        let Some(pass) = shadow_pass_kind(material) else {
            return;
        };
        if shadow_caster_info(renderer, object, shadow).is_none() {
            return;
        }

        match pass {
            ShadowPassKind::Opaque | ShadowPassKind::Cutout => flags |= SHADOW_FLAG_DEPTH,
            ShadowPassKind::Transparent => flags |= SHADOW_FLAG_TRANSPARENT,
        }
    });

    flags
}

fn shadow_caster_info(
    renderer: &Renderer,
    object: RenderObject,
    shadow: PreparedShadow,
) -> Option<([f32; 3], f32)> {
    let Some(mesh) = renderer.assets.mesh(object.mesh) else {
        return None;
    };
    let center = transform_point(object.transform.matrix(), mesh.center);
    let radius = transformed_radius(object.transform, mesh.radius).max(0.01);
    let light_center = transform_point(shadow.view, center);
    let depth = -light_center[2];
    let margin = radius + shadow.radius * 0.04 + 1.0;

    (light_center[0].abs() <= shadow.radius + margin
        && light_center[1].abs() <= shadow.radius + margin
        && depth + radius >= shadow.near
        && depth - radius <= shadow.far)
        .then_some((light_center, radius))
}

fn collect_shadow_casters(
    renderer: &Renderer,
    scene: &RenderScene,
    shadow: PreparedShadow,
) -> ShadowCasterBatches {
    let mut opaque = Vec::new();
    let mut cutout = Vec::new();
    let mut transparent = Vec::new();
    let mut depth_bounds = ShadowTexelBounds::empty(renderer.shadow_map.extent);
    let mut transparent_bounds = ShadowTexelBounds::empty(renderer.shadow_map.extent);

    for_each_render_object(scene, &renderer.assets, |object| {
        let material = renderer.assets.material(object.material);
        let Some(pass) = shadow_pass_kind(material) else {
            return;
        };
        let Some((light_center, radius)) = shadow_caster_info(renderer, object, shadow) else {
            return;
        };
        let item = ShadowDrawItem { object, material };

        match pass {
            ShadowPassKind::Opaque => {
                depth_bounds.include(shadow, light_center, radius);
                opaque.push(item);
            }
            ShadowPassKind::Cutout => {
                depth_bounds.include(shadow, light_center, radius);
                cutout.push(item);
            }
            ShadowPassKind::Transparent => {
                transparent_bounds.include(shadow, light_center, radius);
                transparent.push(item);
            }
        }
    });

    sort_shadow_items(&mut opaque);
    sort_shadow_items(&mut cutout);
    sort_shadow_items(&mut transparent);

    ShadowCasterBatches {
        opaque,
        cutout,
        transparent,
        depth_scissor: depth_bounds.rect(),
        transparent_scissor: transparent_bounds.rect(),
    }
}

fn sort_shadow_items(items: &mut [ShadowDrawItem]) {
    items.sort_by_key(|item| (item.object.material.0, item.object.mesh.0));
}

fn set_shadow_scissor(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    scissor: vk::Rect2D,
) {
    unsafe {
        device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
    }
}

fn draw_shadow_pass(
    renderer: &Renderer,
    command_buffer: vk::CommandBuffer,
    frame_index: usize,
    slot: &PipelineSlot,
    pass: ShadowPassKind,
    items: &[ShadowDrawItem],
) {
    if items.is_empty() {
        return;
    }

    unsafe {
        renderer.logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            slot.pipeline.handle,
        );
        renderer.logical_device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            slot.pipeline.layout,
            0,
            std::slice::from_ref(&renderer.shadow_scene_bindings.sets[frame_index]),
            &[],
        );
    }

    let mut bound_material = None;
    let mut bound_mesh = None;
    for item in items {
        draw_shadow_object(
            renderer,
            command_buffer,
            item.object,
            item.material,
            slot,
            pass != ShadowPassKind::Opaque,
            &mut bound_material,
            &mut bound_mesh,
        );
    }
}

fn shadow_pass_kind(material: Material) -> Option<ShadowPassKind> {
    if material.casts_transparent_shadow() {
        Some(ShadowPassKind::Transparent)
    } else if material.alpha_cutoff() > f32::EPSILON {
        Some(ShadowPassKind::Cutout)
    } else if material.casts_depth_shadow() {
        Some(ShadowPassKind::Opaque)
    } else {
        None
    }
}

fn draw_shadow_object(
    renderer: &Renderer,
    command_buffer: vk::CommandBuffer,
    object: RenderObject,
    material: Material,
    slot: &PipelineSlot,
    bind_material_textures: bool,
    bound_material: &mut Option<MaterialId>,
    bound_mesh: &mut Option<MeshId>,
) {
    let Some(mesh) = renderer.assets.mesh(object.mesh) else {
        return;
    };

    unsafe {
        if bind_material_textures
            && *bound_material != Some(object.material)
            && let Some(texture_set) = renderer.assets.material_texture_set(object.material)
        {
            renderer.logical_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                slot.pipeline.layout,
                1,
                std::slice::from_ref(&texture_set),
                &[],
            );
            *bound_material = Some(object.material);
        }

        if *bound_mesh != Some(object.mesh) {
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
            *bound_mesh = Some(object.mesh);
        }

        let push = ObjectPush::new(object, material);
        renderer.logical_device.cmd_push_constants(
            command_buffer,
            slot.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes_of(&push),
        );

        renderer
            .logical_device
            .cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
    }
}
