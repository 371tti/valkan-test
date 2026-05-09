use std::mem;

use ash::vk;

use super::{
    MAX_EMISSIVE_LIGHTS, RenderObject, RenderScene, TextureId,
    assets::GpuAssets,
    math::{transform_point, transformed_radius},
    reflections::PreparedReflections,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SceneUniform {
    pub(super) view_proj: [f32; 16],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
    pub(super) camera_pos: [f32; 4],
    reflection_probe_pos_radius: [f32; 4],
    reflection_probe_box_min: [f32; 4],
    reflection_probe_box_max: [f32; 4],
    point_light_count: [f32; 4],
    point_light_pos_radius: [[f32; 4]; MAX_EMISSIVE_LIGHTS],
    point_light_color_power: [[f32; 4]; MAX_EMISSIVE_LIGHTS],
    pub(super) planar_view_proj: [f32; 16],
    reflection_params: [f32; 4],
    planar_plane: [f32; 4],
    planar_params: [f32; 4],
    planar_texture_info: [f32; 4],
}

impl SceneUniform {
    pub(super) fn new(
        scene: &RenderScene,
        extent: vk::Extent2D,
        assets: &GpuAssets,
        reflections: PreparedReflections,
    ) -> Self {
        let mut uniform = Self::base(scene, extent);
        uniform.set_reflection_info(reflections, false);
        uniform.collect_emissive_lights(scene, assets);
        uniform
    }

    pub(super) fn base(scene: &RenderScene, extent: vk::Extent2D) -> Self {
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let light = scene.light;

        Self {
            view_proj: scene.camera.view_projection(aspect),
            light_dir: [
                light.direction[0],
                light.direction[1],
                light.direction[2],
                0.0,
            ],
            light_color: [
                light.color[0] * light.intensity,
                light.color[1] * light.intensity,
                light.color[2] * light.intensity,
                0.0,
            ],
            ambient: [light.ambient[0], light.ambient[1], light.ambient[2], 0.0],
            camera_pos: [
                scene.camera.eye[0],
                scene.camera.eye[1],
                scene.camera.eye[2],
                1.0,
            ],
            reflection_probe_pos_radius: [
                scene.camera.target[0],
                scene.camera.target[1],
                scene.camera.target[2],
                0.0,
            ],
            reflection_probe_box_min: [
                scene.camera.target[0] - 1.0,
                scene.camera.target[1] - 1.0,
                scene.camera.target[2] - 1.0,
                0.0,
            ],
            reflection_probe_box_max: [
                scene.camera.target[0] + 1.0,
                scene.camera.target[1] + 1.0,
                scene.camera.target[2] + 1.0,
                0.0,
            ],
            point_light_count: [0.0; 4],
            point_light_pos_radius: [[0.0; 4]; MAX_EMISSIVE_LIGHTS],
            point_light_color_power: [[0.0; 4]; MAX_EMISSIVE_LIGHTS],
            planar_view_proj: scene.camera.view_projection(aspect),
            reflection_params: [0.0, 0.0, 0.35, 0.0],
            planar_plane: [0.0, 1.0, 0.0, 0.0],
            planar_params: [0.0, 0.75, 0.35, 3.5],
            planar_texture_info: [0.0, 0.03, 0.0, 0.0],
        }
    }

    pub(super) fn set_reflection_info(
        &mut self,
        reflections: PreparedReflections,
        planar_pass: bool,
    ) {
        let reflection = reflections.probe;
        self.reflection_probe_pos_radius = [
            reflection.center[0],
            reflection.center[1],
            reflection.center[2],
            reflection.radius,
        ];
        self.reflection_probe_box_min = [
            reflection.box_min[0],
            reflection.box_min[1],
            reflection.box_min[2],
            0.0,
        ];
        self.reflection_probe_box_max = [
            reflection.box_max[0],
            reflection.box_max[1],
            reflection.box_max[2],
            0.0,
        ];
        self.reflection_params = [
            reflections.probe.intensity,
            reflections.probe.parallax_correction as u8 as f32,
            reflections.probe.roughness_fallback,
            reflections.probe.enabled as u8 as f32,
        ];
        self.planar_view_proj = reflections.planar.view_proj;
        self.planar_plane = [
            reflections.planar.normal[0],
            reflections.planar.normal[1],
            reflections.planar.normal[2],
            reflections.planar.d,
        ];
        self.planar_params = [
            reflections.planar.intensity,
            reflections.planar.max_roughness,
            reflections.planar.normal_alignment,
            reflections.planar.distance_fade,
        ];
        self.planar_texture_info = [
            reflections.planar.enabled as u8 as f32,
            reflections.planar.clip_bias,
            reflections.planar.uv_flip_y as u8 as f32,
            planar_pass as u8 as f32,
        ];
    }

    pub(super) fn collect_emissive_lights(&mut self, scene: &RenderScene, assets: &GpuAssets) {
        for object in &scene.objects {
            self.push_emissive_light(assets, *object);
        }

        for model in &scene.models {
            let Some(gpu_model) = assets.model(model.model) else {
                continue;
            };

            for primitive in &gpu_model.primitives {
                self.push_emissive_light(
                    assets,
                    RenderObject {
                        mesh: primitive.mesh,
                        pipeline: model.pipeline,
                        transform: model.transform,
                        material: primitive.material,
                    },
                );
            }
        }
    }

    fn push_emissive_light(&mut self, assets: &GpuAssets, object: RenderObject) {
        let index = self.point_light_count[0] as usize;
        if index >= MAX_EMISSIVE_LIGHTS {
            return;
        }

        let material = assets.material(object.material);
        let power = material
            .emissive_color
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        if power <= 0.001 {
            return;
        }

        let Some(mesh) = assets.mesh(object.mesh) else {
            return;
        };

        let matrix = object.transform.matrix();
        let position = transform_point(matrix, mesh.center);
        let radius = transformed_radius(object.transform, mesh.radius).max(0.35);
        self.point_light_pos_radius[index] = [position[0], position[1], position[2], radius];
        self.point_light_color_power[index] = [
            material.emissive_color[0],
            material.emissive_color[1],
            material.emissive_color[2],
            1.5 + power * 3.5,
        ];
        self.point_light_count[0] += 1.0;
    }
}

impl Default for SceneUniform {
    fn default() -> Self {
        Self::base(
            &RenderScene::default(),
            vk::Extent2D {
                width: 1,
                height: 1,
            },
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ObjectPush {
    pub(super) model: [f32; 16],
    pub(super) base_color: [f32; 4],
    pub(super) emissive_color: [f32; 4],
    pub(super) material: [f32; 4],
    pub(super) texture_flags: [f32; 4],
    pub(super) texture_info: [f32; 4],
}

pub(super) fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) }
}

pub(super) fn has_texture(texture: Option<TextureId>) -> f32 {
    texture.is_some() as u8 as f32
}
