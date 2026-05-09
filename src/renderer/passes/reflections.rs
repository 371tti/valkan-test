use ash::vk;

use super::{
    BoxReflectionSettings, Camera, PlanarReflectionSettings, RenderObject, RenderScene, Renderer,
    math::{
        desired_reflection_probe_size, distance_squared, dot3, normalize_or, normalize_or_zero,
        reflect_camera, reflection_probe_face_axes, transform_point, transformed_radius,
    },
    uniforms::SceneUniform,
};

#[derive(Clone, Copy)]
pub(super) struct ReflectionProbeInfo {
    pub center: [f32; 3],
    pub radius: f32,
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    pub enabled: bool,
    pub parallax_correction: bool,
    pub intensity: f32,
    pub roughness_fallback: f32,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedPlanarReflection {
    pub enabled: bool,
    pub camera: Camera,
    pub view_proj: [f32; 16],
    pub normal: [f32; 3],
    pub d: f32,
    pub intensity: f32,
    pub max_roughness: f32,
    pub normal_alignment: f32,
    pub distance_fade: f32,
    pub clip_bias: f32,
    pub uv_flip_y: bool,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedReflections {
    pub probe: ReflectionProbeInfo,
    pub planar: PreparedPlanarReflection,
}

struct SceneBounds {
    min: [f32; 3],
    max: [f32; 3],
    has_value: bool,
}

impl SceneBounds {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            has_value: false,
        }
    }

    fn include_sphere(&mut self, center: [f32; 3], radius: f32) {
        let radius = radius.max(0.01);
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(center[axis] - radius);
            self.max[axis] = self.max[axis].max(center[axis] + radius);
        }
        self.has_value = true;
    }

    fn info(
        &self,
        fallback_center: [f32; 3],
        preferred_center: Option<[f32; 3]>,
        settings: BoxReflectionSettings,
    ) -> ReflectionProbeInfo {
        if !self.has_value {
            return ReflectionProbeInfo {
                center: fallback_center,
                radius: 1.0,
                box_min: [
                    fallback_center[0] - 1.0,
                    fallback_center[1] - 1.0,
                    fallback_center[2] - 1.0,
                ],
                box_max: [
                    fallback_center[0] + 1.0,
                    fallback_center[1] + 1.0,
                    fallback_center[2] + 1.0,
                ],
                enabled: settings.enabled,
                parallax_correction: settings.parallax_correction,
                intensity: settings.intensity.clamp(0.0, 2.0),
                roughness_fallback: settings.roughness_fallback.clamp(0.0, 1.0),
            };
        }

        let scene_center = [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ];
        let center = preferred_center.unwrap_or(scene_center);
        let extent = [
            (self.max[0] - self.min[0]).max(0.1),
            (self.max[1] - self.min[1]).max(0.1),
            (self.max[2] - self.min[2]).max(0.1),
        ];
        let margin = extent[0].max(extent[1]).max(extent[2]).max(1.0)
            * settings.bounds_padding.clamp(0.0, 1.0);
        let box_min = [
            self.min[0] - margin,
            self.min[1] - margin,
            self.min[2] - margin,
        ];
        let box_max = [
            self.max[0] + margin,
            self.max[1] + margin,
            self.max[2] + margin,
        ];
        let radius = distance_squared(box_min, box_max).sqrt() * 0.5;

        ReflectionProbeInfo {
            center,
            radius,
            box_min,
            box_max,
            enabled: settings.enabled,
            parallax_correction: settings.parallax_correction,
            intensity: settings.intensity.clamp(0.0, 2.0),
            roughness_fallback: settings.roughness_fallback.clamp(0.0, 1.0),
        }
    }
}

impl Renderer {
    pub(super) fn ensure_reflection_targets(&mut self, scene: &RenderScene) {
        let desired_probe_size = desired_reflection_probe_size(scene.reflections.box_projection);
        let desired_planar_extent = self.desired_planar_reflection_extent(scene.reflections.planar);
        let rebuild_probe = self.reflection_probe.extent.width != desired_probe_size;
        let rebuild_planar = self.planar_reflection.extent != desired_planar_extent;

        if !rebuild_probe && !rebuild_planar {
            return;
        }

        self.wait_for_swapchain_idle();

        unsafe {
            if rebuild_probe {
                self.reflection_probe.destroy(&self.logical_device);
                self.reflection_probe = super::assets::ReflectionProbe::new(
                    &self.instance,
                    &self.logical_device,
                    self.physical_device,
                    self.command_pool,
                    self.graphics_queue,
                    self.swapchain.format,
                    desired_probe_size,
                );
            }

            if rebuild_planar {
                self.planar_reflection.destroy(&self.logical_device);
                self.planar_reflection = super::assets::PlanarReflectionTarget::new(
                    &self.instance,
                    &self.logical_device,
                    self.physical_device,
                    self.command_pool,
                    self.graphics_queue,
                    self.swapchain.format,
                    desired_planar_extent,
                );
            }
        }

        self.update_reflection_descriptors();
    }

    fn desired_planar_reflection_extent(&self, planar: PlanarReflectionSettings) -> vk::Extent2D {
        if !planar.enabled {
            return vk::Extent2D {
                width: 1,
                height: 1,
            };
        }

        let scale = planar.resolution_scale.clamp(0.25, 1.0);
        vk::Extent2D {
            width: ((self.swapchain.extent.width as f32 * scale).round() as u32).clamp(
                super::PLANAR_REFLECTION_MIN_SIZE,
                super::PLANAR_REFLECTION_MAX_SIZE,
            ),
            height: ((self.swapchain.extent.height as f32 * scale).round() as u32).clamp(
                super::PLANAR_REFLECTION_MIN_SIZE,
                super::PLANAR_REFLECTION_MAX_SIZE,
            ),
        }
    }

    pub(super) fn update_reflection_descriptors(&self) {
        self.scene_bindings.update_reflections(
            &self.logical_device,
            self.reflection_probe.descriptor(),
            self.planar_reflection.descriptor(),
        );
        self.probe_scene_bindings.update_reflections(
            &self.logical_device,
            self.fallback_reflection_probe.descriptor(),
            self.fallback_planar_reflection.descriptor(),
        );
        self.planar_scene_bindings.update_reflections(
            &self.logical_device,
            self.fallback_reflection_probe.descriptor(),
            self.fallback_planar_reflection.descriptor(),
        );
    }

    pub(super) fn prepare_reflections(&self, scene: &RenderScene) -> PreparedReflections {
        PreparedReflections {
            probe: self.reflection_probe_info(scene),
            planar: self.planar_reflection_info(scene),
        }
    }

    fn planar_reflection_info(&self, scene: &RenderScene) -> PreparedPlanarReflection {
        let settings = scene.reflections.planar;
        let normal = normalize_or(settings.plane_normal, [0.0, 1.0, 0.0]);
        let d = -dot3(normal, settings.plane_origin);
        let camera = reflect_camera(scene.camera, normal, d);
        let aspect = self.planar_reflection.extent.width as f32
            / self.planar_reflection.extent.height.max(1) as f32;

        PreparedPlanarReflection {
            enabled: settings.enabled,
            camera,
            view_proj: camera.view_projection(aspect),
            normal,
            d,
            intensity: settings.intensity.clamp(0.0, 2.0),
            max_roughness: settings.max_roughness.clamp(0.08, 1.0),
            normal_alignment: settings.normal_alignment.clamp(0.0, 1.0),
            distance_fade: settings.distance_fade.max(0.25),
            clip_bias: settings.clip_bias.max(0.0),
            uv_flip_y: settings.uv_flip_y,
        }
    }

    fn reflection_probe_info(&self, scene: &RenderScene) -> ReflectionProbeInfo {
        let mut bounds = SceneBounds::empty();
        let camera_forward = normalize_or_zero([
            scene.camera.target[0] - scene.camera.eye[0],
            scene.camera.target[1] - scene.camera.eye[1],
            scene.camera.target[2] - scene.camera.eye[2],
        ]);

        for object in &scene.objects {
            self.consider_reflection_probe_object(
                &mut bounds,
                *object,
                scene.camera.eye,
                camera_forward,
            );
        }

        for model in &scene.models {
            let Some(gpu_model) = self.assets.model(model.model) else {
                continue;
            };

            for primitive in &gpu_model.primitives {
                self.consider_reflection_probe_object(
                    &mut bounds,
                    RenderObject {
                        mesh: primitive.mesh,
                        pipeline: model.pipeline,
                        transform: model.transform,
                        material: primitive.material,
                    },
                    scene.camera.eye,
                    camera_forward,
                );
            }
        }

        bounds.info(scene.camera.target, None, scene.reflections.box_projection)
    }

    fn consider_reflection_probe_object(
        &self,
        bounds: &mut SceneBounds,
        object: RenderObject,
        camera_eye: [f32; 3],
        camera_forward: [f32; 3],
    ) {
        let Some((center, radius)) = self.object_center_radius(object) else {
            return;
        };
        bounds.include_sphere(center, radius);

        let material = self.assets.material(object.material);
        let reflectivity = material.metallic
            + material
                .metallic_roughness_texture
                .is_some()
                .then_some(0.5)
                .unwrap_or(0.0);
        if reflectivity <= 0.2 {
            return;
        }

        let _ = (camera_eye, camera_forward);
    }

    fn object_center_radius(&self, object: RenderObject) -> Option<([f32; 3], f32)> {
        let Some(mesh) = self.assets.mesh(object.mesh) else {
            return None;
        };
        let matrix = object.transform.matrix();
        let center = transform_point(matrix, mesh.center);
        let radius = transformed_radius(object.transform, mesh.radius);

        Some((center, radius))
    }
}

impl SceneUniform {
    pub(super) fn reflection_probe_face(
        scene: &RenderScene,
        assets: &super::assets::GpuAssets,
        reflections: PreparedReflections,
        face: usize,
    ) -> Self {
        let (direction, up) = reflection_probe_face_axes(face);
        let center = reflections.probe.center;
        let camera = Camera {
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
        };
        let mut uniform = Self::base(
            &RenderScene {
                camera,
                light: scene.light,
                reflections: scene.reflections,
                objects: Vec::new(),
                models: Vec::new(),
            },
            vk::Extent2D {
                width: 1,
                height: 1,
            },
        );
        uniform.view_proj = camera.cubemap_view_projection();
        uniform.planar_view_proj = reflections.planar.view_proj;
        uniform.camera_pos = [center[0], center[1], center[2], 0.0];
        uniform.set_reflection_info(reflections, false);
        uniform.collect_emissive_lights(scene, assets);
        uniform
    }

    pub(super) fn planar_reflection(
        scene: &RenderScene,
        extent: vk::Extent2D,
        assets: &super::assets::GpuAssets,
        reflections: PreparedReflections,
    ) -> Self {
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let mut uniform = Self::base(
            &RenderScene {
                camera: reflections.planar.camera,
                light: scene.light,
                reflections: scene.reflections,
                objects: Vec::new(),
                models: Vec::new(),
            },
            extent,
        );
        uniform.view_proj = reflections.planar.camera.view_projection(aspect);
        uniform.planar_view_proj = reflections.planar.view_proj;
        uniform.camera_pos = [
            reflections.planar.camera.eye[0],
            reflections.planar.camera.eye[1],
            reflections.planar.camera.eye[2],
            0.0,
        ];
        uniform.set_reflection_info(reflections, true);
        uniform.collect_emissive_lights(scene, assets);
        uniform
    }
}
