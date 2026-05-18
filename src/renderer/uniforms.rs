use std::mem;

use ash::vk;

use super::{
    Camera, DirectionalLight, Material, RenderObject, RenderScene, TextureId,
    reflections::PreparedReflections, shadows::PreparedShadow,
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
    pub(super) planar_view_proj: [f32; 16],
    reflection_params: [f32; 4],
    planar_plane: [f32; 4],
    planar_params: [f32; 4],
    planar_texture_info: [f32; 4],
    shadow_view_proj: [f32; 16],
    shadow_params: [f32; 4],
    debug_params: [f32; 4],
    camera_response: [f32; 4],
    white_balance: [f32; 4],
    gi_probe_pos_radius: [f32; 4],
    gi_params: [f32; 4],
    gi_sh: [[f32; 4]; 9],
    camera_basis_x: [f32; 4],
    camera_basis_y: [f32; 4],
    camera_basis_z: [f32; 4],
    post_params: [f32; 4],
}

impl SceneUniform {
    pub(super) fn new(
        scene: &RenderScene,
        extent: vk::Extent2D,
        reflections: PreparedReflections,
        shadow: PreparedShadow,
    ) -> Self {
        let mut uniform = Self::base(scene, extent);
        uniform.set_reflection_info(reflections, false);
        uniform.set_shadow_info(shadow);
        uniform
    }

    pub(super) fn shadow(scene: &RenderScene, shadow: PreparedShadow) -> Self {
        let mut uniform = Self::base(
            scene,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
        );
        uniform.view_proj = shadow.view_proj;
        uniform.set_shadow_info(shadow);
        uniform
    }

    pub(super) fn base(scene: &RenderScene, extent: vk::Extent2D) -> Self {
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let light = scene.light;
        let camera_response = scene.camera_response;
        let basis = camera_basis(scene.camera);
        let tan_half_fov = (scene.camera.fov_y * 0.5).tan();

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
            planar_view_proj: scene.camera.view_projection(aspect),
            reflection_params: [0.0, 0.0, 0.35, 0.0],
            planar_plane: [0.0, 1.0, 0.0, 0.0],
            planar_params: [0.0, 0.75, 0.35, 3.5],
            planar_texture_info: [0.0, 0.03, 0.0, 0.0],
            shadow_view_proj: identity_mat4(),
            shadow_params: [0.0; 4],
            debug_params: [
                scene.debug_mode.shader_value(),
                scene.camera.near,
                scene.camera.far,
                0.0,
            ],
            camera_response: [
                camera_response.exposure.max(0.0),
                camera_response.contrast.max(0.0),
                camera_response.saturation.max(0.0),
                camera_response.enabled as u8 as f32,
            ],
            white_balance: [
                camera_response.white_balance[0].max(0.0),
                camera_response.white_balance[1].max(0.0),
                camera_response.white_balance[2].max(0.0),
                0.0,
            ],
            gi_probe_pos_radius: [
                scene.camera.target[0],
                scene.camera.target[1],
                scene.camera.target[2],
                scene.camera.far.max(1.0),
            ],
            gi_params: [1.0, 1.0, 0.18, 0.14],
            gi_sh: irradiance_sh(light),
            camera_basis_x: [
                basis.right[0],
                basis.right[1],
                basis.right[2],
                scene.camera.near,
            ],
            camera_basis_y: [basis.up[0], basis.up[1], basis.up[2], scene.camera.far],
            camera_basis_z: [basis.forward[0], basis.forward[1], basis.forward[2], 0.0],
            post_params: [aspect, tan_half_fov, 1.0, 0.0],
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
        self.gi_probe_pos_radius = [
            reflection.center[0],
            reflection.center[1],
            reflection.center[2],
            reflection.radius.max(1.0),
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

    pub(super) fn set_shadow_info(&mut self, shadow: PreparedShadow) {
        self.shadow_view_proj = shadow.view_proj;
        self.shadow_params = [
            shadow.resolution,
            shadow.bias,
            shadow.strength.clamp(0.0, 1.0),
            1.0,
        ];
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

impl ObjectPush {
    pub(super) fn new(object: RenderObject, material: Material) -> Self {
        Self {
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
        }
    }
}

pub(super) fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) }
}

pub(super) fn has_texture(texture: Option<TextureId>) -> f32 {
    texture.is_some() as u8 as f32
}

struct CameraBasis {
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
}

fn camera_basis(camera: Camera) -> CameraBasis {
    let forward = normalize3([
        camera.target[0] - camera.eye[0],
        camera.target[1] - camera.eye[1],
        camera.target[2] - camera.eye[2],
    ]);
    let right = normalize3(cross3(forward, camera.up));
    let up = cross3(right, forward);

    CameraBasis { right, up, forward }
}

fn irradiance_sh(light: DirectionalLight) -> [[f32; 4]; 9] {
    let mut sh = [[0.0; 4]; 9];
    let scale = std::f32::consts::PI / 0.282_095;

    sh[0] = [
        light.ambient[0] * scale,
        light.ambient[1] * scale,
        light.ambient[2] * scale,
        0.0,
    ];
    sh
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(f32::EPSILON);

    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
