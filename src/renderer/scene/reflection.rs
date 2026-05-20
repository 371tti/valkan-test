#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectionSettings {
    pub box_projection: BoxReflectionSettings,
    pub planar: PlanarReflectionSettings,
}

#[derive(Debug, Clone, Copy)]
pub struct BoxReflectionSettings {
    pub enabled: bool,
    pub parallax_correction: bool,
    pub resolution: u32,
    pub intensity: f32,
    pub roughness_fallback: f32,
    pub bounds_padding: f32,
}

impl Default for BoxReflectionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            parallax_correction: true,
            resolution: 128,
            intensity: 1.0,
            roughness_fallback: 0.35,
            bounds_padding: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlanarReflectionSettings {
    pub enabled: bool,
    pub plane_origin: [f32; 3],
    pub plane_normal: [f32; 3],
    pub resolution_scale: f32,
    pub intensity: f32,
    pub max_roughness: f32,
    pub normal_alignment: f32,
    pub distance_fade: f32,
    pub clip_bias: f32,
    pub uv_flip_y: bool,
}

impl Default for PlanarReflectionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            plane_origin: [0.0, 0.0, 0.0],
            plane_normal: [0.0, 1.0, 0.0],
            resolution_scale: 0.5,
            intensity: 1.0,
            max_roughness: 0.75,
            normal_alignment: 0.35,
            distance_fade: 3.5,
            clip_bias: 0.03,
            uv_flip_y: false,
        }
    }
}
