use super::TextureId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialAlpha {
    Opaque,
    Mask(f32),
    Blend,
}

impl MaterialAlpha {
    pub fn cutoff(self) -> f32 {
        match self {
            Self::Mask(cutoff) => cutoff.clamp(0.0, 1.0),
            Self::Opaque | Self::Blend => 0.0,
        }
    }

    pub fn is_blend(self) -> bool {
        matches!(self, Self::Blend)
    }
}

impl Default for MaterialAlpha {
    fn default() -> Self {
        Self::Opaque
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialTextures {
    pub base_color: Option<TextureId>,
    pub metallic_roughness: Option<TextureId>,
    pub normal: Option<TextureId>,
    pub occlusion: Option<TextureId>,
    pub emissive: Option<TextureId>,
}

impl MaterialTextures {
    pub const fn empty() -> Self {
        Self {
            base_color: None,
            metallic_roughness: None,
            normal: None,
            occlusion: None,
            emissive: None,
        }
    }
}

impl Default for MaterialTextures {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub base_color: [f32; 4],
    pub textures: MaterialTextures,
    pub emissive_color: [f32; 3],
    pub emissive_strength: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub specular: f32,
    pub specular_color: [f32; 3],
    pub transmission: f32,
    pub ambient_occlusion: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha: MaterialAlpha,
    pub double_sided: bool,
}

impl Material {
    pub const fn new(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            textures: MaterialTextures::empty(),
            emissive_color: [0.0; 3],
            emissive_strength: 1.0,
            metallic: 0.0,
            roughness: 0.55,
            specular: 0.5,
            specular_color: [1.0; 3],
            transmission: 0.0,
            ambient_occlusion: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha: MaterialAlpha::Opaque,
            double_sided: false,
        }
    }

    pub const fn matte(base_color: [f32; 4]) -> Self {
        Self {
            roughness: 0.96,
            specular: 0.08,
            ..Self::new(base_color)
        }
    }

    pub const fn metal(base_color: [f32; 4], roughness: f32) -> Self {
        Self {
            metallic: 1.0,
            roughness,
            specular: 0.75,
            ..Self::new(base_color)
        }
    }

    pub const fn emissive(base_color: [f32; 4], emissive_color: [f32; 3]) -> Self {
        Self {
            emissive_color,
            specular: 0.1,
            ..Self::new(base_color)
        }
    }

    pub fn base_color_texture(&self) -> Option<TextureId> {
        self.textures.base_color
    }

    pub fn metallic_roughness_texture(&self) -> Option<TextureId> {
        self.textures.metallic_roughness
    }

    pub fn normal_texture(&self) -> Option<TextureId> {
        self.textures.normal
    }

    pub fn occlusion_texture(&self) -> Option<TextureId> {
        self.textures.occlusion
    }

    pub fn emissive_texture(&self) -> Option<TextureId> {
        self.textures.emissive
    }

    pub fn alpha_cutoff(&self) -> f32 {
        self.alpha.cutoff()
    }

    pub fn alpha_blend(&self) -> bool {
        self.alpha.is_blend()
    }

    pub fn emissive_radiance(&self) -> [f32; 3] {
        [
            self.emissive_color[0] * self.emissive_strength,
            self.emissive_color[1] * self.emissive_strength,
            self.emissive_color[2] * self.emissive_strength,
        ]
    }

    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.04, 1.0);
        self
    }

    pub fn with_specular(mut self, specular: f32) -> Self {
        self.specular = specular.clamp(0.0, 1.0);
        self
    }

    pub fn with_specular_color(mut self, specular_color: [f32; 3]) -> Self {
        self.specular_color = [
            specular_color[0].clamp(0.0, 1.0),
            specular_color[1].clamp(0.0, 1.0),
            specular_color[2].clamp(0.0, 1.0),
        ];
        self
    }

    pub fn with_transmission(mut self, transmission: f32) -> Self {
        self.transmission = transmission.clamp(0.0, 1.0);
        self
    }

    pub fn with_emissive(mut self, emissive_color: [f32; 3]) -> Self {
        self.emissive_color = emissive_color;
        self
    }

    pub fn with_emissive_strength(mut self, emissive_strength: f32) -> Self {
        self.emissive_strength = emissive_strength.max(0.0);
        self
    }

    pub fn with_base_color_texture(mut self, texture: TextureId) -> Self {
        self.textures.base_color = Some(texture);
        self
    }

    pub fn with_metallic_roughness_texture(mut self, texture: TextureId) -> Self {
        self.textures.metallic_roughness = Some(texture);
        self
    }

    pub fn with_normal_texture(mut self, texture: TextureId, scale: f32) -> Self {
        self.textures.normal = Some(texture);
        self.normal_scale = scale;
        self
    }

    pub fn with_occlusion_texture(mut self, texture: TextureId, strength: f32) -> Self {
        self.textures.occlusion = Some(texture);
        self.occlusion_strength = strength.clamp(0.0, 1.0);
        self
    }

    pub fn with_emissive_texture(mut self, texture: TextureId) -> Self {
        self.textures.emissive = Some(texture);
        self
    }

    pub fn with_alpha_mode(mut self, alpha: MaterialAlpha) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_alpha_cutoff(self, alpha_cutoff: f32) -> Self {
        self.with_alpha_mode(MaterialAlpha::Mask(alpha_cutoff))
    }

    pub fn with_alpha_blend(mut self, alpha_blend: bool) -> Self {
        if alpha_blend {
            self.alpha = MaterialAlpha::Blend;
        } else if self.alpha.is_blend() {
            self.alpha = MaterialAlpha::Opaque;
        }
        self
    }

    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    pub fn is_translucent(&self) -> bool {
        self.alpha.is_blend() || self.base_color[3] < 0.999 || self.transmission > 0.001
    }

    pub fn shadow_opacity_hint(&self) -> f32 {
        self.base_color[3].clamp(0.0, 1.0) * (1.0 - self.transmission.clamp(0.0, 1.0) * 0.82)
    }

    pub fn casts_shadow(&self) -> bool {
        self.base_color_texture().is_some() || self.shadow_opacity_hint() > 0.02
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([1.0; 4])
    }
}
