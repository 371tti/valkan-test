use std::collections::BTreeMap;

use super::TextureHandle;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MaterialTextureSlot {
    BaseColor,
    Normal,
    MetallicRoughness,
    Occlusion,
    Emissive,
}

impl MaterialTextureSlot {
    /// Parses a stable material texture slot name used by import manifests and shader docs.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "base_color" => Some(Self::BaseColor),
            "normal" => Some(Self::Normal),
            "metallic_roughness" => Some(Self::MetallicRoughness),
            "occlusion" => Some(Self::Occlusion),
            "emissive" => Some(Self::Emissive),
            _ => None,
        }
    }

    /// Returns the stable slot name shared by import, material store, and shader interface docs.
    pub fn name(self) -> &'static str {
        match self {
            Self::BaseColor => "base_color",
            Self::Normal => "normal",
            Self::MetallicRoughness => "metallic_roughness",
            Self::Occlusion => "occlusion",
            Self::Emissive => "emissive",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialAlphaMode {
    Opaque,
    Cutout,
    Transparent,
}

impl MaterialAlphaMode {
    /// Parses an alpha mode name without silently changing the material behavior.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "opaque" => Some(Self::Opaque),
            "cutout" => Some(Self::Cutout),
            "transparent" => Some(Self::Transparent),
            _ => None,
        }
    }

    /// Returns the stable alpha mode name used by logs and import manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::Opaque => "opaque",
            Self::Cutout => "cutout",
            Self::Transparent => "transparent",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDescriptor {
    alpha_mode: MaterialAlphaMode,
    alpha_cutoff_milli: u16,
    base_color_factor: [f32; 4],
    metallic_factor_milli: u16,
    roughness_factor_milli: u16,
    emissive_factor: [f32; 3],
    occlusion_strength_milli: u16,
    normal_scale_milli: u16,
    double_sided: bool,
    textures: BTreeMap<MaterialTextureSlot, TextureHandle>,
}

impl MaterialDescriptor {
    /// Creates material parameters after import has resolved texture references to handles.
    pub fn new(alpha_mode: MaterialAlphaMode, alpha_cutoff_milli: u16) -> Self {
        Self {
            alpha_mode,
            alpha_cutoff_milli,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor_milli: 0,
            roughness_factor_milli: 1000,
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength_milli: 1000,
            normal_scale_milli: 1000,
            double_sided: false,
            textures: BTreeMap::new(),
        }
    }

    /// Creates full material parameters imported from glTF PBR data.
    #[allow(clippy::too_many_arguments)]
    pub fn with_pbr(
        alpha_mode: MaterialAlphaMode,
        alpha_cutoff_milli: u16,
        base_color_factor: [f32; 4],
        metallic_factor_milli: u16,
        roughness_factor_milli: u16,
        emissive_factor: [f32; 3],
        occlusion_strength_milli: u16,
        normal_scale_milli: u16,
        double_sided: bool,
    ) -> Option<Self> {
        let valid_base_color = base_color_factor
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value));
        let valid_emissive = emissive_factor
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0);
        (valid_base_color && valid_emissive).then_some(Self {
            alpha_mode,
            alpha_cutoff_milli: alpha_cutoff_milli.min(1000),
            base_color_factor,
            metallic_factor_milli: metallic_factor_milli.min(1000),
            roughness_factor_milli: roughness_factor_milli.min(1000),
            emissive_factor,
            occlusion_strength_milli: occlusion_strength_milli.min(1000),
            normal_scale_milli: normal_scale_milli.min(4000),
            double_sided,
            textures: BTreeMap::new(),
        })
    }

    /// Records one named texture slot binding for this material.
    pub fn set_texture(&mut self, slot: MaterialTextureSlot, texture: TextureHandle) {
        self.textures.insert(slot, texture);
    }

    /// Returns the alpha mode that selects the material pipeline variant.
    pub fn alpha_mode(&self) -> MaterialAlphaMode {
        self.alpha_mode
    }

    /// Returns the alpha cutoff as a milli value to keep protocol data deterministic.
    pub fn alpha_cutoff_milli(&self) -> u16 {
        self.alpha_cutoff_milli
    }

    /// Returns the linear base-color multiplier imported from material data.
    pub fn base_color_factor(&self) -> [f32; 4] {
        self.base_color_factor
    }

    /// Returns metallic factor in deterministic milli units.
    pub fn metallic_factor_milli(&self) -> u16 {
        self.metallic_factor_milli
    }

    /// Returns roughness factor in deterministic milli units.
    pub fn roughness_factor_milli(&self) -> u16 {
        self.roughness_factor_milli
    }

    /// Returns emissive RGB multiplier imported from material data.
    pub fn emissive_factor(&self) -> [f32; 3] {
        self.emissive_factor
    }

    /// Returns occlusion strength in deterministic milli units.
    pub fn occlusion_strength_milli(&self) -> u16 {
        self.occlusion_strength_milli
    }

    /// Returns normal-map strength in deterministic milli units.
    pub fn normal_scale_milli(&self) -> u16 {
        self.normal_scale_milli
    }

    /// Returns whether this material should render both winding sides.
    pub fn double_sided(&self) -> bool {
        self.double_sided
    }

    /// Returns the texture handle bound to one named slot when the material provides it.
    pub fn texture(&self, slot: MaterialTextureSlot) -> Option<TextureHandle> {
        self.textures.get(&slot).copied()
    }

    /// Returns all named texture slots in deterministic order.
    pub fn textures(&self) -> &BTreeMap<MaterialTextureSlot, TextureHandle> {
        &self.textures
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureFormat {
    Rgba8Srgb,
    Rgba8Linear,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextureDescriptor {
    width: u32,
    height: u32,
    format: TextureFormat,
    pixels: Vec<u8>,
}

impl TextureDescriptor {
    /// Creates one explicit RGBA8 sRGB texture payload.
    pub fn rgba8_srgb(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        Self::rgba8(width, height, TextureFormat::Rgba8Srgb, pixels)
    }

    /// Creates one explicit RGBA8 linear texture payload for non-color material maps.
    pub fn rgba8_linear(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        Self::rgba8(width, height, TextureFormat::Rgba8Linear, pixels)
    }

    /// Creates one checked RGBA8 texture payload with an explicit color-space contract.
    fn rgba8(width: u32, height: u32, format: TextureFormat, pixels: Vec<u8>) -> Option<Self> {
        let expected = width.checked_mul(height)?.checked_mul(4)? as usize;
        (width > 0 && height > 0 && pixels.len() == expected).then_some(Self {
            width,
            height,
            format,
            pixels,
        })
    }

    /// Creates a solid one-pixel texture explicitly requested by an import manifest.
    pub fn solid_rgba8_srgb(rgba: [u8; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            format: TextureFormat::Rgba8Srgb,
            pixels: rgba.to_vec(),
        }
    }

    /// Creates a solid one-pixel linear texture for non-color material maps.
    pub fn solid_rgba8_linear(rgba: [u8; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            format: TextureFormat::Rgba8Linear,
            pixels: rgba.to_vec(),
        }
    }

    /// Returns the texture width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the texture format stored in this descriptor.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Returns raw RGBA8 pixel bytes.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

impl TextureFormat {
    /// Returns the stable format name used by protocol and asset trace logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8Srgb => "rgba8_srgb",
            Self::Rgba8Linear => "rgba8_linear",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that texture payload size is checked at the protocol boundary.
    #[test]
    fn texture_descriptor_rejects_wrong_byte_count() {
        assert!(TextureDescriptor::rgba8_srgb(1, 1, vec![255, 0, 0]).is_none());
        assert!(TextureDescriptor::rgba8_srgb(1, 1, vec![255, 0, 0, 255]).is_some());
        assert_eq!(
            TextureDescriptor::rgba8_linear(1, 1, vec![128, 128, 255, 255])
                .expect("valid linear texture")
                .format(),
            TextureFormat::Rgba8Linear
        );
    }

    // Verifies that PBR scalar inputs are constrained before crossing into the renderer.
    #[test]
    fn material_descriptor_rejects_invalid_pbr_values() {
        assert!(
            MaterialDescriptor::with_pbr(
                MaterialAlphaMode::Opaque,
                500,
                [1.0, 0.5, 0.25, 1.0],
                100,
                800,
                [0.0, 0.1, 0.2],
                1000,
                1000,
                true,
            )
            .is_some()
        );
        assert!(
            MaterialDescriptor::with_pbr(
                MaterialAlphaMode::Opaque,
                500,
                [1.0, f32::NAN, 0.25, 1.0],
                100,
                800,
                [0.0, 0.1, 0.2],
                1000,
                1000,
                false,
            )
            .is_none()
        );
    }
}
