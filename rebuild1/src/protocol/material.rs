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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialDescriptor {
    alpha_mode: MaterialAlphaMode,
    alpha_cutoff_milli: u16,
    textures: BTreeMap<MaterialTextureSlot, TextureHandle>,
}

impl MaterialDescriptor {
    /// Creates material parameters after import has resolved texture references to handles.
    pub fn new(alpha_mode: MaterialAlphaMode, alpha_cutoff_milli: u16) -> Self {
        Self {
            alpha_mode,
            alpha_cutoff_milli,
            textures: BTreeMap::new(),
        }
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
        let expected = width.checked_mul(height)?.checked_mul(4)? as usize;
        (width > 0 && height > 0 && pixels.len() == expected).then_some(Self {
            width,
            height,
            format: TextureFormat::Rgba8Srgb,
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
    }
}
