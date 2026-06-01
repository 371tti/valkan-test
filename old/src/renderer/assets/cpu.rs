use std::{
    io,
    path::{Path, PathBuf},
};

use crate::renderer::{Material, ModelVertex, TextureId};

#[path = "cpu/gltf.rs"]
mod gltf_import;
#[path = "cpu/obj.rs"]
mod obj_import;

#[derive(Debug, Clone)]
pub struct CpuMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn load_obj(path: impl AsRef<Path>) -> io::Result<Self> {
        let model = CpuModel::load_obj(path)?;
        Ok(model
            .primitives
            .into_iter()
            .next()
            .map(|primitive| primitive.mesh)
            .unwrap_or_else(|| Self {
                vertices: Vec::new(),
                indices: Vec::new(),
            }))
    }

    pub fn from_obj_str(source: &str, path: impl Into<PathBuf>) -> io::Result<Self> {
        let model = CpuModel::from_obj_str(source, path)?;
        Ok(model
            .primitives
            .into_iter()
            .next()
            .map(|primitive| primitive.mesh)
            .unwrap_or_else(|| Self {
                vertices: Vec::new(),
                indices: Vec::new(),
            }))
    }
}

#[derive(Debug, Clone)]
pub struct CpuPrimitive {
    pub mesh: CpuMesh,
    pub material: Material,
}

#[derive(Debug, Clone)]
pub struct CpuTexture {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub sampler: TextureSampler,
    pub srgb: bool,
}

impl CpuTexture {
    pub(in crate::renderer) fn white() -> Self {
        Self {
            pixels: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: true,
        }
    }

    pub(in crate::renderer) fn flat_normal() -> Self {
        Self {
            pixels: vec![128, 128, 255, 255],
            width: 1,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: false,
        }
    }

    fn alpha_usage(&self) -> TextureAlphaUsage {
        let mut has_transparent = false;
        let mut has_partial = false;

        for pixel in self.pixels.chunks_exact(4) {
            match pixel[3] {
                255 => {}
                0 => has_transparent = true,
                _ => {
                    has_transparent = true;
                    has_partial = true;
                }
            }
        }

        match (has_transparent, has_partial) {
            (false, _) => TextureAlphaUsage::Opaque,
            (true, false) => TextureAlphaUsage::Cutout,
            (true, true) => TextureAlphaUsage::Blend,
        }
    }

    fn bleed_alpha_rgb(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;
        if width == 0 || height == 0 || self.pixels.len() != width * height * 4 {
            return;
        }

        let mut pixels = self.pixels.clone();
        for _ in 0..8 {
            let source = pixels.clone();
            let mut changed = false;

            for y in 0..height {
                for x in 0..width {
                    let index = (y * width + x) * 4;
                    if source[index + 3] >= 250 {
                        continue;
                    }

                    let mut sum = [0_u32; 3];
                    let mut count = 0_u32;
                    let y0 = y.saturating_sub(1);
                    let y1 = (y + 1).min(height - 1);
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(width - 1);
                    for ny in y0..=y1 {
                        for nx in x0..=x1 {
                            if nx == x && ny == y {
                                continue;
                            }

                            let neighbor = (ny * width + nx) * 4;
                            if source[neighbor + 3] == 0
                                || source[neighbor + 3] <= source[index + 3]
                            {
                                continue;
                            }

                            sum[0] += source[neighbor] as u32;
                            sum[1] += source[neighbor + 1] as u32;
                            sum[2] += source[neighbor + 2] as u32;
                            count += 1;
                        }
                    }

                    if count == 0 {
                        continue;
                    }

                    let rgb = [
                        (sum[0] / count) as u8,
                        (sum[1] / count) as u8,
                        (sum[2] / count) as u8,
                    ];
                    if pixels[index] != rgb[0]
                        || pixels[index + 1] != rgb[1]
                        || pixels[index + 2] != rgb[2]
                    {
                        pixels[index..index + 3].copy_from_slice(&rgb);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        self.pixels = pixels;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureAlphaUsage {
    Opaque,
    Cutout,
    Blend,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureSampler {
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub wrap_s: TextureWrap,
    pub wrap_t: TextureWrap,
}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            wrap_s: TextureWrap::Repeat,
            wrap_t: TextureWrap::Repeat,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum TextureWrap {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
}

#[derive(Debug, Clone)]
pub struct CpuModel {
    pub primitives: Vec<CpuPrimitive>,
    pub textures: Vec<CpuTexture>,
}

impl CpuModel {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();

        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension) if extension.eq_ignore_ascii_case("gltf") => Self::load_gltf(path),
            Some(extension) if extension.eq_ignore_ascii_case("glb") => Self::load_gltf(path),
            Some(extension) if extension.eq_ignore_ascii_case("obj") => Self::load_obj(path),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported model format: {}", path.display()),
            )),
        }
    }

    pub fn load_gltf(path: impl AsRef<Path>) -> io::Result<Self> {
        gltf_import::load(path.as_ref())
    }

    pub fn load_obj(path: impl AsRef<Path>) -> io::Result<Self> {
        obj_import::load(path.as_ref())
    }

    pub fn from_obj_str(source: &str, path: impl Into<PathBuf>) -> io::Result<Self> {
        obj_import::parse(source, path.into())
    }
}

pub(super) fn mark_texture_linear(textures: &mut [CpuTexture], texture: TextureId) {
    if let Some(texture) = textures.get_mut(texture.0) {
        texture.srgb = false;
    }
}

pub(super) fn prepare_base_color_texture(textures: &mut [CpuTexture], texture: TextureId) {
    if let Some(texture) = textures.get_mut(texture.0) {
        texture.bleed_alpha_rgb();
    }
}

pub(super) fn infer_base_color_alpha(
    material: Material,
    texture: TextureId,
    textures: &[CpuTexture],
) -> Material {
    if material.alpha_blend() || material.alpha_cutoff() > f32::EPSILON {
        return material;
    }

    match textures.get(texture.0).map(CpuTexture::alpha_usage) {
        Some(TextureAlphaUsage::Cutout) => material.with_alpha_cutoff(0.5),
        Some(TextureAlphaUsage::Blend) => material.with_alpha_blend(true),
        Some(TextureAlphaUsage::Opaque) | None => material,
    }
}

pub(super) fn invalid_model<T>(message: &'static str) -> io::Result<T> {
    Err(invalid_model_error(message))
}

pub(super) fn invalid_model_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(super) fn model_error(path: &Path, source: gltf::Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {source}", path.display()),
    )
}

pub(super) fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize_or(
        [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ],
        [0.0, 1.0, 0.0],
    )
}

pub(super) fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return fallback;
    }

    [v[0] / len, v[1] / len, v[2] / len]
}
