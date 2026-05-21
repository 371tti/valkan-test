use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

use crate::renderer::{Material, ModelVertex, TextureId};

#[path = "cpu/gltf.rs"]
mod gltf_import;

#[derive(Debug, Clone)]
pub struct CpuMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn cube() -> Self {
        Self {
            vertices: CUBE_VERTICES.to_vec(),
            indices: CUBE_INDICES.to_vec(),
        }
    }

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
    pub fn cube() -> Self {
        Self {
            primitives: vec![CpuPrimitive {
                mesh: CpuMesh::cube(),
                material: Material::new([0.18, 0.48, 1.0, 1.0]).with_roughness(0.45),
            }],
            textures: Vec::new(),
        }
    }

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
        let path = path.as_ref();
        Self::from_obj_str(&fs::read_to_string(path)?, path)
    }

    pub fn from_obj_str(source: &str, path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let base_dir = path.parent().unwrap_or(Path::new("."));
        let mut positions = Vec::<[f32; 3]>::new();
        let mut normals = Vec::<[f32; 3]>::new();
        let mut uvs = Vec::<[f32; 2]>::new();
        let mut materials = HashMap::<String, Material>::new();
        let mut textures = Vec::<CpuTexture>::new();
        let mut builder_indices = HashMap::<String, usize>::new();
        let mut builders = Vec::<PrimitiveBuilder>::new();
        let mut current = String::from("default");
        materials.insert(current.clone(), Material::default());
        builder_indices.insert(current.clone(), builders.len());
        builders.push(PrimitiveBuilder::new(current.clone()));

        for (line_index, line) in source.lines().enumerate() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((tag, rest)) = split_statement(line) else {
                continue;
            };
            let parts = rest.split_whitespace();

            match tag {
                "mtllib" => {
                    load_mtl_libraries(base_dir, rest, &mut materials, &mut textures)?;
                }
                "usemtl" => {
                    current = if rest.is_empty() {
                        String::from("default")
                    } else {
                        rest.to_string()
                    };
                }
                "v" => positions.push(parse_vec3(parts, &path, line_index)?),
                "vn" => normals.push(parse_vec3(parts, &path, line_index)?),
                "vt" => {
                    let uv = parse_vec2(parts, &path, line_index)?;
                    uvs.push([uv[0], 1.0 - uv[1]]);
                }
                "f" => {
                    let face = parts
                        .map(|part| parse_face_ref(part, positions.len(), uvs.len(), normals.len()))
                        .collect::<io::Result<Vec<_>>>()?;

                    if face.len() < 3 {
                        return invalid_obj(&path, line_index, "face has fewer than 3 vertices");
                    }

                    let builder_index =
                        *builder_indices.entry(current.clone()).or_insert_with(|| {
                            materials.entry(current.clone()).or_default();
                            builders.push(PrimitiveBuilder::new(current.clone()));
                            builders.len() - 1
                        });
                    let builder = &mut builders[builder_index];

                    for i in 1..face.len() - 1 {
                        builder.triangle(
                            [face[0], face[i], face[i + 1]],
                            &positions,
                            &uvs,
                            &normals,
                        );
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            primitives: builders
                .into_iter()
                .filter_map(|builder| {
                    (!builder.mesh.vertices.is_empty()).then(|| CpuPrimitive {
                        material: materials
                            .get(&builder.material)
                            .copied()
                            .unwrap_or_default(),
                        mesh: builder.mesh.finish(),
                    })
                })
                .collect(),
            textures,
        })
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

struct PrimitiveBuilder {
    material: String,
    mesh: MeshBuilder,
}

impl PrimitiveBuilder {
    fn new(material: String) -> Self {
        Self {
            material,
            mesh: MeshBuilder::default(),
        }
    }

    fn triangle(
        &mut self,
        refs: [ObjVertexRef; 3],
        positions: &[[f32; 3]],
        uvs: &[[f32; 2]],
        normals: &[[f32; 3]],
    ) {
        self.mesh.triangle(refs, positions, uvs, normals);
    }
}

#[derive(Default)]
struct MeshBuilder {
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,
    vertex_cache: HashMap<VertexKey, u32>,
}

impl MeshBuilder {
    fn triangle(
        &mut self,
        refs: [ObjVertexRef; 3],
        positions: &[[f32; 3]],
        uvs: &[[f32; 2]],
        normals: &[[f32; 3]],
    ) {
        let face_normal = triangle_normal(
            positions[refs[0].position],
            positions[refs[1].position],
            positions[refs[2].position],
        );

        for vertex_ref in refs {
            self.vertex(vertex_ref, positions, uvs, normals, face_normal);
        }
    }

    fn vertex(
        &mut self,
        vertex_ref: ObjVertexRef,
        positions: &[[f32; 3]],
        uvs: &[[f32; 2]],
        normals: &[[f32; 3]],
        face_normal: [f32; 3],
    ) {
        let normal = vertex_ref
            .normal
            .map(|index| normals[index])
            .unwrap_or(face_normal);
        let vertex = ModelVertex {
            position: positions[vertex_ref.position],
            normal,
            uv: vertex_ref.uv.map(|index| uvs[index]).unwrap_or([0.0; 2]),
        };

        let key = vertex_ref.normal.map(|normal| VertexKey {
            position: vertex_ref.position,
            uv: vertex_ref.uv,
            normal,
        });

        if let Some(key) = key {
            if let Some(index) = self.vertex_cache.get(&key) {
                self.indices.push(*index);
                return;
            }

            let index = self.push_unique(vertex);
            self.vertex_cache.insert(key, index);
            return;
        }

        self.push_unique(vertex);
    }

    fn push_unique(&mut self, vertex: ModelVertex) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        self.indices.push(index);
        index
    }

    fn finish(self) -> CpuMesh {
        CpuMesh {
            vertices: self.vertices,
            indices: self.indices,
        }
    }
}

fn split_statement(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();

    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let split = line
        .char_indices()
        .find_map(|(index, ch)| ch.is_whitespace().then_some(index))
        .unwrap_or(line.len());

    Some((&line[..split], line[split..].trim()))
}

fn load_mtl_libraries(
    base_dir: &Path,
    spec: &str,
    materials: &mut HashMap<String, Material>,
    textures: &mut Vec<CpuTexture>,
) -> io::Result<()> {
    if spec.is_empty() {
        return Ok(());
    }

    let direct_path = base_dir.join(spec);
    if direct_path.exists() {
        return load_mtl(direct_path, materials, textures);
    }

    for name in spec.split_whitespace() {
        load_mtl(base_dir.join(name), materials, textures)?;
    }

    Ok(())
}

fn resolve_mtl_texture_path(base_dir: &Path, spec: &str) -> Option<PathBuf> {
    let spec = spec.trim();

    if spec.is_empty() {
        return None;
    }

    let direct_path = base_dir.join(spec);
    if direct_path.exists() {
        return Some(direct_path);
    }

    let tokens = spec.split_whitespace().collect::<Vec<_>>();
    let mut first_path = 0;

    while first_path < tokens.len() && tokens[first_path].starts_with('-') {
        let option = tokens[first_path];
        first_path += 1 + mtl_texture_option_args(option);
    }

    if first_path >= tokens.len() {
        return None;
    }

    let joined = tokens[first_path..].join(" ");
    let joined_path = base_dir.join(&joined);
    if joined_path.exists() {
        return Some(joined_path);
    }

    tokens.last().map(|name| base_dir.join(name))
}

fn mtl_texture_option_args(option: &str) -> usize {
    match option {
        "-mm" => 2,
        "-o" | "-s" | "-t" => 3,
        "-blendu" | "-blendv" | "-boost" | "-texres" | "-clamp" | "-bm" | "-imfchan" | "-type" => 1,
        _ => 0,
    }
}

fn load_mtl(
    path: PathBuf,
    materials: &mut HashMap<String, Material>,
    textures: &mut Vec<CpuTexture>,
) -> io::Result<()> {
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    let mut current = None::<String>;

    for line in source.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((tag, rest)) = split_statement(line) else {
            continue;
        };
        let parts = rest.split_whitespace();

        match tag {
            "newmtl" => {
                let name = if rest.is_empty() {
                    String::from("default")
                } else {
                    rest.to_string()
                };
                materials.entry(name.clone()).or_default();
                current = Some(name);
            }
            "Ka" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let color = parse_mtl_rgb(parts, [1.0; 3]);
                    material.ambient_occlusion =
                        ((color[0] + color[1] + color[2]) / 3.0).clamp(0.0, 1.0);
                }
            }
            "Kd" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let color = parse_mtl_rgb(parts, [1.0; 3]);
                    material.base_color[0] = color[0];
                    material.base_color[1] = color[1];
                    material.base_color[2] = color[2];
                }
            }
            "Ks" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let color = parse_mtl_rgb(parts, [0.5; 3]);
                    material.specular = ((color[0] + color[1] + color[2]) / 3.0).clamp(0.0, 1.0);
                }
            }
            "Ke" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.emissive_color = parse_mtl_rgb(parts, [0.0; 3]);
                }
            }
            "illum" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let illumination_model = parse_mtl_scalar(parts, 2.0);
                    if illumination_model <= 1.0 {
                        apply_matte_defaults(material);
                    }
                }
            }
            "Ns" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.roughness = shininess_to_roughness(parse_mtl_scalar(parts, 32.0));
                }
            }
            "d" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let alpha = parse_mtl_scalar(parts, 1.0).clamp(0.0, 1.0);
                    material.base_color[3] = alpha;
                    *material = material.with_alpha_blend(alpha < 0.999);
                }
            }
            "Tr" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    let alpha = 1.0 - parse_mtl_scalar(parts, 0.0).clamp(0.0, 1.0);
                    material.base_color[3] = alpha;
                    *material = material.with_alpha_blend(alpha < 0.999);
                }
            }
            "Pm" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.metallic = parse_mtl_scalar(parts, 0.0).clamp(0.0, 1.0);
                }
            }
            "Pr" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.roughness = parse_mtl_scalar(parts, 0.55).clamp(0.04, 1.0);
                }
            }
            "Ps" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.specular = parse_mtl_scalar(parts, 0.5).clamp(0.0, 1.0);
                }
            }
            "map_Kd" => {
                let Some(material) = current_material_mut(&current, materials) else {
                    continue;
                };
                let Some(texture_path) =
                    resolve_mtl_texture_path(path.parent().unwrap_or(Path::new(".")), rest)
                else {
                    continue;
                };
                match load_image_texture(&texture_path) {
                    Ok(texture) => {
                        let texture_id = TextureId(textures.len());
                        textures.push(texture);
                        prepare_base_color_texture(textures, texture_id);
                        material.textures.base_color = Some(texture_id);
                        *material = infer_base_color_alpha(*material, texture_id, textures);
                    }
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => return Err(err),
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn load_image_texture(path: &Path) -> io::Result<CpuTexture> {
    let image = image::ImageReader::open(path)
        .map_err(|source| io::Error::new(source.kind(), format!("{}: {source}", path.display())))?
        .decode()
        .map_err(|source| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {source}", path.display()),
            )
        })?
        .to_rgba8();
    let (width, height) = image.dimensions();

    Ok(CpuTexture {
        pixels: image.into_raw(),
        width: width.max(1),
        height: height.max(1),
        sampler: TextureSampler::default(),
        srgb: true,
    })
}

fn current_material_mut<'a>(
    current: &Option<String>,
    materials: &'a mut HashMap<String, Material>,
) -> Option<&'a mut Material> {
    let name = current.as_ref()?;
    materials.get_mut(name)
}

fn parse_mtl_rgb<'a>(mut parts: impl Iterator<Item = &'a str>, default: [f32; 3]) -> [f32; 3] {
    [
        parse_mtl_value(parts.next(), default[0]),
        parse_mtl_value(parts.next(), default[1]),
        parse_mtl_value(parts.next(), default[2]),
    ]
}

fn parse_mtl_scalar<'a>(mut parts: impl Iterator<Item = &'a str>, default: f32) -> f32 {
    parse_mtl_value(parts.next(), default)
}

fn apply_matte_defaults(material: &mut Material) {
    material.metallic = 0.0;
    material.roughness = material.roughness.max(0.92);
    material.specular = material.specular.min(0.08);
}

fn parse_mtl_value(value: Option<&str>, default: f32) -> f32 {
    value
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn shininess_to_roughness(shininess: f32) -> f32 {
    (2.0 / (shininess.max(0.0) + 2.0)).sqrt().clamp(0.04, 1.0)
}

fn parse_vec3<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    path: &Path,
    line: usize,
) -> io::Result<[f32; 3]> {
    Ok([
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
    ])
}

fn parse_vec2<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    path: &Path,
    line: usize,
) -> io::Result<[f32; 2]> {
    Ok([
        parse_f32(parts.next(), path, line)?,
        parse_f32(parts.next(), path, line)?,
    ])
}

fn parse_f32(value: Option<&str>, path: &Path, line: usize) -> io::Result<f32> {
    value
        .ok_or_else(|| obj_error(path, line, "missing float"))?
        .parse()
        .map_err(|_| obj_error(path, line, "invalid float"))
}

#[derive(Debug, Clone, Copy)]
struct ObjVertexRef {
    position: usize,
    uv: Option<usize>,
    normal: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VertexKey {
    position: usize,
    uv: Option<usize>,
    normal: usize,
}

fn parse_face_ref(
    part: &str,
    position_len: usize,
    uv_len: usize,
    normal_len: usize,
) -> io::Result<ObjVertexRef> {
    let mut ids = part.split('/');
    let position = obj_index(ids.next(), position_len)?;
    let uv = ids
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| obj_index(Some(value), uv_len))
        .transpose()?;
    let normal = ids
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| obj_index(Some(value), normal_len))
        .transpose()?;

    Ok(ObjVertexRef {
        position,
        uv,
        normal,
    })
}

fn obj_index(value: Option<&str>, len: usize) -> io::Result<usize> {
    let index = value
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing obj index"))?
        .parse::<isize>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid obj index"))?;

    let resolved = if index > 0 {
        index - 1
    } else if index < 0 {
        len as isize + index
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "obj indices are 1-based",
        ));
    };

    if resolved < 0 || resolved >= len as isize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "obj index out of bounds",
        ));
    }

    Ok(resolved as usize)
}

fn invalid_obj<T>(path: &Path, line: usize, message: &'static str) -> io::Result<T> {
    Err(obj_error(path, line, message))
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

fn obj_error(path: &Path, line: usize, message: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}:{}: {message}", path.display(), line + 1),
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

const S: f32 = 0.75;

const CUBE_VERTICES: [ModelVertex; 24] = [
    ModelVertex {
        position: [-S, -S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [-S, -S, S],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, S],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, S, S],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, S, -S],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, S, -S],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, -S],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, -S],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 1.0],
    },
    ModelVertex {
        position: [S, -S, S],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 0.0],
    },
    ModelVertex {
        position: [-S, -S, S],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 0.0],
    },
];

const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17, 18,
    18, 19, 16, 20, 21, 22, 22, 23, 20,
];

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn obj_faces_without_normals_get_face_normals() {
        let mesh = CpuMesh::from_obj_str(
            "
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
",
            "inline.obj",
        )
        .unwrap();

        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.vertices.len(), 3);
        for vertex in &mesh.vertices {
            assert_close(vertex.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn obj_faces_with_indices_share_vertices() {
        let mesh = CpuMesh::from_obj_str(
            "
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1 4//1
",
            "inline.obj",
        )
        .unwrap();

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn obj_loads_mtl_surface_values() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("demo.mtl"),
            "
newmtl copper
Kd 0.8 0.4 0.2
Ke 0.1 0.2 0.3
Pm 0.9
Pr 0.25
Ps 0.7
d 0.8
",
        )
        .unwrap();

        let model = CpuModel::from_obj_str(
            "
mtllib demo.mtl
usemtl copper
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
",
            dir.join("demo.obj"),
        )
        .unwrap();
        let material = model.primitives[0].material;

        assert_close(material.base_color, [0.8, 0.4, 0.2, 0.8]);
        assert_close(material.emissive_color, [0.1, 0.2, 0.3]);
        assert!((material.metallic - 0.9).abs() < 0.001);
        assert!((material.roughness - 0.25).abs() < 0.001);
        assert!((material.specular - 0.7).abs() < 0.001);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn obj_loads_space_separated_mtl_and_texture_names() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-spaces-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        let texture = image::RgbaImage::from_raw(1, 1, vec![32, 64, 128, 255]).unwrap();
        texture.save(dir.join("brick wall.png")).unwrap();
        fs::write(
            dir.join("Castelia City.mtl"),
            "
newmtl city material
Kd 0.2 0.3 0.4
map_Kd -s 1 1 1 brick wall.png
",
        )
        .unwrap();

        let model = CpuModel::from_obj_str(
            "
mtllib Castelia City.mtl
usemtl city material
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
",
            dir.join("Castelia City.obj"),
        )
        .unwrap();
        let material = model.primitives[0].material;

        assert_eq!(model.textures.len(), 1);
        assert_eq!(material.base_color_texture(), Some(TextureId(0)));
        assert_close(material.base_color, [0.2, 0.3, 0.4, 1.0]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn obj_base_color_texture_with_binary_alpha_uses_cutout() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-alpha-cutout-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        let texture = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
        texture.save(dir.join("cutout.png")).unwrap();
        fs::write(
            dir.join("demo.mtl"),
            "
newmtl cutout
map_Kd cutout.png
",
        )
        .unwrap();

        let model = CpuModel::from_obj_str(
            "
mtllib demo.mtl
usemtl cutout
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
",
            dir.join("demo.obj"),
        )
        .unwrap();
        let material = model.primitives[0].material;

        assert_eq!(material.alpha_cutoff(), 0.5);
        assert!(!material.alpha_blend());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn partial_alpha_base_color_texture_uses_blend() {
        let textures = vec![CpuTexture {
            pixels: vec![255, 255, 255, 255, 255, 255, 255, 128],
            width: 2,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: true,
        }];
        let material = infer_base_color_alpha(Material::default(), TextureId(0), &textures);

        assert!(material.alpha_blend());
        assert_eq!(material.alpha_cutoff(), 0.0);
    }

    #[test]
    fn transparent_base_color_pixels_keep_neighbor_rgb() {
        let mut texture = CpuTexture {
            pixels: vec![
                220, 40, 10, 255, //
                0, 0, 0, 96, //
                0, 0, 0, 0,
            ],
            width: 3,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: true,
        };

        texture.bleed_alpha_rgb();

        assert_eq!(&texture.pixels[4..8], &[220, 40, 10, 96]);
        assert_eq!(&texture.pixels[8..12], &[220, 40, 10, 0]);
    }

    #[test]
    fn obj_material_names_do_not_force_metallic() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-material-name-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("demo.mtl"),
            "
newmtl gold hair
Kd 0.8 0.7 0.2
",
        )
        .unwrap();

        let model = CpuModel::from_obj_str(
            "
mtllib demo.mtl
usemtl gold hair
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
",
            dir.join("demo.obj"),
        )
        .unwrap();

        assert_eq!(model.primitives[0].material.metallic, 0.0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn obj_illum_one_uses_matte_defaults() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-matte-illum-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("demo.mtl"),
            "
newmtl matte_wall
Kd 0.7 0.7 0.7
illum 1
",
        )
        .unwrap();

        let model = CpuModel::from_obj_str(
            "
mtllib demo.mtl
usemtl matte_wall
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
",
            dir.join("demo.obj"),
        )
        .unwrap();
        let material = model.primitives[0].material;

        assert_eq!(material.metallic, 0.0);
        assert!(material.roughness >= 0.92);
        assert!(material.specular <= 0.08);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gltf_loads_mesh_and_pbr_material() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-gltf-{stamp}"));
        fs::create_dir_all(&dir).unwrap();

        let mut bin = Vec::new();
        for value in [
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // POSITION
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // NORMAL
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // TEXCOORD_0
        ] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0_u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        fs::write(dir.join("mesh.bin"), &bin).unwrap();
        let texture = image::RgbaImage::from_raw(1, 1, vec![255, 128, 64, 255]).unwrap();
        texture.save(dir.join("tex.png")).unwrap();
        fs::write(
            dir.join("model.gltf"),
            format!(
                r#"{{
    "asset": {{"version": "2.0"}},
    "extensionsUsed": ["KHR_materials_emissive_strength", "KHR_materials_specular"],
    "scene": 0,
    "scenes": [{{"nodes": [0]}}],
    "nodes": [{{"mesh": 0}}],
    "meshes": [{{
        "primitives": [{{
            "attributes": {{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}},
            "indices": 3,
            "material": 0
        }}]
    }}],
    "materials": [{{
        "pbrMetallicRoughness": {{
            "baseColorFactor": [0.2, 0.4, 0.8, 0.75],
            "baseColorTexture": {{"index": 0}},
            "metallicFactor": 0.3,
            "roughnessFactor": 0.7,
            "metallicRoughnessTexture": {{"index": 1}}
        }},
        "normalTexture": {{"index": 2, "scale": 0.5}},
        "occlusionTexture": {{"index": 3, "strength": 0.25}},
        "emissiveTexture": {{"index": 4}},
        "emissiveFactor": [0.1, 0.0, 0.2],
        "alphaMode": "MASK",
        "alphaCutoff": 0.42,
        "doubleSided": true,
        "extensions": {{
            "KHR_materials_emissive_strength": {{"emissiveStrength": 2.5}},
            "KHR_materials_specular": {{
                "specularFactor": 0.8,
                "specularColorFactor": [0.9, 0.8, 0.7]
            }}
        }}
    }}],
    "images": [{{"uri": "tex.png"}}],
    "textures": [
        {{"source": 0}},
        {{"source": 0}},
        {{"source": 0}},
        {{"source": 0}},
        {{"source": 0}}
    ],
    "buffers": [{{"uri": "mesh.bin", "byteLength": {}}}],
    "bufferViews": [
        {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
        {{"buffer": 0, "byteOffset": 36, "byteLength": 36}},
        {{"buffer": 0, "byteOffset": 72, "byteLength": 24}},
        {{"buffer": 0, "byteOffset": 96, "byteLength": 6}}
    ],
    "accessors": [
        {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
        {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
        {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}},
        {{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}}
    ]
}}"#,
                bin.len()
            ),
        )
        .unwrap();

        let model = CpuModel::load_gltf(dir.join("model.gltf")).unwrap();
        let primitive = &model.primitives[0];

        assert_eq!(model.primitives.len(), 1);
        assert_eq!(model.textures.len(), 5);
        assert!(model.textures[0].srgb);
        assert!(!model.textures[1].srgb);
        assert!(!model.textures[2].srgb);
        assert!(!model.textures[3].srgb);
        assert!(model.textures[4].srgb);
        assert_eq!(primitive.mesh.vertices.len(), 3);
        assert_eq!(primitive.mesh.indices, vec![0, 1, 2]);
        assert_close(primitive.material.base_color, [0.2, 0.4, 0.8, 0.75]);
        assert_close(primitive.material.emissive_color, [0.1, 0.0, 0.2]);
        assert!((primitive.material.emissive_strength - 2.5).abs() < 0.001);
        assert!((primitive.material.metallic - 0.3).abs() < 0.001);
        assert!((primitive.material.roughness - 0.7).abs() < 0.001);
        assert!((primitive.material.specular - 0.8).abs() < 0.001);
        assert_close(primitive.material.specular_color, [0.9, 0.8, 0.7]);
        assert!((primitive.material.alpha_cutoff() - 0.42).abs() < 0.001);
        assert!(primitive.material.double_sided);
        assert_eq!(primitive.material.base_color_texture(), Some(TextureId(0)));
        assert_eq!(
            primitive.material.metallic_roughness_texture(),
            Some(TextureId(1))
        );
        assert_eq!(primitive.material.normal_texture(), Some(TextureId(2)));
        assert_eq!(primitive.material.occlusion_texture(), Some(TextureId(3)));
        assert_eq!(primitive.material.emissive_texture(), Some(TextureId(4)));
        assert!((primitive.material.normal_scale - 0.5).abs() < 0.001);
        assert!((primitive.material.occlusion_strength - 0.25).abs() < 0.001);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gltf_missing_material_uses_non_metal_default() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-gltf-default-material-{stamp}"));
        fs::create_dir_all(&dir).unwrap();

        let mut bin = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0_u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        fs::write(dir.join("mesh.bin"), &bin).unwrap();
        fs::write(
            dir.join("model.gltf"),
            format!(
                r#"{{
    "asset": {{"version": "2.0"}},
    "scene": 0,
    "scenes": [{{"nodes": [0]}}],
    "nodes": [{{"mesh": 0}}],
    "meshes": [{{
        "primitives": [{{
            "attributes": {{"POSITION": 0}},
            "indices": 1
        }}]
    }}],
    "buffers": [{{"uri": "mesh.bin", "byteLength": {}}}],
    "bufferViews": [
        {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
        {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
    ],
    "accessors": [
        {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
        {{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}}
    ]
}}"#,
                bin.len()
            ),
        )
        .unwrap();

        let model = CpuModel::load_gltf(dir.join("model.gltf")).unwrap();
        let material = model.primitives[0].material;

        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.roughness, Material::default().roughness);
        assert_eq!(material.specular, Material::default().specular);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gltf_skin_bakes_first_animation_pose() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("valkan-test-gltf-skin-{stamp}"));
        fs::create_dir_all(&dir).unwrap();

        let mut bin = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for joints in [[0_u16, 0, 0, 0]; 3] {
            for joint in joints {
                bin.extend_from_slice(&joint.to_le_bytes());
            }
        }
        for weights in [[1.0_f32, 0.0, 0.0, 0.0]; 3] {
            for weight in weights {
                bin.extend_from_slice(&weight.to_le_bytes());
            }
        }
        for value in [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
        ] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        bin.extend_from_slice(&0.0_f32.to_le_bytes());
        for value in [4.0_f32, 0.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0_u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }

        fs::write(dir.join("mesh.bin"), &bin).unwrap();
        fs::write(
            dir.join("model.gltf"),
            format!(
                r#"{{
    "asset": {{"version": "2.0"}},
    "scene": 0,
    "scenes": [{{"nodes": [0, 1]}}],
    "animations": [{{
        "samplers": [{{"input": 4, "output": 5, "interpolation": "STEP"}}],
        "channels": [{{"sampler": 0, "target": {{"node": 1, "path": "translation"}}}}]
    }}],
    "nodes": [
        {{"mesh": 0, "skin": 0, "translation": [5, 0, 0]}},
        {{"translation": [2, 0, 0]}}
    ],
    "skins": [{{"joints": [1], "inverseBindMatrices": 3}}],
    "meshes": [{{
        "primitives": [{{
            "attributes": {{"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}},
            "indices": 6
        }}]
    }}],
    "buffers": [{{"uri": "mesh.bin", "byteLength": {}}}],
    "bufferViews": [
        {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
        {{"buffer": 0, "byteOffset": 36, "byteLength": 24}},
        {{"buffer": 0, "byteOffset": 60, "byteLength": 48}},
        {{"buffer": 0, "byteOffset": 108, "byteLength": 64}},
        {{"buffer": 0, "byteOffset": 172, "byteLength": 4}},
        {{"buffer": 0, "byteOffset": 176, "byteLength": 12}},
        {{"buffer": 0, "byteOffset": 188, "byteLength": 6}}
    ],
    "accessors": [
        {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
        {{"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4"}},
        {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"}},
        {{"bufferView": 3, "componentType": 5126, "count": 1, "type": "MAT4"}},
        {{"bufferView": 4, "componentType": 5126, "count": 1, "type": "SCALAR", "min": [0], "max": [0]}},
        {{"bufferView": 5, "componentType": 5126, "count": 1, "type": "VEC3"}},
        {{"bufferView": 6, "componentType": 5123, "count": 3, "type": "SCALAR"}}
    ]
}}"#,
                bin.len()
            ),
        )
        .unwrap();

        let model = CpuModel::load_gltf(dir.join("model.gltf")).unwrap();
        let vertices = &model.primitives[0].mesh.vertices;

        assert_close(vertices[0].position, [3.0, 0.0, 0.0]);
        assert_close(vertices[1].position, [4.0, 0.0, 0.0]);
        assert_close(vertices[2].position, [3.0, 1.0, 0.0]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bundled_model_loads_when_present() {
        let path = Path::new("assets/model.glb");

        if !path.exists() {
            return;
        }

        let model = CpuModel::load(path).unwrap();
        assert!(!model.primitives.is_empty());
        for texture in &model.textures {
            assert_eq!(
                texture.pixels.len(),
                texture.width as usize * texture.height as usize * 4
            );
        }
    }

    fn assert_close<const N: usize>(actual: [f32; N], expected: [f32; N]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.001,
                "expected {expected}, got {actual}"
            );
        }
    }
}
