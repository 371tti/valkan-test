use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

use crate::renderer::{Material, ModelVertex, TextureId};

use super::{
    CpuMesh, CpuModel, CpuPrimitive, CpuTexture, TextureSampler, infer_base_color_alpha,
    prepare_base_color_texture, triangle_normal,
};

pub(super) fn load(path: &Path) -> io::Result<CpuModel> {
    parse(&fs::read_to_string(path)?, path.to_path_buf())
}

pub(super) fn parse(source: &str, path: PathBuf) -> io::Result<CpuModel> {
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

                let builder_index = *builder_indices.entry(current.clone()).or_insert_with(|| {
                    materials.entry(current.clone()).or_default();
                    builders.push(PrimitiveBuilder::new(current.clone()));
                    builders.len() - 1
                });
                let builder = &mut builders[builder_index];

                for i in 1..face.len() - 1 {
                    builder.triangle([face[0], face[i], face[i + 1]], &positions, &uvs, &normals);
                }
            }
            _ => {}
        }
    }

    Ok(CpuModel {
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

fn obj_error(path: &Path, line: usize, message: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}:{}: {message}", path.display(), line + 1),
    )
}
