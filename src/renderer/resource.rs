use std::{
    collections::HashMap,
    fs, io, mem,
    path::{Path, PathBuf},
};

use ash::{Instance, vk};

use super::{MAX_FRAMES_IN_FLIGHT, Material, MaterialId, MeshId, ModelId, ModelVertex, TextureId};

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
    fn white() -> Self {
        Self {
            pixels: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: true,
        }
    }

    fn flat_normal() -> Self {
        Self {
            pixels: vec![128, 128, 255, 255],
            width: 1,
            height: 1,
            sampler: TextureSampler::default(),
            srgb: false,
        }
    }
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
        let path = path.as_ref();
        let (document, buffers, images) =
            gltf::import(path).map_err(|source| model_error(path, source))?;
        let mut primitives = Vec::new();
        let mut textures = document
            .textures()
            .map(|texture| gltf_texture_to_cpu(texture, &images))
            .collect::<io::Result<Vec<_>>>()?;

        if let Some(scene) = document.default_scene() {
            for node in scene.nodes() {
                load_gltf_node(
                    node,
                    MAT4_IDENTITY,
                    &buffers,
                    &mut textures,
                    &mut primitives,
                )?;
            }
        } else {
            for scene in document.scenes() {
                for node in scene.nodes() {
                    load_gltf_node(
                        node,
                        MAT4_IDENTITY,
                        &buffers,
                        &mut textures,
                        &mut primitives,
                    )?;
                }
            }
        }

        if primitives.is_empty() {
            for mesh in document.meshes() {
                load_gltf_mesh(
                    mesh,
                    MAT4_IDENTITY,
                    &buffers,
                    &mut textures,
                    &mut primitives,
                )?;
            }
        }

        Ok(Self {
            primitives,
            textures,
        })
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

const MAT4_IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn load_gltf_node(
    node: gltf::Node<'_>,
    parent_transform: [f32; 16],
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
    primitives: &mut Vec<CpuPrimitive>,
) -> io::Result<()> {
    let transform = mat4_mul(parent_transform, gltf_node_transform(node.transform()));

    if let Some(mesh) = node.mesh() {
        load_gltf_mesh(mesh, transform, buffers, textures, primitives)?;
    }

    for child in node.children() {
        load_gltf_node(child, transform, buffers, textures, primitives)?;
    }

    Ok(())
}

fn load_gltf_mesh(
    mesh: gltf::Mesh<'_>,
    transform: [f32; 16],
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
    primitives: &mut Vec<CpuPrimitive>,
) -> io::Result<()> {
    for primitive in mesh.primitives() {
        if let Some(primitive) = load_gltf_primitive(primitive, transform, buffers, textures)? {
            primitives.push(primitive);
        }
    }

    Ok(())
}

fn load_gltf_primitive(
    primitive: gltf::Primitive<'_>,
    transform: [f32; 16],
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
) -> io::Result<Option<CpuPrimitive>> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Ok(None);
    }

    let material = material_from_gltf(primitive.material(), textures);
    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let positions = reader
        .read_positions()
        .ok_or_else(|| invalid_model_error("glTF primitive is missing POSITION"))?
        .map(|position| transform_position(transform, position))
        .collect::<Vec<_>>();
    let normals = reader
        .read_normals()
        .map(|normals| {
            normals
                .map(|normal| transform_normal(transform, normal))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let uvs = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_default();
    let indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<_>>(),
        None => sequential_indices(positions.len())?,
    };

    if indices.len() % 3 != 0 {
        return invalid_model("glTF triangle index count is not divisible by 3");
    }

    let has_normals = normals.len() == positions.len();
    let mut vertices = positions
        .into_iter()
        .enumerate()
        .map(|(index, position)| ModelVertex {
            position,
            normal: normals.get(index).copied().unwrap_or([0.0; 3]),
            uv: uvs.get(index).copied().unwrap_or([0.0; 2]),
        })
        .collect::<Vec<_>>();

    validate_indices(&indices, vertices.len())?;

    if !has_normals {
        accumulate_normals(&mut vertices, &indices);
    }

    Ok(Some(CpuPrimitive {
        mesh: CpuMesh { vertices, indices },
        material,
    }))
}

fn material_from_gltf(material: gltf::Material<'_>, textures: &mut [CpuTexture]) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let alpha_cutoff = match material.alpha_mode() {
        gltf::material::AlphaMode::Mask => material.alpha_cutoff().unwrap_or(0.5),
        _ => 0.0,
    };
    let base_color_texture = pbr
        .base_color_texture()
        .map(|info| TextureId(info.texture().index()));
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .map(|info| TextureId(info.texture().index()));
    let normal_texture = material.normal_texture();
    let occlusion_texture = material.occlusion_texture();
    let emissive_texture = material
        .emissive_texture()
        .map(|info| TextureId(info.texture().index()));

    let mut material = Material::new(pbr.base_color_factor())
        .with_metallic(pbr.metallic_factor())
        .with_roughness(pbr.roughness_factor())
        .with_emissive(material.emissive_factor())
        .with_alpha_cutoff(alpha_cutoff);

    if let Some(texture) = base_color_texture {
        material = material.with_base_color_texture(texture);
    }
    if let Some(texture) = metallic_roughness_texture {
        mark_texture_linear(textures, texture);
        material = material.with_metallic_roughness_texture(texture);
    }
    if let Some(texture) = normal_texture {
        mark_texture_linear(textures, TextureId(texture.texture().index()));
        material =
            material.with_normal_texture(TextureId(texture.texture().index()), texture.scale());
    }
    if let Some(texture) = occlusion_texture {
        mark_texture_linear(textures, TextureId(texture.texture().index()));
        material = material
            .with_occlusion_texture(TextureId(texture.texture().index()), texture.strength());
    }
    if let Some(texture) = emissive_texture {
        material = material.with_emissive_texture(texture);
    }

    material
}

fn mark_texture_linear(textures: &mut [CpuTexture], texture: TextureId) {
    if let Some(texture) = textures.get_mut(texture.0) {
        texture.srgb = false;
    }
}

fn gltf_texture_to_cpu(
    texture: gltf::Texture<'_>,
    images: &[gltf::image::Data],
) -> io::Result<CpuTexture> {
    let image = images
        .get(texture.source().index())
        .ok_or_else(|| invalid_model_error("glTF texture references a missing image"))?;

    Ok(CpuTexture {
        pixels: image_to_rgba8(image)?,
        width: image.width.max(1),
        height: image.height.max(1),
        sampler: sampler_from_gltf(texture.sampler()),
        srgb: true,
    })
}

fn sampler_from_gltf(sampler: gltf::texture::Sampler<'_>) -> TextureSampler {
    TextureSampler {
        mag_filter: sampler
            .mag_filter()
            .map(|filter| match filter {
                gltf::texture::MagFilter::Nearest => TextureFilter::Nearest,
                gltf::texture::MagFilter::Linear => TextureFilter::Linear,
            })
            .unwrap_or(TextureFilter::Linear),
        min_filter: sampler
            .min_filter()
            .map(|filter| match filter {
                gltf::texture::MinFilter::Nearest
                | gltf::texture::MinFilter::NearestMipmapNearest
                | gltf::texture::MinFilter::NearestMipmapLinear => TextureFilter::Nearest,
                gltf::texture::MinFilter::Linear
                | gltf::texture::MinFilter::LinearMipmapNearest
                | gltf::texture::MinFilter::LinearMipmapLinear => TextureFilter::Linear,
            })
            .unwrap_or(TextureFilter::Linear),
        wrap_s: wrap_from_gltf(sampler.wrap_s()),
        wrap_t: wrap_from_gltf(sampler.wrap_t()),
    }
}

fn wrap_from_gltf(wrap: gltf::texture::WrappingMode) -> TextureWrap {
    match wrap {
        gltf::texture::WrappingMode::ClampToEdge => TextureWrap::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => TextureWrap::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => TextureWrap::Repeat,
    }
}

fn image_to_rgba8(image: &gltf::image::Data) -> io::Result<Vec<u8>> {
    use gltf::image::Format;

    let pixels = match image.format {
        Format::R8 => image.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect(),
        Format::R8G8 => image
            .pixels
            .chunks_exact(2)
            .flat_map(|pixel| [pixel[0], pixel[1], 0, 255])
            .collect(),
        Format::R8G8B8 => image
            .pixels
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
            .collect(),
        Format::R8G8B8A8 => image.pixels.clone(),
        Format::R16 => image
            .pixels
            .chunks_exact(2)
            .flat_map(|pixel| [pixel[1], pixel[1], pixel[1], 255])
            .collect(),
        Format::R16G16 => image
            .pixels
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[1], pixel[3], 0, 255])
            .collect(),
        Format::R16G16B16 => image
            .pixels
            .chunks_exact(6)
            .flat_map(|pixel| [pixel[1], pixel[3], pixel[5], 255])
            .collect(),
        Format::R16G16B16A16 => image
            .pixels
            .chunks_exact(8)
            .flat_map(|pixel| [pixel[1], pixel[3], pixel[5], pixel[7]])
            .collect(),
        Format::R32G32B32FLOAT => image
            .pixels
            .chunks_exact(12)
            .flat_map(|pixel| {
                [
                    f32_bytes_to_u8(&pixel[0..4]),
                    f32_bytes_to_u8(&pixel[4..8]),
                    f32_bytes_to_u8(&pixel[8..12]),
                    255,
                ]
            })
            .collect(),
        Format::R32G32B32A32FLOAT => image
            .pixels
            .chunks_exact(16)
            .flat_map(|pixel| {
                [
                    f32_bytes_to_u8(&pixel[0..4]),
                    f32_bytes_to_u8(&pixel[4..8]),
                    f32_bytes_to_u8(&pixel[8..12]),
                    f32_bytes_to_u8(&pixel[12..16]),
                ]
            })
            .collect(),
    };

    let expected_len = image.width as usize * image.height as usize * 4;
    if pixels.len() != expected_len {
        return invalid_model("glTF image data size does not match its dimensions");
    }

    Ok(pixels)
}

fn f32_bytes_to_u8(bytes: &[u8]) -> u8 {
    let value = f32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]));
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn sequential_indices(len: usize) -> io::Result<Vec<u32>> {
    let len = u32::try_from(len)
        .map_err(|_| invalid_model_error("glTF primitive has too many vertices"))?;

    Ok((0..len).collect())
}

fn validate_indices(indices: &[u32], vertex_count: usize) -> io::Result<()> {
    if indices.iter().any(|&index| index as usize >= vertex_count) {
        return invalid_model("glTF index is out of bounds");
    }

    Ok(())
}

fn accumulate_normals(vertices: &mut [ModelVertex], indices: &[u32]) {
    for triangle in indices.chunks_exact(3) {
        let a = triangle[0] as usize;
        let b = triangle[1] as usize;
        let c = triangle[2] as usize;
        let normal = triangle_normal(
            vertices[a].position,
            vertices[b].position,
            vertices[c].position,
        );

        for index in [a, b, c] {
            vertices[index].normal[0] += normal[0];
            vertices[index].normal[1] += normal[1];
            vertices[index].normal[2] += normal[2];
        }
    }

    for vertex in vertices {
        vertex.normal = normalize_or(vertex.normal, [0.0, 1.0, 0.0]);
    }
}

fn gltf_node_transform(transform: gltf::scene::Transform) -> [f32; 16] {
    let (translation, rotation, scale) = transform.decomposed();
    let [x, y, z, w] = rotation;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    [
        (1.0 - yy - zz) * scale[0],
        (xy + wz) * scale[0],
        (xz - wy) * scale[0],
        0.0,
        (xy - wz) * scale[1],
        (1.0 - xx - zz) * scale[1],
        (yz + wx) * scale[1],
        0.0,
        (xz + wy) * scale[2],
        (yz - wx) * scale[2],
        (1.0 - xx - yy) * scale[2],
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ]
}

fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];

    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|i| a[i * 4 + row] * b[col * 4 + i]).sum();
        }
    }

    out
}

fn transform_position(matrix: [f32; 16], position: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2] + matrix[12],
        matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2] + matrix[13],
        matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2] + matrix[14],
    ]
}

fn transform_normal(matrix: [f32; 16], normal: [f32; 3]) -> [f32; 3] {
    normalize_or(
        [
            matrix[0] * normal[0] + matrix[4] * normal[1] + matrix[8] * normal[2],
            matrix[1] * normal[0] + matrix[5] * normal[1] + matrix[9] * normal[2],
            matrix[2] * normal[0] + matrix[6] * normal[1] + matrix[10] * normal[2],
        ],
        [0.0, 1.0, 0.0],
    )
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

pub(super) struct MeshBuffers {
    pub vertex: GpuBuffer,
    pub index: GpuBuffer,
    pub index_count: u32,
}

impl MeshBuffers {
    pub fn from_mesh(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: &CpuMesh,
    ) -> Self {
        Self {
            vertex: GpuBuffer::device_local(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &mesh.vertices,
            ),
            index: GpuBuffer::device_local(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &mesh.indices,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.vertex.destroy(device);
            self.index.destroy(device);
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct GpuPrimitive {
    pub mesh: MeshId,
    pub material: MaterialId,
}

#[derive(Debug, Clone)]
pub(super) struct GpuModel {
    pub primitives: Vec<GpuPrimitive>,
}

pub(super) struct GpuTexture {
    view: vk::ImageView,
    image: vk::Image,
    memory: vk::DeviceMemory,
    sampler: vk::Sampler,
}

impl GpuTexture {
    fn from_texture(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture: &CpuTexture,
    ) -> Self {
        let size = texture.pixels.len() as vk::DeviceSize;
        let mut staging = GpuBuffer::new(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { staging.write_slice(device, &texture.pixels) };

        let (image, memory) = create_texture_image(
            instance,
            device,
            physical_device,
            texture.width,
            texture.height,
            texture.srgb,
        );
        transition_texture_image(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging.buffer,
            image,
            texture.width,
            texture.height,
        );
        transition_texture_image(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe { staging.destroy(device) };

        let view = create_texture_view(device, image, texture.srgb);
        let sampler = create_texture_sampler(device, texture.sampler);

        Self {
            view,
            image,
            memory,
            sampler,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.sampler != vk::Sampler::null() {
            unsafe { device.destroy_sampler(self.sampler, None) };
            self.sampler = vk::Sampler::null();
        }

        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }

        if self.image != vk::Image::null() {
            unsafe { device.destroy_image(self.image, None) };
            self.image = vk::Image::null();
        }

        if self.memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
        }
    }
}

struct GpuMaterial {
    material: Material,
    texture_set: vk::DescriptorSet,
}

pub(super) struct GpuAssets {
    meshes: Vec<MeshBuffers>,
    materials: Vec<GpuMaterial>,
    textures: Vec<GpuTexture>,
    models: Vec<GpuModel>,
    texture_layout: vk::DescriptorSetLayout,
    texture_pool: vk::DescriptorPool,
}

impl GpuAssets {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Self {
        let texture_layout = create_texture_set_layout(device);
        let texture_pool = create_texture_descriptor_pool(device);
        let mut assets = Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            models: Vec::new(),
            texture_layout,
            texture_pool,
        };
        assets.upload_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuTexture::white(),
        );
        assets.upload_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuTexture::flat_normal(),
        );
        assets.upload_material(device, Material::default());
        assets.upload_model(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &CpuModel::cube(),
        );
        assets
    }

    pub fn upload_mesh(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: &CpuMesh,
    ) -> MeshId {
        self.meshes.push(MeshBuffers::from_mesh(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            mesh,
        ));
        MeshId(self.meshes.len() - 1)
    }

    pub fn upload_material(&mut self, device: &ash::Device, material: Material) -> MaterialId {
        let texture_set = allocate_texture_set(
            device,
            self.texture_layout,
            self.texture_pool,
            &self.textures,
            material,
        );
        self.materials.push(GpuMaterial {
            material,
            texture_set,
        });
        MaterialId(self.materials.len() - 1)
    }

    pub fn upload_texture(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture: &CpuTexture,
    ) -> TextureId {
        self.textures.push(GpuTexture::from_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            texture,
        ));
        TextureId(self.textures.len() - 1)
    }

    pub fn upload_model(
        &mut self,
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        model: &CpuModel,
    ) -> ModelId {
        let textures = model
            .textures
            .iter()
            .map(|texture| {
                self.upload_texture(
                    instance,
                    device,
                    physical_device,
                    command_pool,
                    queue,
                    texture,
                )
            })
            .collect::<Vec<_>>();

        let primitives = model
            .primitives
            .iter()
            .map(|primitive| {
                let mut material = primitive.material;
                material.base_color_texture = material
                    .base_color_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.metallic_roughness_texture = material
                    .metallic_roughness_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.normal_texture = material
                    .normal_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.occlusion_texture = material
                    .occlusion_texture
                    .and_then(|texture| textures.get(texture.0).copied());
                material.emissive_texture = material
                    .emissive_texture
                    .and_then(|texture| textures.get(texture.0).copied());

                GpuPrimitive {
                    mesh: self.upload_mesh(
                        instance,
                        device,
                        physical_device,
                        command_pool,
                        queue,
                        &primitive.mesh,
                    ),
                    material: self.upload_material(device, material),
                }
            })
            .collect();

        self.models.push(GpuModel { primitives });
        ModelId(self.models.len() - 1)
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshBuffers> {
        self.meshes.get(id.0)
    }

    pub fn material(&self, id: MaterialId) -> Material {
        self.materials
            .get(id.0)
            .map(|material| material.material)
            .unwrap_or_default()
    }

    pub fn model(&self, id: ModelId) -> Option<&GpuModel> {
        self.models.get(id.0)
    }

    pub fn material_texture_set(&self, id: MaterialId) -> Option<vk::DescriptorSet> {
        self.materials
            .get(id.0)
            .map(|material| material.texture_set)
    }

    pub fn texture_set_layout(&self) -> vk::DescriptorSetLayout {
        self.texture_layout
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for mesh in &mut self.meshes {
            unsafe { mesh.destroy(device) };
        }

        for texture in &mut self.textures {
            unsafe { texture.destroy(device) };
        }

        if self.texture_pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.texture_pool, None) };
            self.texture_pool = vk::DescriptorPool::null();
        }

        if self.texture_layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.texture_layout, None) };
            self.texture_layout = vk::DescriptorSetLayout::null();
        }
    }
}

pub(super) struct GpuBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl GpuBuffer {
    pub fn host_uniform<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        value: &T,
    ) -> Self {
        let mut buffer = Self::new(
            instance,
            device,
            physical_device,
            mem::size_of::<T>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { buffer.write(device, value) };
        buffer
    }

    pub fn device_local<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Self {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        let mut staging = Self::new(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe { staging.write_slice(device, data) };

        let buffer = Self::new(
            instance,
            device,
            physical_device,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        copy_buffer(
            device,
            command_pool,
            queue,
            staging.buffer,
            buffer.buffer,
            size,
        );

        unsafe { staging.destroy(device) };
        buffer
    }

    fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&info, None)
                .expect("renderer: failed to create buffer")
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            properties,
        );

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("renderer: failed to allocate buffer memory")
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("renderer: failed to bind buffer memory")
        };

        Self { buffer, memory }
    }

    pub unsafe fn write<T: Copy>(&mut self, device: &ash::Device, data: &T) {
        let size = mem::size_of::<T>() as vk::DeviceSize;
        let mapped = unsafe {
            device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("renderer: failed to map buffer memory")
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                (data as *const T).cast::<u8>(),
                mapped.cast::<u8>(),
                size as usize,
            );
            device.unmap_memory(self.memory);
        }
    }

    unsafe fn write_slice<T: Copy>(&mut self, device: &ash::Device, data: &[T]) {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        let mapped = unsafe {
            device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("renderer: failed to map buffer memory")
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                size as usize,
            );
            device.unmap_memory(self.memory);
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe { device.destroy_buffer(self.buffer, None) };
            self.buffer = vk::Buffer::null();
        }

        if self.memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
        }
    }
}

pub(super) struct SceneBindings {
    pub layout: vk::DescriptorSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
    pool: vk::DescriptorPool,
    buffers: Vec<GpuBuffer>,
    range: vk::DeviceSize,
}

impl SceneBindings {
    pub fn new<T: Copy>(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        initial: &T,
    ) -> Self {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("renderer: failed to create scene descriptor layout")
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        };
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("renderer: failed to create scene descriptor pool")
        };

        let layouts = vec![layout; MAX_FRAMES_IN_FLIGHT];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc)
                .expect("renderer: failed to allocate scene descriptor sets")
        };

        let range = mem::size_of::<T>() as vk::DeviceSize;
        let buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| GpuBuffer::host_uniform(instance, device, physical_device, initial))
            .collect::<Vec<_>>();

        for (set, buffer) in sets.iter().zip(&buffers) {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .range(range);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        Self {
            layout,
            sets,
            pool,
            buffers,
            range,
        }
    }

    pub fn update<T: Copy>(&mut self, device: &ash::Device, frame_index: usize, value: &T) {
        debug_assert_eq!(self.range, mem::size_of::<T>() as vk::DeviceSize);
        unsafe { self.buffers[frame_index].write(device, value) };
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for buffer in &mut self.buffers {
            unsafe { buffer.destroy(device) };
        }

        if self.pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.pool, None) };
            self.pool = vk::DescriptorPool::null();
        }

        if self.layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
            self.layout = vk::DescriptorSetLayout::null();
        }
    }
}

pub(super) struct DepthTarget {
    pub view: vk::ImageView,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl DepthTarget {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("renderer: failed to create depth image")
        };

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe {
            device
                .allocate_memory(&alloc, None)
                .expect("renderer: failed to allocate depth memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("renderer: failed to bind depth memory")
        };

        transition_depth_image(device, command_pool, queue, image);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("renderer: failed to create depth view")
        };

        Self {
            view,
            image,
            memory,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.view != vk::ImageView::null() {
            unsafe { device.destroy_image_view(self.view, None) };
            self.view = vk::ImageView::null();
        }

        if self.image != vk::Image::null() {
            unsafe { device.destroy_image(self.image, None) };
            self.image = vk::Image::null();
        }

        if self.memory != vk::DeviceMemory::null() {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
        }
    }
}

fn create_texture_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = (0..5)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .collect::<Vec<_>>();

    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&info, None)
            .expect("renderer: failed to create texture descriptor layout")
    }
}

fn create_texture_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
    const MAX_MATERIALS: u32 = 1024;
    const TEXTURES_PER_MATERIAL: u32 = 5;

    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: MAX_MATERIALS * TEXTURES_PER_MATERIAL,
    };
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MAX_MATERIALS)
        .pool_sizes(std::slice::from_ref(&pool_size));

    unsafe {
        device
            .create_descriptor_pool(&info, None)
            .expect("renderer: failed to create texture descriptor pool")
    }
}

fn allocate_texture_set(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    textures: &[GpuTexture],
    material: Material,
) -> vk::DescriptorSet {
    let layouts = [layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe {
        device
            .allocate_descriptor_sets(&alloc)
            .expect("renderer: failed to allocate texture descriptor set")[0]
    };
    let texture_ids = [
        material.base_color_texture.unwrap_or(TextureId::DEFAULT),
        material
            .metallic_roughness_texture
            .unwrap_or(TextureId::DEFAULT),
        material.normal_texture.unwrap_or(TextureId::NORMAL),
        material.occlusion_texture.unwrap_or(TextureId::DEFAULT),
        material.emissive_texture.unwrap_or(TextureId::DEFAULT),
    ];
    let image_infos = texture_ids
        .into_iter()
        .enumerate()
        .map(|(index, texture)| {
            let fallback = if index == 2 {
                TextureId::NORMAL
            } else {
                TextureId::DEFAULT
            };
            let texture = textures
                .get(texture.0)
                .or_else(|| textures.get(fallback.0))
                .expect("renderer: missing default texture");

            vk::DescriptorImageInfo::default()
                .sampler(texture.sampler)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();
    let writes = image_infos
        .iter()
        .enumerate()
        .map(|(binding, image_info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(image_info))
        })
        .collect::<Vec<_>>();

    unsafe { device.update_descriptor_sets(&writes, &[]) };
    set
}

fn create_texture_image(
    instance: &Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    srgb: bool,
) -> (vk::Image, vk::DeviceMemory) {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(texture_format(srgb))
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("renderer: failed to create texture image")
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe {
        device
            .allocate_memory(&alloc, None)
            .expect("renderer: failed to allocate texture memory")
    };

    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("renderer: failed to bind texture memory")
    };

    (image, memory)
}

fn create_texture_view(device: &ash::Device, image: vk::Image, srgb: bool) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(texture_format(srgb))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device
            .create_image_view(&info, None)
            .expect("renderer: failed to create texture view")
    }
}

fn texture_format(srgb: bool) -> vk::Format {
    if srgb {
        vk::Format::R8G8B8A8_SRGB
    } else {
        vk::Format::R8G8B8A8_UNORM
    }
}

fn create_texture_sampler(device: &ash::Device, sampler: TextureSampler) -> vk::Sampler {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(texture_filter(sampler.mag_filter))
        .min_filter(texture_filter(sampler.min_filter))
        .address_mode_u(texture_wrap(sampler.wrap_s))
        .address_mode_v(texture_wrap(sampler.wrap_t))
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(0.0);

    unsafe {
        device
            .create_sampler(&info, None)
            .expect("renderer: failed to create texture sampler")
    }
}

fn texture_filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

fn texture_wrap(wrap: TextureWrap) -> vk::SamplerAddressMode {
    match wrap {
        TextureWrap::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        TextureWrap::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        TextureWrap::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

fn copy_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let region = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(command_buffer, src, dst, std::slice::from_ref(&region)) };
    });
}

fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            )
        };
    });
}

fn transition_texture_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        ),
        _ => panic!("unsupported texture layout transition"),
    };

    submit_once(device, command_pool, queue, |command_buffer| {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
    });
}

fn transition_depth_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
) {
    submit_once(device, command_pool, queue, |command_buffer| {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
    });
}

fn submit_once<F: FnOnce(vk::CommandBuffer)>(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    record: F,
) {
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc)
            .expect("renderer: failed to allocate transient command buffer")[0]
    };

    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin)
            .expect("renderer: failed to begin transient command buffer")
    };

    record(command_buffer);

    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("renderer: failed to end transient command buffer")
    };

    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

    unsafe {
        device
            .queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())
            .expect("renderer: failed to submit transient command buffer");
        device
            .queue_wait_idle(queue)
            .expect("renderer: failed to wait transient command buffer");
        device.free_command_buffers(command_pool, &command_buffers);
    }
}

fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    (0..memory.memory_type_count)
        .find(|&index| {
            let supported = (type_filter & (1_u32 << index)) != 0;
            let flags = memory.memory_types[index as usize].property_flags;
            supported && flags.contains(properties)
        })
        .expect("renderer: failed to find suitable memory type")
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
        "-blendu" | "-blendv" | "-boost" | "-texres" | "-clamp" | "-bm" | "-imfchan"
        | "-type" => 1,
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
            "Ns" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.roughness = shininess_to_roughness(parse_mtl_scalar(parts, 32.0));
                }
            }
            "d" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.base_color[3] = parse_mtl_scalar(parts, 1.0).clamp(0.0, 1.0);
                }
            }
            "Tr" => {
                if let Some(material) = current_material_mut(&current, materials) {
                    material.base_color[3] = 1.0 - parse_mtl_scalar(parts, 0.0).clamp(0.0, 1.0);
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
                        material.base_color_texture = Some(texture_id);
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

fn invalid_model<T>(message: &'static str) -> io::Result<T> {
    Err(invalid_model_error(message))
}

fn invalid_model_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn model_error(path: &Path, source: gltf::Error) -> io::Error {
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

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize_or(
        [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ],
        [0.0, 1.0, 0.0],
    )
}

fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return fallback;
    }

    [v[0] / len, v[1] / len, v[2] / len]
}

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
        assert_eq!(material.base_color_texture, Some(TextureId(0)));
        assert_close(material.base_color, [0.2, 0.3, 0.4, 1.0]);

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
        "emissiveFactor": [0.1, 0.0, 0.2]
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
        assert!((primitive.material.metallic - 0.3).abs() < 0.001);
        assert!((primitive.material.roughness - 0.7).abs() < 0.001);
        assert_eq!(primitive.material.base_color_texture, Some(TextureId(0)));
        assert_eq!(
            primitive.material.metallic_roughness_texture,
            Some(TextureId(1))
        );
        assert_eq!(primitive.material.normal_texture, Some(TextureId(2)));
        assert_eq!(primitive.material.occlusion_texture, Some(TextureId(3)));
        assert_eq!(primitive.material.emissive_texture, Some(TextureId(4)));
        assert!((primitive.material.normal_scale - 0.5).abs() < 0.001);
        assert!((primitive.material.occlusion_strength - 0.25).abs() < 0.001);

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
