use std::{
    fs, io,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{
    math::{add3, cross3, normalize_or, sub3},
    protocol::{MaterialAlphaMode, MaterialTextureSlot, SceneBounds, TextureDescriptor},
};

const REBUILD1_SCENE_EXTENSION: &str = "r1scene";
const REBUILD1_SCENE_HEADER: &str = "rebuild1-scene";
const GLB_EXTENSION: &str = "glb";
const GLTF_EXTENSION: &str = "gltf";
const MAT4_IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedScene {
    source: PathBuf,
    meshes: Vec<ImportedMesh>,
    materials: Vec<ImportedMaterial>,
    textures: Vec<ImportedTexture>,
    bounds: Option<SceneBounds>,
}

impl ImportedScene {
    /// Creates imported scene metadata after the file-format layer has validated it.
    pub fn new(
        source: PathBuf,
        mesh_count: usize,
        material_count: usize,
        texture_count: usize,
    ) -> Self {
        let meshes = vec![ImportedMesh::Plane; mesh_count];
        let materials = vec![ImportedMaterial::opaque(); material_count];
        let textures = vec![ImportedTexture::solid([255, 255, 255, 255]); texture_count];

        Self::from_parts(source, meshes, materials, textures)
    }

    /// Creates imported scene metadata from explicit intermediate scene parts.
    pub fn from_parts(
        source: PathBuf,
        meshes: Vec<ImportedMesh>,
        materials: Vec<ImportedMaterial>,
        textures: Vec<ImportedTexture>,
    ) -> Self {
        let bounds = scene_bounds_from_meshes(&meshes);

        Self {
            source,
            meshes,
            materials,
            textures,
            bounds,
        }
    }

    /// Returns the source path used for asset load diagnostics.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Returns how many mesh handles the GPU asset store must allocate.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Returns how many material handles the GPU asset store must allocate.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Returns how many texture handles the GPU asset store must allocate.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Returns imported mesh records in file order.
    pub fn meshes(&self) -> &[ImportedMesh] {
        &self.meshes
    }

    /// Returns imported material records in file order.
    pub fn materials(&self) -> &[ImportedMaterial] {
        &self.materials
    }

    /// Returns imported texture records in file order.
    pub fn textures(&self) -> &[ImportedTexture] {
        &self.textures
    }

    /// Returns imported scene bounds when at least one mesh has finite positions.
    pub fn bounds(&self) -> Option<SceneBounds> {
        self.bounds
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImportedMesh {
    Plane,
    Indexed(ImportedMeshData),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedMeshData {
    vertices: Vec<ImportedVertex>,
    indices: Vec<u32>,
}

impl ImportedMeshData {
    /// Creates one indexed triangle mesh after the importer has validated its bounds.
    pub fn new(vertices: Vec<ImportedVertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Returns vertices in importer order.
    pub fn vertices(&self) -> &[ImportedVertex] {
        &self.vertices
    }

    /// Returns triangle-list indices in importer order.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl ImportedVertex {
    /// Creates one imported vertex in renderer-independent CPU memory.
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position,
            normal: normalize_or(normal, [0.0, 1.0, 0.0]),
            uv,
            color: color.map(clamp_unit),
        }
    }

    /// Returns the imported vertex position.
    pub fn position(self) -> [f32; 3] {
        self.position
    }

    /// Returns the imported vertex normal after importer-side normalization.
    pub fn normal(self) -> [f32; 3] {
        self.normal
    }

    /// Returns the imported texture coordinate.
    pub fn uv(self) -> [f32; 2] {
        self.uv
    }

    /// Returns the imported debug/base color.
    pub fn color(self) -> [f32; 4] {
        self.color
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedMaterial {
    alpha_mode: MaterialAlphaMode,
    alpha_cutoff_milli: u16,
    texture_slots: Vec<ImportedMaterialTextureSlot>,
}

impl ImportedMaterial {
    /// Creates an explicit opaque material with no texture slots.
    pub fn opaque() -> Self {
        Self {
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff_milli: 500,
            texture_slots: Vec::new(),
        }
    }

    /// Creates a material descriptor before texture indices are resolved to handles.
    pub fn new(
        alpha_mode: MaterialAlphaMode,
        alpha_cutoff_milli: u16,
        texture_slots: Vec<ImportedMaterialTextureSlot>,
    ) -> Self {
        Self {
            alpha_mode,
            alpha_cutoff_milli,
            texture_slots,
        }
    }

    /// Returns the alpha mode imported for this material.
    pub fn alpha_mode(&self) -> MaterialAlphaMode {
        self.alpha_mode
    }

    /// Returns the alpha cutoff as a deterministic milli value.
    pub fn alpha_cutoff_milli(&self) -> u16 {
        self.alpha_cutoff_milli
    }

    /// Returns texture slot references by imported texture index.
    pub fn texture_slots(&self) -> &[ImportedMaterialTextureSlot] {
        &self.texture_slots
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportedMaterialTextureSlot {
    slot: MaterialTextureSlot,
    texture_index: usize,
}

impl ImportedMaterialTextureSlot {
    /// Creates a named material slot reference to an imported texture index.
    pub fn new(slot: MaterialTextureSlot, texture_index: usize) -> Self {
        Self {
            slot,
            texture_index,
        }
    }

    /// Returns the material slot this texture fills.
    pub fn slot(self) -> MaterialTextureSlot {
        self.slot
    }

    /// Returns the imported texture index referenced by this slot.
    pub fn texture_index(self) -> usize {
        self.texture_index
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedTexture {
    descriptor: TextureDescriptor,
}

impl ImportedTexture {
    /// Creates one explicit solid color imported texture.
    pub fn solid(rgba: [u8; 4]) -> Self {
        Self {
            descriptor: TextureDescriptor::solid_rgba8_srgb(rgba),
        }
    }

    /// Creates one explicit RGBA8 sRGB imported texture payload.
    pub fn rgba8_srgb(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        TextureDescriptor::rgba8_srgb(width, height, pixels).map(|descriptor| Self { descriptor })
    }

    /// Returns the texture payload that the GPU asset store should upload.
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }
}

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("asset file does not exist: {0}")]
    NotFound(PathBuf),
    #[error("asset format is unsupported for path: {0}")]
    UnsupportedFormat(PathBuf),
    #[error("failed to read asset file {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("failed to import glTF asset {path}: {message}")]
    Gltf { path: PathBuf, message: String },
    #[error("asset file {path} is missing the rebuild1-scene header")]
    MissingHeader { path: PathBuf },
    #[error("asset file {path} contains an unknown directive on line {line}: {directive}")]
    UnknownDirective {
        path: PathBuf,
        line: usize,
        directive: String,
    },
    #[error("asset file {path} has invalid directive on line {line}: {message}")]
    InvalidDirective {
        path: PathBuf,
        line: usize,
        message: String,
    },
}

#[derive(Debug, Error)]
pub enum ImportTaskError {
    #[error("{0}")]
    Import(#[from] ImportError),
    #[error("asset import worker task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Imports one explicit asset file on a blocking worker instead of the renderer thread.
pub async fn import_asset_on_worker(path: PathBuf) -> Result<ImportedScene, ImportTaskError> {
    Ok(tokio::task::spawn_blocking(move || import_asset(&path)).await??)
}

/// Imports one explicit asset file into renderer-independent CPU metadata.
pub fn import_asset(path: &Path) -> Result<ImportedScene, ImportError> {
    match asset_format(path)? {
        AssetFormat::Rebuild1Scene => {
            let text = fs::read_to_string(path).map_err(|source| read_error(path, source))?;
            parse_rebuild1_scene(path, &text)
        }
        AssetFormat::Gltf => import_gltf_scene(path),
    }
}

/// Returns the explicit importer selected by the asset extension.
fn asset_format(path: &Path) -> Result<AssetFormat, ImportError> {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return Err(ImportError::UnsupportedFormat(path.to_path_buf()));
    };

    if extension.eq_ignore_ascii_case(REBUILD1_SCENE_EXTENSION) {
        Ok(AssetFormat::Rebuild1Scene)
    } else if extension.eq_ignore_ascii_case(GLB_EXTENSION)
        || extension.eq_ignore_ascii_case(GLTF_EXTENSION)
    {
        Ok(AssetFormat::Gltf)
    } else {
        Err(ImportError::UnsupportedFormat(path.to_path_buf()))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssetFormat {
    Rebuild1Scene,
    Gltf,
}

/// Converts filesystem read failures into import-level diagnostics.
fn read_error(path: &Path, source: io::Error) -> ImportError {
    if source.kind() == io::ErrorKind::NotFound {
        ImportError::NotFound(path.to_path_buf())
    } else {
        ImportError::Read {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Parses the tiny Stage 5 scene manifest without creating fallback geometry.
fn parse_rebuild1_scene(path: &Path, text: &str) -> Result<ImportedScene, ImportError> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'));

    let Some((_, header)) = lines.next() else {
        return Err(ImportError::MissingHeader {
            path: path.to_path_buf(),
        });
    };

    if header != REBUILD1_SCENE_HEADER {
        return Err(ImportError::MissingHeader {
            path: path.to_path_buf(),
        });
    }

    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut textures = Vec::new();
    for (line, directive) in lines {
        let parts = directive.split_whitespace().collect::<Vec<_>>();
        match parts.as_slice() {
            ["mesh"] | ["mesh", "plane"] => meshes.push(ImportedMesh::Plane),
            ["material"] => materials.push(ImportedMaterial::opaque()),
            ["material", rest @ ..] => {
                materials.push(parse_material(path, line, rest, textures.len())?)
            }
            ["texture"] => textures.push(ImportedTexture::solid([255, 255, 255, 255])),
            ["texture", "solid", r, g, b, a] => {
                textures.push(ImportedTexture::solid([
                    parse_u8(path, line, r)?,
                    parse_u8(path, line, g)?,
                    parse_u8(path, line, b)?,
                    parse_u8(path, line, a)?,
                ]));
            }
            ["texture", "checker", r0, g0, b0, a0, r1, g1, b1, a1] => {
                textures.push(checker_texture([
                    parse_u8(path, line, r0)?,
                    parse_u8(path, line, g0)?,
                    parse_u8(path, line, b0)?,
                    parse_u8(path, line, a0)?,
                    parse_u8(path, line, r1)?,
                    parse_u8(path, line, g1)?,
                    parse_u8(path, line, b1)?,
                    parse_u8(path, line, a1)?,
                ]));
            }
            ["node"] => {}
            _ => {
                return Err(ImportError::UnknownDirective {
                    path: path.to_path_buf(),
                    line,
                    directive: directive.to_owned(),
                });
            }
        }
    }

    Ok(ImportedScene::from_parts(
        path.to_path_buf(),
        meshes,
        materials,
        textures,
    ))
}

/// Builds a tiny explicit checker texture used by verification scenes.
fn checker_texture(rgba: [u8; 8]) -> ImportedTexture {
    let a = [rgba[0], rgba[1], rgba[2], rgba[3]];
    let b = [rgba[4], rgba[5], rgba[6], rgba[7]];
    let pixels = [a, b, b, a].into_iter().flatten().collect::<Vec<_>>();

    ImportedTexture::rgba8_srgb(2, 2, pixels)
        .expect("hard-coded checker texture byte count is valid")
}

/// Imports glTF/GLB triangle primitives into the renderer-independent intermediate scene.
fn import_gltf_scene(path: &Path) -> Result<ImportedScene, ImportError> {
    let (document, buffers, images) =
        gltf::import(path).map_err(|source| gltf_error(path, source))?;
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let textures = import_gltf_textures(path, &images)?;

    if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            import_gltf_node(
                path,
                node,
                MAT4_IDENTITY,
                &buffers,
                &textures,
                &mut meshes,
                &mut materials,
            )?;
        }
    } else {
        for scene in document.scenes() {
            for node in scene.nodes() {
                import_gltf_node(
                    path,
                    node,
                    MAT4_IDENTITY,
                    &buffers,
                    &textures,
                    &mut meshes,
                    &mut materials,
                )?;
            }
        }
    }

    if meshes.is_empty() {
        return Err(gltf_message(
            path,
            "glTF asset contains no triangle primitives",
        ));
    }

    tracing::trace!(
        source = %path.display(),
        meshes = meshes.len(),
        materials = materials.len(),
        bounds = ?scene_bounds_from_meshes(&meshes),
        "imported glTF asset"
    );

    Ok(ImportedScene::from_parts(
        path.to_path_buf(),
        meshes,
        materials,
        textures,
    ))
}

/// Imports one glTF node and recursively applies parent transforms to child meshes.
fn import_gltf_node(
    path: &Path,
    node: gltf::Node<'_>,
    parent_transform: [f32; 16],
    buffers: &[gltf::buffer::Data],
    textures: &[ImportedTexture],
    meshes: &mut Vec<ImportedMesh>,
    materials: &mut Vec<ImportedMaterial>,
) -> Result<(), ImportError> {
    let transform = mat4_mul(parent_transform, gltf_matrix(node.transform().matrix()));

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            import_gltf_primitive(
                path, primitive, transform, buffers, textures, meshes, materials,
            )?;
        }
    }

    for child in node.children() {
        import_gltf_node(path, child, transform, buffers, textures, meshes, materials)?;
    }

    Ok(())
}

/// Imports one glTF triangle primitive as one mesh/material pair.
fn import_gltf_primitive(
    path: &Path,
    primitive: gltf::Primitive<'_>,
    transform: [f32; 16],
    buffers: &[gltf::buffer::Data],
    textures: &[ImportedTexture],
    meshes: &mut Vec<ImportedMesh>,
    materials: &mut Vec<ImportedMaterial>,
) -> Result<(), ImportError> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        tracing::trace!(
            mode = ?primitive.mode(),
            "skipped non-triangle glTF primitive"
        );
        return Ok(());
    }

    let material = import_gltf_material(path, primitive.material(), textures.len())?;
    let color = gltf_base_color(primitive.material());
    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let positions = reader
        .read_positions()
        .ok_or_else(|| gltf_message(path, "glTF primitive is missing POSITION"))?
        .map(|position| transform_position(transform, position))
        .collect::<Vec<_>>();
    let uvs = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_default();
    let indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<_>>(),
        None => sequential_indices(path, positions.len())?,
    };

    validate_indices(path, &indices, positions.len())?;
    if indices.len() % 3 != 0 {
        return Err(gltf_message(
            path,
            "glTF primitive index count is not divisible by three",
        ));
    }
    let normals = reader
        .read_normals()
        .map(|normals| {
            normals
                .map(|normal| normalize_or(transform_direction(transform, normal), [0.0, 1.0, 0.0]))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| compute_vertex_normals(&positions, &indices));

    let vertices = positions
        .into_iter()
        .enumerate()
        .map(|(index, position)| {
            ImportedVertex::new(
                position,
                normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
                uvs.get(index).copied().unwrap_or([0.0, 0.0]),
                color,
            )
        })
        .collect::<Vec<_>>();
    meshes.push(ImportedMesh::Indexed(ImportedMeshData::new(
        vertices, indices,
    )));
    materials.push(material);

    Ok(())
}

/// Converts a glTF material into the subset currently understood by the renderer.
fn import_gltf_material(
    path: &Path,
    material: gltf::Material<'_>,
    texture_count: usize,
) -> Result<ImportedMaterial, ImportError> {
    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => MaterialAlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => MaterialAlphaMode::Cutout,
        gltf::material::AlphaMode::Blend => MaterialAlphaMode::Transparent,
    };
    let alpha_cutoff_milli =
        (material.alpha_cutoff().unwrap_or(0.5).clamp(0.0, 1.0) * 1000.0).round() as u16;
    let mut texture_slots = Vec::new();
    if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
        let index = info.texture().source().index();
        if index >= texture_count {
            return Err(gltf_message(
                path,
                "glTF material references a texture image that was not imported",
            ));
        }
        texture_slots.push(ImportedMaterialTextureSlot::new(
            MaterialTextureSlot::BaseColor,
            index,
        ));
    }

    Ok(ImportedMaterial::new(
        alpha_mode,
        alpha_cutoff_milli,
        texture_slots,
    ))
}

/// Converts glTF image payloads into renderer-owned RGBA8 sRGB texture descriptors.
fn import_gltf_textures(
    path: &Path,
    images: &[gltf::image::Data],
) -> Result<Vec<ImportedTexture>, ImportError> {
    images
        .iter()
        .map(|image| {
            let pixels = image_to_rgba8(path, image)?;
            ImportedTexture::rgba8_srgb(image.width, image.height, pixels).ok_or_else(|| {
                gltf_message(path, "glTF image payload has an invalid RGBA8 byte count")
            })
        })
        .collect()
}

/// Expands common glTF image formats into RGBA8 while rejecting unsupported formats explicitly.
fn image_to_rgba8(path: &Path, image: &gltf::image::Data) -> Result<Vec<u8>, ImportError> {
    match image.format {
        gltf::image::Format::R8G8B8A8 => Ok(image.pixels.clone()),
        gltf::image::Format::R8G8B8 => Ok(image
            .pixels
            .chunks_exact(3)
            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
            .collect()),
        gltf::image::Format::R8 => Ok(image.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect()),
        gltf::image::Format::R8G8 => Ok(image
            .pixels
            .chunks_exact(2)
            .flat_map(|rg| [rg[0], rg[0], rg[0], rg[1]])
            .collect()),
        other => Err(ImportError::Gltf {
            path: path.to_path_buf(),
            message: format!("unsupported glTF image format: {other:?}"),
        }),
    }
}

/// Returns the glTF base color factor as the temporary mesh vertex color.
fn gltf_base_color(material: gltf::Material<'_>) -> [f32; 4] {
    material.pbr_metallic_roughness().base_color_factor()
}

/// Returns aggregate world-space scene bounds for imported meshes.
fn scene_bounds_from_meshes(meshes: &[ImportedMesh]) -> Option<SceneBounds> {
    let (min, max) = mesh_min_max(meshes)?;
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let radius = mesh_radius(meshes, center)?;

    SceneBounds::new(center, radius)
}

/// Returns aggregate min/max bounds for imported mesh positions.
fn mesh_min_max(meshes: &[ImportedMesh]) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut found = false;

    for mesh in meshes {
        for position in mesh_positions(mesh) {
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
            found = true;
        }
    }

    found.then_some((min, max))
}

/// Returns a radius that encloses all imported mesh positions around `center`.
fn mesh_radius(meshes: &[ImportedMesh], center: [f32; 3]) -> Option<f32> {
    let mut radius2 = 0.0_f32;
    let mut found = false;

    for mesh in meshes {
        for position in mesh_positions(mesh) {
            let dx = position[0] - center[0];
            let dy = position[1] - center[1];
            let dz = position[2] - center[2];
            radius2 = radius2.max(dx * dx + dy * dy + dz * dz);
            found = true;
        }
    }

    found.then_some(radius2.sqrt().max(0.001))
}

/// Returns an owned list of mesh positions for scene bound calculation.
fn mesh_positions(mesh: &ImportedMesh) -> Vec<[f32; 3]> {
    match mesh {
        ImportedMesh::Plane => vec![
            [-0.75, -0.55, 0.0],
            [0.75, -0.55, 0.0],
            [0.75, 0.55, 0.0],
            [-0.75, 0.55, 0.0],
        ],
        ImportedMesh::Indexed(data) => data
            .vertices()
            .iter()
            .map(|vertex| vertex.position())
            .collect(),
    }
}

/// Creates sequential indices for an unindexed glTF triangle primitive.
fn sequential_indices(path: &Path, len: usize) -> Result<Vec<u32>, ImportError> {
    let count = u32::try_from(len)
        .map_err(|_| gltf_message(path, "glTF primitive has too many vertices"))?;

    Ok((0..count).collect())
}

/// Rejects index buffers that point outside the imported vertex array.
fn validate_indices(path: &Path, indices: &[u32], vertex_count: usize) -> Result<(), ImportError> {
    if indices.iter().any(|&index| index as usize >= vertex_count) {
        return Err(gltf_message(path, "glTF primitive index is out of bounds"));
    }

    Ok(())
}

/// Converts a glTF matrix into the flat column-major matrix used by local helpers.
fn gltf_matrix(matrix: [[f32; 4]; 4]) -> [f32; 16] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
        matrix[3][0],
        matrix[3][1],
        matrix[3][2],
        matrix[3][3],
    ]
}

/// Multiplies two flat column-major matrices.
fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = a[row] * b[column * 4]
                + a[4 + row] * b[column * 4 + 1]
                + a[8 + row] * b[column * 4 + 2]
                + a[12 + row] * b[column * 4 + 3];
        }
    }
    out
}

/// Applies a flat column-major transform to one position.
fn transform_position(matrix: [f32; 16], position: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2] + matrix[12],
        matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2] + matrix[13],
        matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2] + matrix[14],
    ]
}

/// Applies a flat column-major transform to a direction without translation.
fn transform_direction(matrix: [f32; 16], direction: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * direction[0] + matrix[4] * direction[1] + matrix[8] * direction[2],
        matrix[1] * direction[0] + matrix[5] * direction[1] + matrix[9] * direction[2],
        matrix[2] * direction[0] + matrix[6] * direction[1] + matrix[10] * direction[2],
    ]
}

/// Computes smooth vertex normals for glTF primitives that omit normal attributes.
fn compute_vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0, 0.0, 0.0]; positions.len()];

    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let normal = cross3(
            sub3(positions[b], positions[a]),
            sub3(positions[c], positions[a]),
        );
        for index in [a, b, c] {
            normals[index] = add3(normals[index], normal);
        }
    }

    normals
        .into_iter()
        .map(|normal| normalize_or(normal, [0.0, 1.0, 0.0]))
        .collect()
}

/// Clamps imported color channels into the visible range.
fn clamp_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Wraps a glTF library error in an import diagnostic with the source path.
fn gltf_error(path: &Path, source: gltf::Error) -> ImportError {
    ImportError::Gltf {
        path: path.to_path_buf(),
        message: source.to_string(),
    }
}

/// Builds one glTF import diagnostic from a static validation message.
fn gltf_message(path: &Path, message: &'static str) -> ImportError {
    ImportError::Gltf {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

/// Parses one material directive with named texture slots and alpha policy.
fn parse_material(
    path: &Path,
    line: usize,
    parts: &[&str],
    texture_count: usize,
) -> Result<ImportedMaterial, ImportError> {
    let Some((mode, rest)) = parts.split_first() else {
        return Ok(ImportedMaterial::opaque());
    };
    let alpha_mode = MaterialAlphaMode::from_name(mode)
        .ok_or_else(|| invalid_directive(path, line, format!("unknown alpha mode: {mode}")))?;
    let mut alpha_cutoff_milli = 500;
    let mut texture_slots = Vec::new();

    for token in rest {
        let Some((name, value)) = token.split_once('=') else {
            return Err(invalid_directive(
                path,
                line,
                format!("expected key=value material option: {token}"),
            ));
        };

        if name == "alpha_cutoff" {
            alpha_cutoff_milli = parse_alpha_cutoff(path, line, value)?;
            continue;
        }

        let slot = MaterialTextureSlot::from_name(name).ok_or_else(|| {
            invalid_directive(path, line, format!("unknown material slot: {name}"))
        })?;
        let texture_index = parse_usize(path, line, value)?;
        if texture_index >= texture_count {
            return Err(invalid_directive(
                path,
                line,
                format!("texture index {texture_index} is out of range"),
            ));
        }
        texture_slots.push(ImportedMaterialTextureSlot::new(slot, texture_index));
    }

    Ok(ImportedMaterial::new(
        alpha_mode,
        alpha_cutoff_milli,
        texture_slots,
    ))
}

/// Parses one color byte used by a solid texture directive.
fn parse_u8(path: &Path, line: usize, value: &str) -> Result<u8, ImportError> {
    value
        .parse::<u8>()
        .map_err(|_| invalid_directive(path, line, format!("invalid u8 value: {value}")))
}

/// Parses one imported texture index.
fn parse_usize(path: &Path, line: usize, value: &str) -> Result<usize, ImportError> {
    value
        .parse::<usize>()
        .map_err(|_| invalid_directive(path, line, format!("invalid index value: {value}")))
}

/// Parses alpha cutoff and stores it as a milli value.
fn parse_alpha_cutoff(path: &Path, line: usize, value: &str) -> Result<u16, ImportError> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| invalid_directive(path, line, format!("invalid alpha cutoff: {value}")))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(invalid_directive(
            path,
            line,
            format!("alpha cutoff must be between 0 and 1: {value}"),
        ));
    }

    Ok((parsed * 1000.0).round() as u16)
}

/// Builds one invalid directive error with file and line context.
fn invalid_directive(path: &Path, line: usize, message: String) -> ImportError {
    ImportError::InvalidDirective {
        path: path.to_path_buf(),
        line,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that unsupported paths do not trigger hidden placeholder creation.
    #[test]
    fn import_rejects_unsupported_extension() {
        let path = PathBuf::from("missing.obj");

        let result = import_asset(&path);

        assert!(matches!(result, Err(ImportError::UnsupportedFormat(_))));
    }

    // Verifies that Stage 6 material slots and alpha mode survive import.
    #[test]
    fn import_reads_textured_cutout_material() {
        let path = PathBuf::from("scene.r1scene");
        let text = "rebuild1-scene\ntexture solid 255 0 0 255\nmaterial cutout base_color=0 alpha_cutoff=0.4\nmesh plane\n";

        let scene = parse_rebuild1_scene(&path, text).expect("manifest should parse");
        let material = &scene.materials()[0];

        assert_eq!(scene.texture_count(), 1);
        assert_eq!(material.alpha_mode(), MaterialAlphaMode::Cutout);
        assert_eq!(material.alpha_cutoff_milli(), 400);
        assert_eq!(
            material.texture_slots()[0].slot(),
            MaterialTextureSlot::BaseColor
        );
        assert_eq!(material.texture_slots()[0].texture_index(), 0);
    }

    // Verifies that Stage 8 verification scenes can request an explicit alpha checker texture.
    #[test]
    fn import_reads_checker_texture() {
        let path = PathBuf::from("scene.r1scene");
        let text = "rebuild1-scene\ntexture checker 255 255 255 255 0 0 0 0\nmaterial cutout base_color=0 alpha_cutoff=0.5\nmesh plane\n";

        let scene = parse_rebuild1_scene(&path, text).expect("manifest should parse");
        let texture = scene.textures()[0].descriptor();

        assert_eq!(texture.width(), 2);
        assert_eq!(texture.height(), 2);
        assert_eq!(texture.pixels().len(), 16);
    }
}
