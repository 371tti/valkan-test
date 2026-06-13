use std::{
    fs, io,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{
    math::{add3, cross3, dot3, normalize_or, sub3},
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
    tangent: [f32; 4],
    color: [f32; 4],
}

impl ImportedVertex {
    /// Creates one imported vertex in renderer-independent CPU memory.
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
        color: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal: normalize_or(normal, [0.0, 1.0, 0.0]),
            uv,
            tangent: normalize_tangent(tangent, normal),
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

    /// Returns the imported tangent xyz and bitangent handedness sign in w.
    pub fn tangent(self) -> [f32; 4] {
        self.tangent
    }

    /// Returns the imported debug/base color.
    pub fn color(self) -> [f32; 4] {
        self.color
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedMaterial {
    alpha_mode: MaterialAlphaMode,
    alpha_cutoff_milli: u16,
    base_color_factor: [f32; 4],
    metallic_factor_milli: u16,
    roughness_factor_milli: u16,
    emissive_factor: [f32; 3],
    occlusion_strength_milli: u16,
    normal_scale_milli: u16,
    double_sided: bool,
    texture_slots: Vec<ImportedMaterialTextureSlot>,
}

impl ImportedMaterial {
    /// Creates an explicit opaque material with no texture slots.
    pub fn opaque() -> Self {
        Self {
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff_milli: 500,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor_milli: 0,
            roughness_factor_milli: 1000,
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength_milli: 1000,
            normal_scale_milli: 1000,
            double_sided: false,
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
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor_milli: 0,
            roughness_factor_milli: 1000,
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength_milli: 1000,
            normal_scale_milli: 1000,
            double_sided: false,
            texture_slots,
        }
    }

    /// Creates a full imported material payload from glTF PBR properties.
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
        texture_slots: Vec<ImportedMaterialTextureSlot>,
    ) -> Self {
        Self {
            alpha_mode,
            alpha_cutoff_milli,
            base_color_factor: base_color_factor.map(clamp_unit),
            metallic_factor_milli: metallic_factor_milli.min(1000),
            roughness_factor_milli: roughness_factor_milli.min(1000),
            emissive_factor: emissive_factor.map(|value| {
                if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                }
            }),
            occlusion_strength_milli: occlusion_strength_milli.min(1000),
            normal_scale_milli: normal_scale_milli.min(4000),
            double_sided,
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

    /// Returns the imported base-color factor before texture handles are resolved.
    pub fn base_color_factor(&self) -> [f32; 4] {
        self.base_color_factor
    }

    /// Returns the imported metallic factor in deterministic milli units.
    pub fn metallic_factor_milli(&self) -> u16 {
        self.metallic_factor_milli
    }

    /// Returns the imported roughness factor in deterministic milli units.
    pub fn roughness_factor_milli(&self) -> u16 {
        self.roughness_factor_milli
    }

    /// Returns the imported emissive factor before texture handles are resolved.
    pub fn emissive_factor(&self) -> [f32; 3] {
        self.emissive_factor
    }

    /// Returns the imported occlusion strength in deterministic milli units.
    pub fn occlusion_strength_milli(&self) -> u16 {
        self.occlusion_strength_milli
    }

    /// Returns the imported normal-map scale in deterministic milli units.
    pub fn normal_scale_milli(&self) -> u16 {
        self.normal_scale_milli
    }

    /// Returns whether this imported material should disable back-face culling.
    pub fn double_sided(&self) -> bool {
        self.double_sided
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
    /// Wraps a validated protocol texture descriptor as imported CPU data.
    pub fn new(descriptor: TextureDescriptor) -> Self {
        Self { descriptor }
    }

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
    let texture_import = import_gltf_textures(path, &document, &images)?;
    let transforms = gltf_scene_transforms(&document, &buffers)?;

    if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            import_gltf_node(
                path,
                node,
                MAT4_IDENTITY,
                &transforms,
                &buffers,
                &texture_import,
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
                    &transforms,
                    &buffers,
                    &texture_import,
                    &mut meshes,
                    &mut materials,
                )?;
            }
        }

        if meshes.is_empty() {
            for mesh in document.meshes() {
                import_gltf_mesh(
                    path,
                    mesh,
                    MAT4_IDENTITY,
                    None,
                    &transforms,
                    &buffers,
                    &texture_import,
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
        texture_import.into_textures(),
    ))
}

#[derive(Clone, Copy)]
struct NodePose {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl NodePose {
    /// Captures one glTF node transform as TRS so animation channels can override components.
    fn from_transform(transform: gltf::scene::Transform) -> Self {
        let (translation, rotation, scale) = transform.decomposed();
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Rebuilds the node local transform after optional animation sampling.
    fn matrix(self) -> [f32; 16] {
        trs_matrix(self.translation, self.rotation, self.scale)
    }
}

struct GltfSceneTransforms {
    local: Vec<[f32; 16]>,
    global: Vec<[f32; 16]>,
}

/// Builds local/global glTF node transforms after sampling the first animation pose.
fn gltf_scene_transforms(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<GltfSceneTransforms, ImportError> {
    let mut poses = document
        .nodes()
        .map(|node| NodePose::from_transform(node.transform()))
        .collect::<Vec<_>>();
    apply_first_animation_pose(document, buffers, &mut poses);

    let local = poses.iter().map(|pose| pose.matrix()).collect::<Vec<_>>();
    let mut global = vec![MAT4_IDENTITY; local.len()];
    let mut has_parent = vec![false; local.len()];

    for node in document.nodes() {
        for child in node.children() {
            if let Some(slot) = has_parent.get_mut(child.index()) {
                *slot = true;
            }
        }
    }

    for node in document.nodes() {
        if !has_parent.get(node.index()).copied().unwrap_or(false) {
            collect_gltf_node_transform(node, MAT4_IDENTITY, &local, &mut global);
        }
    }

    Ok(GltfSceneTransforms { local, global })
}

/// Applies the first animation sample so authored bind-pose previews match the old loader.
fn apply_first_animation_pose(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    poses: &mut [NodePose],
) {
    let Some(animation) = document.animations().next() else {
        return;
    };

    for channel in animation.channels() {
        let node = channel.target().node().index();
        let Some(pose) = poses.get_mut(node) else {
            continue;
        };
        let reader =
            channel.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
        let sample_index = reader
            .read_inputs()
            .map(first_animation_sample)
            .unwrap_or(0);

        match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(outputs)) => {
                if let Some(value) = outputs.skip(sample_index).next() {
                    pose.translation = value;
                }
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(outputs)) => {
                if let Some(value) = outputs.into_f32().skip(sample_index).next() {
                    pose.rotation = normalize_quat(value);
                }
            }
            Some(gltf::animation::util::ReadOutputs::Scales(outputs)) => {
                if let Some(value) = outputs.skip(sample_index).next() {
                    pose.scale = value;
                }
            }
            Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(_)) | None => {}
        }
    }
}

/// Returns the first non-negative animation sample index used by the CPU preview importer.
fn first_animation_sample(times: impl Iterator<Item = f32>) -> usize {
    times
        .enumerate()
        .find_map(|(index, time)| (time >= 0.0).then_some(index))
        .unwrap_or(0)
}

/// Recursively computes global node transforms for skin joint matrix construction.
fn collect_gltf_node_transform(
    node: gltf::Node<'_>,
    parent: [f32; 16],
    local: &[[f32; 16]],
    global: &mut [[f32; 16]],
) {
    let transform = mat4_mul(
        parent,
        local.get(node.index()).copied().unwrap_or(MAT4_IDENTITY),
    );
    if let Some(slot) = global.get_mut(node.index()) {
        *slot = transform;
    }

    for child in node.children() {
        collect_gltf_node_transform(child, transform, local, global);
    }
}

/// Imports one glTF node and recursively applies parent transforms to child meshes.
fn import_gltf_node(
    path: &Path,
    node: gltf::Node<'_>,
    parent_transform: [f32; 16],
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &GltfImportedTextures,
    meshes: &mut Vec<ImportedMesh>,
    materials: &mut Vec<ImportedMaterial>,
) -> Result<(), ImportError> {
    let transform = mat4_mul(
        parent_transform,
        transforms
            .local
            .get(node.index())
            .copied()
            .unwrap_or_else(|| gltf_matrix(node.transform().matrix())),
    );

    if let Some(mesh) = node.mesh() {
        import_gltf_mesh(
            path,
            mesh,
            transform,
            node.skin(),
            transforms,
            buffers,
            textures,
            meshes,
            materials,
        )?;
    }

    for child in node.children() {
        import_gltf_node(
            path, child, transform, transforms, buffers, textures, meshes, materials,
        )?;
    }

    Ok(())
}

/// Imports every triangle primitive from one glTF mesh with one shared node skin.
#[allow(clippy::too_many_arguments)]
fn import_gltf_mesh(
    path: &Path,
    mesh: gltf::Mesh<'_>,
    transform: [f32; 16],
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &GltfImportedTextures,
    meshes: &mut Vec<ImportedMesh>,
    materials: &mut Vec<ImportedMaterial>,
) -> Result<(), ImportError> {
    for primitive in mesh.primitives() {
        import_gltf_primitive(
            path,
            primitive,
            transform,
            skin.clone(),
            transforms,
            buffers,
            textures,
            meshes,
            materials,
        )?;
    }

    Ok(())
}

/// Imports one glTF triangle primitive as one mesh/material pair.
#[allow(clippy::too_many_arguments)]
fn import_gltf_primitive(
    path: &Path,
    primitive: gltf::Primitive<'_>,
    transform: [f32; 16],
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &GltfImportedTextures,
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

    let material = import_gltf_material(path, primitive.material(), textures)?;
    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let source_positions = reader
        .read_positions()
        .ok_or_else(|| gltf_message(path, "glTF primitive is missing POSITION"))?
        .collect::<Vec<_>>();
    let source_normals = reader
        .read_normals()
        .map(|normals| normals.collect::<Vec<_>>())
        .unwrap_or_default();
    let skinning = load_gltf_skin(path, skin, transforms, buffers)?;
    let joints = reader
        .read_joints(0)
        .map(|joints| joints.into_u16().collect::<Vec<_>>());
    let weights = reader
        .read_weights(0)
        .map(|weights| weights.into_f32().collect::<Vec<_>>());
    let positions = import_gltf_positions(
        &source_positions,
        skinning.as_ref(),
        joints.as_deref(),
        weights.as_deref(),
        transform,
    );
    let uvs = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect::<Vec<_>>())
        .unwrap_or_default();
    let colors = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect::<Vec<_>>())
        .unwrap_or_default();
    let mut indices = match reader.read_indices() {
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
    if transform_swaps_handedness(transform) {
        flip_triangle_winding(&mut indices);
    }
    let normals = import_gltf_normals(
        &source_normals,
        positions.len(),
        skinning.as_ref(),
        joints.as_deref(),
        weights.as_deref(),
        transform,
        &positions,
        &indices,
    );
    let tangents = compute_vertex_tangents(&positions, &normals, &uvs, &indices);

    let vertices = positions
        .into_iter()
        .enumerate()
        .map(|(index, position)| {
            ImportedVertex::new(
                position,
                normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
                uvs.get(index).copied().unwrap_or([0.0, 0.0]),
                tangents.get(index).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
                colors.get(index).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
            )
        })
        .collect::<Vec<_>>();
    meshes.push(ImportedMesh::Indexed(ImportedMeshData::new(
        vertices, indices,
    )));
    materials.push(material);

    Ok(())
}

struct GltfSkin {
    joint_matrices: Vec<[f32; 16]>,
}

/// Builds CPU skinning matrices from glTF joints and inverse-bind matrices.
fn load_gltf_skin(
    path: &Path,
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<GltfSkin>, ImportError> {
    let Some(skin) = skin else {
        return Ok(None);
    };

    let joints = skin.joints().map(|joint| joint.index()).collect::<Vec<_>>();
    let inverse_bind_matrices = skin
        .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()))
        .read_inverse_bind_matrices()
        .map(|matrices| matrices.map(gltf_matrix).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![MAT4_IDENTITY; joints.len()]);

    if inverse_bind_matrices.len() != joints.len() {
        return Err(gltf_message(
            path,
            "glTF skin inverse bind matrix count does not match joint count",
        ));
    }

    let joint_matrices = joints
        .into_iter()
        .zip(inverse_bind_matrices)
        .map(|(joint, inverse_bind)| {
            let joint_transform = transforms
                .global
                .get(joint)
                .copied()
                .unwrap_or(MAT4_IDENTITY);
            mat4_mul(joint_transform, inverse_bind)
        })
        .collect();

    Ok(Some(GltfSkin { joint_matrices }))
}

/// Applies CPU skinning to glTF positions when the primitive has valid joints and weights.
fn import_gltf_positions(
    source_positions: &[[f32; 3]],
    skin: Option<&GltfSkin>,
    joints: Option<&[[u16; 4]]>,
    weights: Option<&[[f32; 4]]>,
    fallback_transform: [f32; 16],
) -> Vec<[f32; 3]> {
    if let Some((skin, joints, weights)) =
        valid_skin_inputs(skin, joints, weights, source_positions.len())
    {
        return source_positions
            .iter()
            .copied()
            .zip(joints.iter().copied())
            .zip(weights.iter().copied())
            .map(|((position, joints), weights)| {
                skin_position(skin, position, joints, weights, fallback_transform)
            })
            .collect();
    }

    source_positions
        .iter()
        .copied()
        .map(|position| transform_position(fallback_transform, position))
        .collect()
}

/// Imports glTF normals with the same skinning path used for positions.
#[allow(clippy::too_many_arguments)]
fn import_gltf_normals(
    source_normals: &[[f32; 3]],
    vertex_count: usize,
    skin: Option<&GltfSkin>,
    joints: Option<&[[u16; 4]]>,
    weights: Option<&[[f32; 4]]>,
    fallback_transform: [f32; 16],
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<[f32; 3]> {
    if source_normals.len() != vertex_count {
        return compute_vertex_normals(positions, indices);
    }

    if let Some((skin, joints, weights)) = valid_skin_inputs(skin, joints, weights, vertex_count) {
        return source_normals
            .iter()
            .copied()
            .zip(joints.iter().copied())
            .zip(weights.iter().copied())
            .map(|((normal, joints), weights)| {
                skin_normal(skin, normal, joints, weights, fallback_transform)
            })
            .collect();
    }

    source_normals
        .iter()
        .copied()
        .map(|normal| transform_normal(fallback_transform, normal))
        .collect()
}

/// Returns skin inputs only when all arrays line up with the primitive vertex count.
fn valid_skin_inputs<'a>(
    skin: Option<&'a GltfSkin>,
    joints: Option<&'a [[u16; 4]]>,
    weights: Option<&'a [[f32; 4]]>,
    vertex_count: usize,
) -> Option<(&'a GltfSkin, &'a [[u16; 4]], &'a [[f32; 4]])> {
    let (Some(skin), Some(joints), Some(weights)) = (skin, joints, weights) else {
        return None;
    };

    (joints.len() == vertex_count && weights.len() == vertex_count)
        .then_some((skin, joints, weights))
}

/// Skins one position with normalized joint weights and falls back to the node transform.
fn skin_position(
    skin: &GltfSkin,
    position: [f32; 3],
    joints: [u16; 4],
    weights: [f32; 4],
    fallback_transform: [f32; 16],
) -> [f32; 3] {
    let mut skinned = [0.0; 3];
    let mut total = 0.0;

    for (&joint, &weight) in joints.iter().zip(weights.iter()) {
        if weight <= 0.0 {
            continue;
        }
        let Some(matrix) = skin.joint_matrices.get(joint as usize) else {
            continue;
        };

        let transformed = transform_position(*matrix, position);
        skinned[0] += transformed[0] * weight;
        skinned[1] += transformed[1] * weight;
        skinned[2] += transformed[2] * weight;
        total += weight;
    }

    if total > f32::EPSILON {
        [skinned[0] / total, skinned[1] / total, skinned[2] / total]
    } else {
        transform_position(fallback_transform, position)
    }
}

/// Skins one normal with the same joints used by the imported vertex position.
fn skin_normal(
    skin: &GltfSkin,
    normal: [f32; 3],
    joints: [u16; 4],
    weights: [f32; 4],
    fallback_transform: [f32; 16],
) -> [f32; 3] {
    let mut skinned = [0.0; 3];
    let mut total = 0.0;

    for (&joint, &weight) in joints.iter().zip(weights.iter()) {
        if weight <= 0.0 {
            continue;
        }
        let Some(matrix) = skin.joint_matrices.get(joint as usize) else {
            continue;
        };

        let transformed = transform_direction(*matrix, normal);
        skinned[0] += transformed[0] * weight;
        skinned[1] += transformed[1] * weight;
        skinned[2] += transformed[2] * weight;
        total += weight;
    }

    if total > f32::EPSILON {
        normalize_or(
            [skinned[0] / total, skinned[1] / total, skinned[2] / total],
            [0.0, 1.0, 0.0],
        )
    } else {
        transform_normal(fallback_transform, normal)
    }
}

/// Converts a glTF material into renderer-independent PBR material data.
fn import_gltf_material(
    path: &Path,
    material: gltf::Material<'_>,
    textures: &GltfImportedTextures,
) -> Result<ImportedMaterial, ImportError> {
    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => MaterialAlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => MaterialAlphaMode::Cutout,
        gltf::material::AlphaMode::Blend => MaterialAlphaMode::Transparent,
    };
    let alpha_cutoff_milli =
        (material.alpha_cutoff().unwrap_or(0.5).clamp(0.0, 1.0) * 1000.0).round() as u16;
    let pbr = material.pbr_metallic_roughness();
    let mut texture_slots = Vec::new();
    push_gltf_texture_slot(
        path,
        textures,
        &mut texture_slots,
        MaterialTextureSlot::BaseColor,
        pbr.base_color_texture(),
        GltfImageColorSpace::Srgb,
    )?;
    push_gltf_texture_slot(
        path,
        textures,
        &mut texture_slots,
        MaterialTextureSlot::MetallicRoughness,
        pbr.metallic_roughness_texture(),
        GltfImageColorSpace::Linear,
    )?;
    push_gltf_texture_slot(
        path,
        textures,
        &mut texture_slots,
        MaterialTextureSlot::Normal,
        material.normal_texture(),
        GltfImageColorSpace::Linear,
    )?;
    push_gltf_texture_slot(
        path,
        textures,
        &mut texture_slots,
        MaterialTextureSlot::Occlusion,
        material.occlusion_texture(),
        GltfImageColorSpace::Linear,
    )?;
    push_gltf_texture_slot(
        path,
        textures,
        &mut texture_slots,
        MaterialTextureSlot::Emissive,
        material.emissive_texture(),
        GltfImageColorSpace::Srgb,
    )?;

    let imported = ImportedMaterial::with_pbr(
        alpha_mode,
        alpha_cutoff_milli,
        pbr.base_color_factor(),
        factor_to_milli(pbr.metallic_factor()),
        factor_to_milli(pbr.roughness_factor()),
        gltf_emissive_factor(&material),
        material
            .occlusion_texture()
            .map(|info| factor_to_milli(info.strength()))
            .unwrap_or(1000),
        material
            .normal_texture()
            .map(|info| positive_factor_to_milli(info.scale(), 4000))
            .unwrap_or(1000),
        material.double_sided(),
        texture_slots,
    );
    tracing::trace!(
        alpha_mode = imported.alpha_mode().name(),
        metallic = imported.metallic_factor_milli(),
        roughness = imported.roughness_factor_milli(),
        textures = imported.texture_slots().len(),
        "imported glTF material"
    );

    Ok(imported)
}

/// Returns emissive color after applying the KHR emissive strength extension when present.
fn gltf_emissive_factor(material: &gltf::Material<'_>) -> [f32; 3] {
    let strength = material.emissive_strength().unwrap_or(1.0).max(0.0);
    material.emissive_factor().map(|value| value * strength)
}

/// Adds a glTF texture info to the imported slot list after validating the source image index.
fn push_gltf_texture_slot<T>(
    path: &Path,
    textures: &GltfImportedTextures,
    texture_slots: &mut Vec<ImportedMaterialTextureSlot>,
    slot: MaterialTextureSlot,
    info: Option<T>,
    color_space: GltfImageColorSpace,
) -> Result<(), ImportError>
where
    T: GltfTextureSource,
{
    let Some(info) = info else {
        return Ok(());
    };
    let Some(index) = textures.texture_index(info.source_index(), color_space) else {
        return Err(gltf_message(
            path,
            "glTF material references a texture image that was not imported",
        ));
    };
    texture_slots.push(ImportedMaterialTextureSlot::new(slot, index));
    Ok(())
}

trait GltfTextureSource {
    fn source_index(&self) -> usize;
}

impl GltfTextureSource for gltf::texture::Info<'_> {
    fn source_index(&self) -> usize {
        self.texture().source().index()
    }
}

impl GltfTextureSource for gltf::material::NormalTexture<'_> {
    fn source_index(&self) -> usize {
        self.texture().source().index()
    }
}

impl GltfTextureSource for gltf::material::OcclusionTexture<'_> {
    fn source_index(&self) -> usize {
        self.texture().source().index()
    }
}

struct GltfImportedTextures {
    textures: Vec<ImportedTexture>,
    srgb_indices: Vec<Option<usize>>,
    linear_indices: Vec<Option<usize>>,
}

impl GltfImportedTextures {
    /// Creates the lookup used by material import to select the correct image color space.
    fn new(
        textures: Vec<ImportedTexture>,
        srgb_indices: Vec<Option<usize>>,
        linear_indices: Vec<Option<usize>>,
    ) -> Self {
        Self {
            textures,
            srgb_indices,
            linear_indices,
        }
    }

    /// Returns the imported texture index for a glTF source image and shader sampling role.
    fn texture_index(
        &self,
        source_index: usize,
        color_space: GltfImageColorSpace,
    ) -> Option<usize> {
        match color_space {
            GltfImageColorSpace::Srgb => self.srgb_indices.get(source_index).copied().flatten(),
            GltfImageColorSpace::Linear => self.linear_indices.get(source_index).copied().flatten(),
        }
    }

    /// Moves the imported texture payloads into the final scene intermediate.
    fn into_textures(self) -> Vec<ImportedTexture> {
        self.textures
    }
}

/// Converts glTF image payloads into renderer-owned texture descriptors by sampling role.
fn import_gltf_textures(
    path: &Path,
    document: &gltf::Document,
    images: &[gltf::image::Data],
) -> Result<GltfImportedTextures, ImportError> {
    let usages = gltf_image_usages(document, images.len());
    let mut textures = Vec::new();
    let mut srgb_indices = vec![None; images.len()];
    let mut linear_indices = vec![None; images.len()];

    for (index, image) in images.iter().enumerate() {
        let usage = usages[index];
        if !usage.srgb && !usage.linear {
            continue;
        }

        let pixels = image_to_rgba8(path, image)?;
        if usage.srgb {
            srgb_indices[index] = Some(push_gltf_texture_descriptor(
                path,
                &mut textures,
                image,
                pixels.clone(),
                GltfImageColorSpace::Srgb,
            )?);
        }
        if usage.linear {
            linear_indices[index] = Some(push_gltf_texture_descriptor(
                path,
                &mut textures,
                image,
                pixels,
                GltfImageColorSpace::Linear,
            )?);
        }
    }

    Ok(GltfImportedTextures::new(
        textures,
        srgb_indices,
        linear_indices,
    ))
}

#[derive(Clone, Copy)]
enum GltfImageColorSpace {
    Srgb,
    Linear,
}

#[derive(Clone, Copy, Default)]
struct GltfImageUsage {
    srgb: bool,
    linear: bool,
}

/// Adds one texture descriptor and returns its imported texture index.
fn push_gltf_texture_descriptor(
    path: &Path,
    textures: &mut Vec<ImportedTexture>,
    image: &gltf::image::Data,
    pixels: Vec<u8>,
    color_space: GltfImageColorSpace,
) -> Result<usize, ImportError> {
    let descriptor = match color_space {
        GltfImageColorSpace::Srgb => {
            TextureDescriptor::rgba8_srgb(image.width, image.height, pixels)
        }
        GltfImageColorSpace::Linear => {
            TextureDescriptor::rgba8_linear(image.width, image.height, pixels)
        }
    };
    let texture = descriptor
        .map(ImportedTexture::new)
        .ok_or_else(|| gltf_message(path, "glTF image payload has an invalid RGBA8 byte count"))?;
    let index = textures.len();
    textures.push(texture);
    Ok(index)
}

/// Classifies glTF source images so color and data uses can create separate descriptors.
fn gltf_image_usages(document: &gltf::Document, image_count: usize) -> Vec<GltfImageUsage> {
    let mut usages = vec![GltfImageUsage::default(); image_count];
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        mark_gltf_texture_usage(
            &mut usages,
            pbr.base_color_texture(),
            GltfImageColorSpace::Srgb,
        );
        mark_gltf_texture_usage(
            &mut usages,
            pbr.metallic_roughness_texture(),
            GltfImageColorSpace::Linear,
        );
        mark_gltf_texture_usage(
            &mut usages,
            material.normal_texture(),
            GltfImageColorSpace::Linear,
        );
        mark_gltf_texture_usage(
            &mut usages,
            material.occlusion_texture(),
            GltfImageColorSpace::Linear,
        );
        mark_gltf_texture_usage(
            &mut usages,
            material.emissive_texture(),
            GltfImageColorSpace::Srgb,
        );
    }

    usages
}

/// Marks how one glTF source image is sampled by a material slot.
fn mark_gltf_texture_usage<T>(
    usages: &mut [GltfImageUsage],
    info: Option<T>,
    color_space: GltfImageColorSpace,
) where
    T: GltfTextureSource,
{
    let Some(info) = info else {
        return;
    };
    let Some(usage) = usages.get_mut(info.source_index()) else {
        return;
    };
    match color_space {
        GltfImageColorSpace::Srgb => usage.srgb = true,
        GltfImageColorSpace::Linear => usage.linear = true,
    }
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

/// Applies the inverse-transpose normal matrix required by non-uniform glTF node scale.
fn transform_normal(matrix: [f32; 16], normal: [f32; 3]) -> [f32; 3] {
    let normal_matrix = inverse_transpose_3x3(matrix).unwrap_or_else(|| upper_3x3(matrix));
    normalize_or(mul_3x3(normal_matrix, normal), [0.0, 1.0, 0.0])
}

/// Applies the upper 3x3 transform to a direction without translation.
fn transform_direction(matrix: [f32; 16], direction: [f32; 3]) -> [f32; 3] {
    normalize_or(
        [
            matrix[0] * direction[0] + matrix[4] * direction[1] + matrix[8] * direction[2],
            matrix[1] * direction[0] + matrix[5] * direction[1] + matrix[9] * direction[2],
            matrix[2] * direction[0] + matrix[6] * direction[1] + matrix[10] * direction[2],
        ],
        [0.0, 1.0, 0.0],
    )
}

/// Returns whether a transform changes handedness and therefore triangle winding.
fn transform_swaps_handedness(matrix: [f32; 16]) -> bool {
    determinant_3x3(upper_3x3(matrix)) < 0.0
}

/// Reverses each triangle so a negative-scale node still matches the renderer front face.
fn flip_triangle_winding(indices: &mut [u32]) {
    for triangle in indices.chunks_exact_mut(3) {
        triangle.swap(1, 2);
    }
}

/// Converts a factor in 0..1 into deterministic milli units.
fn factor_to_milli(value: f32) -> u16 {
    if value.is_finite() {
        (value.clamp(0.0, 1.0) * 1000.0).round() as u16
    } else {
        0
    }
}

/// Converts a non-negative scalar into deterministic milli units with an explicit cap.
fn positive_factor_to_milli(value: f32, max_milli: u16) -> u16 {
    if value.is_finite() {
        ((value.max(0.0) * 1000.0).round() as u16).min(max_milli)
    } else {
        0
    }
}

/// Builds a flat column-major matrix from glTF TRS animation components.
fn trs_matrix(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [f32; 16] {
    let [x, y, z, w] = normalize_quat(rotation);
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

/// Normalizes a glTF quaternion while preserving identity for invalid zero rotations.
fn normalize_quat(quat: [f32; 4]) -> [f32; 4] {
    let len =
        (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();

    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [quat[0] / len, quat[1] / len, quat[2] / len, quat[3] / len]
    }
}

/// Returns the upper 3x3 matrix embedded in one flat column-major 4x4 matrix.
fn upper_3x3(matrix: [f32; 16]) -> [[f32; 3]; 3] {
    [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]],
    ]
}

/// Multiplies a column-major 3x3 matrix by one vector.
fn mul_3x3(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * vector[0] + matrix[1][0] * vector[1] + matrix[2][0] * vector[2],
        matrix[0][1] * vector[0] + matrix[1][1] * vector[1] + matrix[2][1] * vector[2],
        matrix[0][2] * vector[0] + matrix[1][2] * vector[1] + matrix[2][2] * vector[2],
    ]
}

/// Computes the determinant of a column-major 3x3 matrix.
fn determinant_3x3(matrix: [[f32; 3]; 3]) -> f32 {
    let a = matrix[0];
    let b = matrix[1];
    let c = matrix[2];

    a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
}

/// Returns the inverse-transpose of a column-major 3x3 matrix.
fn inverse_transpose_3x3(matrix: [f32; 16]) -> Option<[[f32; 3]; 3]> {
    let m = upper_3x3(matrix);
    let determinant = determinant_3x3(m);
    if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return None;
    }
    let inv_det = 1.0 / determinant;

    let inverse = [
        [
            (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det,
            (m[2][1] * m[0][2] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * inv_det,
        ],
        [
            (m[2][0] * m[1][2] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * inv_det,
            (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det,
            (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det,
        ],
    ];

    Some(transpose_3x3(inverse))
}

/// Transposes one column-major 3x3 matrix.
fn transpose_3x3(matrix: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
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

/// Computes per-vertex tangents from positions, normals, UVs, and triangles.
///
/// glTF normal maps require a stable tangent frame. This importer computes the frame after
/// node transform / CPU skinning so the shader can use TANGENT.xyz + TANGENT.w instead of a
/// fragile derivative-only TBN.
fn compute_vertex_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut tangent_sum = vec![[0.0, 0.0, 0.0]; positions.len()];
    let mut bitangent_sum = vec![[0.0, 0.0, 0.0]; positions.len()];

    if uvs.len() != positions.len() || normals.len() != positions.len() {
        return (0..positions.len())
            .map(|index| fallback_tangent(normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0])))
            .collect();
    }

    for triangle in indices.chunks_exact(3) {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let dp1 = sub3(p1, p0);
        let dp2 = sub3(p2, p0);
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
        let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if !det.is_finite() || det.abs() <= 1e-8 {
            continue;
        }

        let inv_det = 1.0 / det;
        let tangent = [
            (dp1[0] * duv2[1] - dp2[0] * duv1[1]) * inv_det,
            (dp1[1] * duv2[1] - dp2[1] * duv1[1]) * inv_det,
            (dp1[2] * duv2[1] - dp2[2] * duv1[1]) * inv_det,
        ];
        let bitangent = [
            (dp2[0] * duv1[0] - dp1[0] * duv2[0]) * inv_det,
            (dp2[1] * duv1[0] - dp1[1] * duv2[0]) * inv_det,
            (dp2[2] * duv1[0] - dp1[2] * duv2[0]) * inv_det,
        ];

        for index in [i0, i1, i2] {
            tangent_sum[index] = add3(tangent_sum[index], tangent);
            bitangent_sum[index] = add3(bitangent_sum[index], bitangent);
        }
    }

    normals
        .iter()
        .copied()
        .enumerate()
        .map(|(index, normal)| {
            let n = normalize_or(normal, [0.0, 1.0, 0.0]);
            let t_raw = tangent_sum[index];
            let t_ortho = sub3(
                t_raw,
                [
                    n[0] * dot3(n, t_raw),
                    n[1] * dot3(n, t_raw),
                    n[2] * dot3(n, t_raw),
                ],
            );
            let fallback = fallback_tangent(n);
            let tangent = normalize_or(t_ortho, [fallback[0], fallback[1], fallback[2]]);
            let b = bitangent_sum[index];
            let sign = if dot3(cross3(n, tangent), b) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [tangent[0], tangent[1], tangent[2], sign]
        })
        .collect()
}

/// Normalizes tangent xyz and keeps tangent.w as a handedness sign.
fn normalize_tangent(tangent: [f32; 4], normal: [f32; 3]) -> [f32; 4] {
    let n = normalize_or(normal, [0.0, 1.0, 0.0]);
    let t = [tangent[0], tangent[1], tangent[2]];
    let orthogonal = sub3(t, [n[0] * dot3(n, t), n[1] * dot3(n, t), n[2] * dot3(n, t)]);
    let fallback = fallback_tangent(n);
    let normalized = normalize_or(orthogonal, [fallback[0], fallback[1], fallback[2]]);
    [
        normalized[0],
        normalized[1],
        normalized[2],
        if tangent[3] < 0.0 { -1.0 } else { 1.0 },
    ]
}

/// Returns any stable tangent orthogonal to a normal.
fn fallback_tangent(normal: [f32; 3]) -> [f32; 4] {
    let n = normalize_or(normal, [0.0, 1.0, 0.0]);
    let axis = if n[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize_or(cross3(axis, n), [1.0, 0.0, 0.0]);
    [tangent[0], tangent[1], tangent[2], 1.0]
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
    let mut base_color_factor = [1.0, 1.0, 1.0, 1.0];
    let mut metallic_factor_milli = 0;
    let mut roughness_factor_milli = 1000;
    let mut emissive_factor = [0.0, 0.0, 0.0];
    let mut occlusion_strength_milli = 1000;
    let mut normal_scale_milli = 1000;
    let mut double_sided = false;
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
        if name == "base_color_factor" {
            base_color_factor = parse_vec4(path, line, value, "base_color_factor")?;
            continue;
        }
        if name == "emissive" {
            emissive_factor = parse_vec3_non_negative(path, line, value, "emissive")?;
            continue;
        }
        if name == "metallic" {
            metallic_factor_milli = parse_unit_milli(path, line, value, "metallic")?;
            continue;
        }
        if name == "roughness" {
            roughness_factor_milli = parse_unit_milli(path, line, value, "roughness")?;
            continue;
        }
        if name == "occlusion" {
            occlusion_strength_milli = parse_unit_milli(path, line, value, "occlusion")?;
            continue;
        }
        if name == "normal_scale" {
            normal_scale_milli = parse_positive_milli(path, line, value, "normal_scale", 4000)?;
            continue;
        }
        if name == "double_sided" {
            double_sided = parse_bool(path, line, value, "double_sided")?;
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

    Ok(ImportedMaterial::with_pbr(
        alpha_mode,
        alpha_cutoff_milli,
        base_color_factor,
        metallic_factor_milli,
        roughness_factor_milli,
        emissive_factor,
        occlusion_strength_milli,
        normal_scale_milli,
        double_sided,
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

/// Parses one finite scalar used by material factor directives.
fn parse_f32_directive(
    path: &Path,
    line: usize,
    value: &str,
    name: &str,
) -> Result<f32, ImportError> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| invalid_directive(path, line, format!("invalid {name}: {value}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid_directive(
            path,
            line,
            format!("{name} must be finite: {value}"),
        ))
    }
}

/// Parses a unit material factor and stores it as deterministic milli units.
fn parse_unit_milli(path: &Path, line: usize, value: &str, name: &str) -> Result<u16, ImportError> {
    let parsed = parse_f32_directive(path, line, value, name)?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(invalid_directive(
            path,
            line,
            format!("{name} must be between 0 and 1: {value}"),
        ));
    }

    Ok((parsed * 1000.0).round() as u16)
}

/// Parses a non-negative factor with an explicit milli-unit ceiling.
fn parse_positive_milli(
    path: &Path,
    line: usize,
    value: &str,
    name: &str,
    max_milli: u16,
) -> Result<u16, ImportError> {
    let parsed = parse_f32_directive(path, line, value, name)?;
    if parsed < 0.0 || parsed > f32::from(max_milli) / 1000.0 {
        return Err(invalid_directive(
            path,
            line,
            format!("{name} is outside the supported range: {value}"),
        ));
    }

    Ok((parsed * 1000.0).round() as u16)
}

/// Parses a boolean directive without accepting ambiguous spellings.
fn parse_bool(path: &Path, line: usize, value: &str, name: &str) -> Result<bool, ImportError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid_directive(
            path,
            line,
            format!("{name} must be true or false: {value}"),
        )),
    }
}

/// Parses a comma-separated RGB factor for simple verification scene materials.
fn parse_vec3_non_negative(
    path: &Path,
    line: usize,
    value: &str,
    name: &str,
) -> Result<[f32; 3], ImportError> {
    let values = parse_f32_list(path, line, value, name, 3)?;
    if values.iter().all(|value| *value >= 0.0) {
        Ok([values[0], values[1], values[2]])
    } else {
        Err(invalid_directive(
            path,
            line,
            format!("{name} channels must be non-negative: {value}"),
        ))
    }
}

/// Parses a comma-separated RGBA factor for simple verification scene materials.
fn parse_vec4(path: &Path, line: usize, value: &str, name: &str) -> Result<[f32; 4], ImportError> {
    let values = parse_f32_list(path, line, value, name, 4)?;
    if values.iter().all(|value| (0.0..=1.0).contains(value)) {
        Ok([values[0], values[1], values[2], values[3]])
    } else {
        Err(invalid_directive(
            path,
            line,
            format!("{name} channels must be between 0 and 1: {value}"),
        ))
    }
}

/// Parses a fixed-length comma-separated list of finite floats.
fn parse_f32_list(
    path: &Path,
    line: usize,
    value: &str,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, ImportError> {
    let values = value
        .split(',')
        .map(|part| parse_f32_directive(path, line, part, name))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() == expected_len {
        Ok(values)
    } else {
        Err(invalid_directive(
            path,
            line,
            format!("{name} expects {expected_len} comma-separated values: {value}"),
        ))
    }
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
        let text = "rebuild1-scene\ntexture solid 255 0 0 255\nmaterial cutout base_color=0 alpha_cutoff=0.4 roughness=0.25 metallic=0.75 double_sided=true\nmesh plane\n";

        let scene = parse_rebuild1_scene(&path, text).expect("manifest should parse");
        let material = &scene.materials()[0];

        assert_eq!(scene.texture_count(), 1);
        assert_eq!(material.alpha_mode(), MaterialAlphaMode::Cutout);
        assert_eq!(material.alpha_cutoff_milli(), 400);
        assert_eq!(material.roughness_factor_milli(), 250);
        assert_eq!(material.metallic_factor_milli(), 750);
        assert!(material.double_sided());
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

    // Verifies that negative-scale glTF nodes keep their front-face winding after import.
    #[test]
    fn negative_handed_transform_flips_triangle_winding() {
        let mut indices = vec![0, 1, 2, 2, 3, 0];
        let negative_x = [
            -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        assert!(transform_swaps_handedness(negative_x));
        flip_triangle_winding(&mut indices);

        assert_eq!(indices, vec![0, 2, 1, 2, 0, 3]);
    }

    // Verifies that non-uniform scale uses a normal matrix instead of the raw object matrix.
    #[test]
    fn normal_transform_uses_inverse_transpose() {
        let scale = [
            2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let normal = transform_normal(scale, [1.0, 1.0, 0.0]);

        assert!(normal[1] > normal[0]);
    }

    // Verifies that the GLB importer applies CPU skinning instead of leaving bind-pose vertices.
    #[test]
    fn skin_position_applies_weighted_joint_transform() {
        let skin = GltfSkin {
            joint_matrices: vec![
                MAT4_IDENTITY,
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 1.0,
                ],
            ],
        };

        let position = skin_position(
            &skin,
            [1.0, 2.0, 3.0],
            [0, 1, 0, 0],
            [0.25, 0.75, 0.0, 0.0],
            MAT4_IDENTITY,
        );

        assert_eq!(position, [4.0, 2.0, 3.0]);
    }

    // Verifies that invalid or zero-weight skinning falls back to the mesh node transform.
    #[test]
    fn skin_position_falls_back_without_valid_weights() {
        let skin = GltfSkin {
            joint_matrices: Vec::new(),
        };
        let fallback = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0,
        ];

        let position = skin_position(
            &skin,
            [1.0, 2.0, 3.0],
            [7, 8, 9, 10],
            [0.0, 0.0, 0.0, 0.0],
            fallback,
        );

        assert_eq!(position, [3.0, 5.0, 7.0]);
    }
}
