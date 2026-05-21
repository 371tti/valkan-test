use std::{io, path::Path};

use crate::renderer::{Material, MaterialAlpha, ModelVertex, TextureId, mat4_mul};

use super::{
    CpuMesh, CpuModel, CpuPrimitive, CpuTexture, TextureFilter, TextureSampler, TextureWrap,
    infer_base_color_alpha, invalid_model, invalid_model_error, mark_texture_linear, model_error,
    normalize_or, prepare_base_color_texture, triangle_normal,
};

const MAT4_IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

pub(super) fn load(path: &Path) -> io::Result<CpuModel> {
    let (document, buffers, images) =
        gltf::import(path).map_err(|source| model_error(path, source))?;
    let mut primitives = Vec::new();
    let mut textures = document
        .textures()
        .map(|texture| gltf_texture_to_cpu(texture, &images))
        .collect::<io::Result<Vec<_>>>()?;
    let transforms = gltf_scene_transforms(&document, &buffers)?;

    if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            load_gltf_node(
                node,
                MAT4_IDENTITY,
                &transforms,
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
                    &transforms,
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
                None,
                &transforms,
                &buffers,
                &mut textures,
                &mut primitives,
            )?;
        }
    }

    Ok(CpuModel {
        primitives,
        textures,
    })
}

#[derive(Clone, Copy)]
struct NodePose {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl NodePose {
    fn from_transform(transform: gltf::scene::Transform) -> Self {
        let (translation, rotation, scale) = transform.decomposed();
        Self {
            translation,
            rotation,
            scale,
        }
    }

    fn matrix(self) -> [f32; 16] {
        trs_matrix(self.translation, self.rotation, self.scale)
    }
}

struct GltfSceneTransforms {
    local: Vec<[f32; 16]>,
    global: Vec<[f32; 16]>,
}

fn gltf_scene_transforms(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> io::Result<GltfSceneTransforms> {
    let mut poses = document
        .nodes()
        .map(|node| NodePose::from_transform(node.transform()))
        .collect::<Vec<_>>();
    apply_first_animation_pose(document, buffers, &mut poses)?;

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
            collect_gltf_node_transforms(node, MAT4_IDENTITY, &local, &mut global);
        }
    }

    Ok(GltfSceneTransforms { local, global })
}

fn apply_first_animation_pose(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    poses: &mut [NodePose],
) -> io::Result<()> {
    let Some(animation) = document.animations().next() else {
        return Ok(());
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

    Ok(())
}

fn first_animation_sample(times: impl Iterator<Item = f32>) -> usize {
    times
        .enumerate()
        .find_map(|(index, time)| (time >= 0.0).then_some(index))
        .unwrap_or(0)
}

fn collect_gltf_node_transforms(
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
        collect_gltf_node_transforms(child, transform, local, global);
    }
}

fn load_gltf_node(
    node: gltf::Node<'_>,
    parent_transform: [f32; 16],
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
    primitives: &mut Vec<CpuPrimitive>,
) -> io::Result<()> {
    let transform = mat4_mul(
        parent_transform,
        transforms
            .local
            .get(node.index())
            .copied()
            .unwrap_or(MAT4_IDENTITY),
    );

    if let Some(mesh) = node.mesh() {
        load_gltf_mesh(
            mesh,
            transform,
            node.skin(),
            transforms,
            buffers,
            textures,
            primitives,
        )?;
    }

    for child in node.children() {
        load_gltf_node(child, transform, transforms, buffers, textures, primitives)?;
    }

    Ok(())
}

fn load_gltf_mesh(
    mesh: gltf::Mesh<'_>,
    transform: [f32; 16],
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
    primitives: &mut Vec<CpuPrimitive>,
) -> io::Result<()> {
    for primitive in mesh.primitives() {
        if let Some(primitive) = load_gltf_primitive(
            primitive,
            transform,
            skin.clone(),
            transforms,
            buffers,
            textures,
        )? {
            primitives.push(primitive);
        }
    }

    Ok(())
}

fn load_gltf_primitive(
    primitive: gltf::Primitive<'_>,
    transform: [f32; 16],
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
    textures: &mut [CpuTexture],
) -> io::Result<Option<CpuPrimitive>> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Ok(None);
    }

    let material = material_from_gltf(primitive.material(), textures);
    let reader =
        primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
    let source_positions = reader
        .read_positions()
        .ok_or_else(|| invalid_model_error("glTF primitive is missing POSITION"))?
        .collect::<Vec<_>>();
    let source_normals = reader
        .read_normals()
        .map(|normals| normals.collect::<Vec<_>>())
        .unwrap_or_default();
    let skinning = load_gltf_skin(skin, transforms, buffers)?;
    let joints = reader
        .read_joints(0)
        .map(|joints| joints.into_u16().collect::<Vec<_>>());
    let weights = reader
        .read_weights(0)
        .map(|weights| weights.into_f32().collect::<Vec<_>>());
    let positions = skinning
        .as_ref()
        .zip(joints.as_ref())
        .zip(weights.as_ref())
        .filter(|((_, joints), weights)| {
            joints.len() == source_positions.len() && weights.len() == source_positions.len()
        })
        .map(|((skinning, joints), weights)| {
            source_positions
                .iter()
                .copied()
                .zip(joints.iter().copied())
                .zip(weights.iter().copied())
                .map(|((position, joints), weights)| {
                    skin_position(skinning, position, joints, weights, transform)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            source_positions
                .iter()
                .copied()
                .map(|position| transform_position(transform, position))
                .collect()
        });
    let normals = if source_normals.len() == source_positions.len() {
        skinning
            .as_ref()
            .zip(joints.as_ref())
            .zip(weights.as_ref())
            .filter(|((_, joints), weights)| {
                joints.len() == source_normals.len() && weights.len() == source_normals.len()
            })
            .map(|((skinning, joints), weights)| {
                source_normals
                    .iter()
                    .copied()
                    .zip(joints.iter().copied())
                    .zip(weights.iter().copied())
                    .map(|((normal, joints), weights)| {
                        skin_normal(skinning, normal, joints, weights, transform)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                source_normals
                    .iter()
                    .copied()
                    .map(|normal| transform_normal(transform, normal))
                    .collect()
            })
    } else {
        Vec::new()
    };
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

struct GltfSkin {
    joint_matrices: Vec<[f32; 16]>,
}

fn load_gltf_skin(
    skin: Option<gltf::Skin<'_>>,
    transforms: &GltfSceneTransforms,
    buffers: &[gltf::buffer::Data],
) -> io::Result<Option<GltfSkin>> {
    let Some(skin) = skin else {
        return Ok(None);
    };

    let joints = skin.joints().map(|joint| joint.index()).collect::<Vec<_>>();
    let inverse_bind_matrices = skin
        .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()))
        .read_inverse_bind_matrices()
        .map(|matrices| matrices.map(gltf_mat4).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![MAT4_IDENTITY; joints.len()]);

    if inverse_bind_matrices.len() != joints.len() {
        return invalid_model("glTF skin inverse bind matrix count does not match joint count");
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

fn material_from_gltf(gltf_material: gltf::Material<'_>, textures: &mut [CpuTexture]) -> Material {
    if gltf_material.index().is_none() {
        return Material::default();
    }

    let pbr = gltf_material.pbr_metallic_roughness();
    let alpha = match gltf_material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => MaterialAlpha::Opaque,
        gltf::material::AlphaMode::Mask => {
            MaterialAlpha::Mask(gltf_material.alpha_cutoff().unwrap_or(0.5))
        }
        gltf::material::AlphaMode::Blend => MaterialAlpha::Blend,
    };
    let base_color_texture = pbr
        .base_color_texture()
        .map(|info| TextureId(info.texture().index()));
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .map(|info| TextureId(info.texture().index()));
    let normal_texture = gltf_material.normal_texture();
    let occlusion_texture = gltf_material.occlusion_texture();
    let emissive_texture = gltf_material
        .emissive_texture()
        .map(|info| TextureId(info.texture().index()));

    let mut material = Material::new(pbr.base_color_factor())
        .with_metallic(pbr.metallic_factor())
        .with_roughness(pbr.roughness_factor())
        .with_emissive(gltf_material.emissive_factor())
        .with_emissive_strength(gltf_material.emissive_strength().unwrap_or(1.0))
        .with_alpha_mode(alpha)
        .with_double_sided(gltf_material.double_sided());
    if let Some(specular) = gltf_material.specular() {
        material = material
            .with_specular(specular.specular_factor())
            .with_specular_color(specular.specular_color_factor());
    }
    if let Some(transmission) = gltf_material.transmission() {
        let transmission = transmission.transmission_factor().clamp(0.0, 1.0);
        if transmission > 0.001 {
            material = material
                .with_transmission(transmission)
                .with_alpha_blend(true);
        }
    }

    if let Some(texture) = base_color_texture {
        material = material.with_base_color_texture(texture);
        prepare_base_color_texture(textures, texture);
        material = infer_base_color_alpha(material, texture, textures);
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

fn gltf_mat4(matrix: [[f32; 4]; 4]) -> [f32; 16] {
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

fn transform_position(matrix: [f32; 16], position: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2] + matrix[12],
        matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2] + matrix[13],
        matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2] + matrix[14],
    ]
}

fn transform_normal(matrix: [f32; 16], normal: [f32; 3]) -> [f32; 3] {
    normalize_or(transform_direction(matrix, normal), [0.0, 1.0, 0.0])
}

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

fn normalize_quat(quat: [f32; 4]) -> [f32; 4] {
    let len =
        (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();

    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [quat[0] / len, quat[1] / len, quat[2] / len, quat[3] / len]
    }
}
