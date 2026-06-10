use std::collections::BTreeMap;

use crate::{
    import::ImportedMaterial,
    protocol::{MaterialDescriptor, TextureHandle},
    renderer::pipeline::shader_interface,
};

use super::texture::GpuTextureAsset;

#[derive(Clone, Debug)]
pub(crate) struct GpuMaterialAsset {
    descriptor: MaterialDescriptor,
}

impl GpuMaterialAsset {
    /// Builds a renderer material record after imported texture indices are mapped to handles.
    pub(crate) fn from_imported(imported: &ImportedMaterial, textures: &[TextureHandle]) -> Self {
        let descriptor = resolve_material_descriptor(imported, textures);
        tracing::trace!(
            alpha_mode = descriptor.alpha_mode().name(),
            binding = shader_interface::MATERIAL_PARAMS_BINDING,
            "registered material parameter binding"
        );
        for slot in descriptor.textures().keys() {
            tracing::trace!(
                slot = slot.name(),
                binding = shader_interface::material_texture_binding(*slot),
                "registered material texture slot"
            );
        }

        Self { descriptor }
    }

    /// Returns whether every texture handle referenced by this material has a payload record.
    pub(crate) fn is_draw_ready(
        &self,
        textures: &BTreeMap<TextureHandle, GpuTextureAsset>,
    ) -> bool {
        self.descriptor.textures().values().all(|texture| {
            textures
                .get(texture)
                .is_some_and(GpuTextureAsset::has_pixels)
        })
    }

    /// Returns the immutable material descriptor consumed by backend descriptor upload.
    pub(crate) fn descriptor(&self) -> &MaterialDescriptor {
        &self.descriptor
    }
}

/// Resolves imported material texture indices into protocol texture handles.
fn resolve_material_descriptor(
    imported: &ImportedMaterial,
    textures: &[TextureHandle],
) -> MaterialDescriptor {
    let mut descriptor = MaterialDescriptor::with_pbr(
        imported.alpha_mode(),
        imported.alpha_cutoff_milli(),
        imported.base_color_factor(),
        imported.metallic_factor_milli(),
        imported.roughness_factor_milli(),
        imported.emissive_factor(),
        imported.occlusion_strength_milli(),
        imported.normal_scale_milli(),
        imported.double_sided(),
    )
    .expect("imported material constrains PBR values before GPU upload");
    for slot in imported.texture_slots() {
        if let Some(texture) = textures.get(slot.texture_index()).copied() {
            descriptor.set_texture(slot.slot(), texture);
        }
    }
    descriptor
}
