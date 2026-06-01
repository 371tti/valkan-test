use crate::protocol::MaterialTextureSlot;

pub(crate) const FRAME_SET: u32 = 0;
pub(crate) const MATERIAL_SET: u32 = 1;
pub(crate) const PASS_SET: u32 = 2;

pub(crate) const FRAME_TINT_BINDING: u32 = 0;
pub(crate) const FRAME_CAMERA_BINDING: u32 = 0;
pub(crate) const MATERIAL_PARAMS_BINDING: u32 = 0;
pub(crate) const MATERIAL_BASE_COLOR_BINDING: u32 = 1;
pub(crate) const MATERIAL_NORMAL_BINDING: u32 = 2;
pub(crate) const MATERIAL_METALLIC_ROUGHNESS_BINDING: u32 = 3;
pub(crate) const MATERIAL_OCCLUSION_BINDING: u32 = 4;
pub(crate) const MATERIAL_EMISSIVE_BINDING: u32 = 5;
pub(crate) const PASS_SHADOW_MAP_BINDING: u32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShaderBinding {
    pub(crate) set: u32,
    pub(crate) binding: u32,
    pub(crate) name: &'static str,
}

pub(crate) const MESH_SHADER_BINDINGS: &[ShaderBinding] = &[
    ShaderBinding {
        set: FRAME_SET,
        binding: FRAME_CAMERA_BINDING,
        name: "frame_camera",
    },
    ShaderBinding {
        set: MATERIAL_SET,
        binding: MATERIAL_PARAMS_BINDING,
        name: "material_params",
    },
    ShaderBinding {
        set: MATERIAL_SET,
        binding: MATERIAL_BASE_COLOR_BINDING,
        name: "base_color",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_SHADOW_MAP_BINDING,
        name: "shadow_map",
    },
];

/// Validates the hard-coded mesh shader descriptor contract before Vulkan layouts are created.
pub(crate) fn validate_mesh_interface() -> Result<(), &'static str> {
    if FRAME_SET != 0 || MATERIAL_SET != 1 || PASS_SET != 2 {
        return Err("mesh shaders require frame/material/pass descriptor set order 0/1/2");
    }
    if MATERIAL_BASE_COLOR_BINDING == MATERIAL_PARAMS_BINDING {
        return Err("material params and base-color texture must use different bindings");
    }
    if MESH_SHADER_BINDINGS.is_empty() {
        return Err("mesh shader interface must declare at least one binding");
    }

    Ok(())
}

/// Returns the shader binding assigned to one named material texture slot.
pub(crate) fn material_texture_binding(slot: MaterialTextureSlot) -> u32 {
    match slot {
        MaterialTextureSlot::BaseColor => MATERIAL_BASE_COLOR_BINDING,
        MaterialTextureSlot::Normal => MATERIAL_NORMAL_BINDING,
        MaterialTextureSlot::MetallicRoughness => MATERIAL_METALLIC_ROUGHNESS_BINDING,
        MaterialTextureSlot::Occlusion => MATERIAL_OCCLUSION_BINDING,
        MaterialTextureSlot::Emissive => MATERIAL_EMISSIVE_BINDING,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that material slot bindings stay centralized for Stage 6 pipelines.
    #[test]
    fn base_color_binding_is_named_once() {
        assert_eq!(
            material_texture_binding(MaterialTextureSlot::BaseColor),
            MATERIAL_BASE_COLOR_BINDING
        );
    }

    // Verifies that Stage 8 mesh shaders keep the descriptor set contract explicit.
    #[test]
    fn mesh_interface_contract_is_valid() {
        validate_mesh_interface().expect("mesh shader interface should be internally consistent");
    }
}
