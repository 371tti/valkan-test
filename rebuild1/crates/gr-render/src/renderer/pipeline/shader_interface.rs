use crate::protocol::MaterialTextureSlot;
use crate::renderer::graph::SHADOW_CASCADE_COUNT;

pub(crate) const FRAME_SET: u32 = 0;
pub(crate) const MATERIAL_SET: u32 = 1;
pub(crate) const PASS_SET: u32 = 2;

pub(crate) const FRAME_CAMERA_BINDING: u32 = 0;
pub(crate) const MATERIAL_PARAMS_BINDING: u32 = 0;
pub(crate) const MATERIAL_BASE_COLOR_BINDING: u32 = 1;
pub(crate) const MATERIAL_NORMAL_BINDING: u32 = 2;
pub(crate) const MATERIAL_METALLIC_ROUGHNESS_BINDING: u32 = 3;
pub(crate) const MATERIAL_OCCLUSION_BINDING: u32 = 4;
pub(crate) const MATERIAL_EMISSIVE_BINDING: u32 = 5;
pub(crate) const PASS_SHADOW_CASCADE_BINDINGS: [u32; SHADOW_CASCADE_COUNT] = [0, 1, 2];
pub(crate) const PASS_TRANSLUCENT_SHADOW_BINDINGS: [u32; SHADOW_CASCADE_COUNT] = [3, 4, 5];

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
        set: MATERIAL_SET,
        binding: MATERIAL_NORMAL_BINDING,
        name: "normal",
    },
    ShaderBinding {
        set: MATERIAL_SET,
        binding: MATERIAL_METALLIC_ROUGHNESS_BINDING,
        name: "metallic_roughness",
    },
    ShaderBinding {
        set: MATERIAL_SET,
        binding: MATERIAL_OCCLUSION_BINDING,
        name: "occlusion",
    },
    ShaderBinding {
        set: MATERIAL_SET,
        binding: MATERIAL_EMISSIVE_BINDING,
        name: "emissive",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_SHADOW_CASCADE_BINDINGS[0],
        name: "shadow_cascade_0",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_SHADOW_CASCADE_BINDINGS[1],
        name: "shadow_cascade_1",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_SHADOW_CASCADE_BINDINGS[2],
        name: "shadow_cascade_2",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_TRANSLUCENT_SHADOW_BINDINGS[0],
        name: "translucent_shadow_0",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_TRANSLUCENT_SHADOW_BINDINGS[1],
        name: "translucent_shadow_1",
    },
    ShaderBinding {
        set: PASS_SET,
        binding: PASS_TRANSLUCENT_SHADOW_BINDINGS[2],
        name: "translucent_shadow_2",
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
    let mut bindings = [
        MATERIAL_BASE_COLOR_BINDING,
        MATERIAL_NORMAL_BINDING,
        MATERIAL_METALLIC_ROUGHNESS_BINDING,
        MATERIAL_OCCLUSION_BINDING,
        MATERIAL_EMISSIVE_BINDING,
    ];
    bindings.sort_unstable();
    if bindings.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("material texture bindings must be unique");
    }
    let mut pass_bindings = [
        PASS_SHADOW_CASCADE_BINDINGS[0],
        PASS_SHADOW_CASCADE_BINDINGS[1],
        PASS_SHADOW_CASCADE_BINDINGS[2],
        PASS_TRANSLUCENT_SHADOW_BINDINGS[0],
        PASS_TRANSLUCENT_SHADOW_BINDINGS[1],
        PASS_TRANSLUCENT_SHADOW_BINDINGS[2],
    ];
    pass_bindings.sort_unstable();
    if pass_bindings.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("pass shadow bindings must be unique");
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
