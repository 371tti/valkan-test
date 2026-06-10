#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    vec4 emissive_occlusion;
    vec4 pbr_alpha;
    uvec4 flags;
} material;

layout(set = 1, binding = 1) uniform sampler2D base_color_texture;

const uint TEX_BASE_COLOR = 1u << 0;
#include "shadow_alpha.glsl"

bool has_base_color_texture() {
    return (material.flags.y & TEX_BASE_COLOR) != 0u;
}

void main() {
    vec4 base_color = frag_color * material.base_color_factor;
    if (has_base_color_texture()) {
        base_color *= texture(base_color_texture, frag_uv);
    }
    discard_opaque_shadow_alpha(material.flags.x, material.pbr_alpha.z, base_color.a);
}
