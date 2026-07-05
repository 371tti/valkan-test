#version 450
#extension GL_GOOGLE_include_directive : require

#include "shadow_alpha.glsl"

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    vec4 emissive_occlusion;
    vec4 pbr_alpha;
    uvec4 flags;
} material;

void main() {
    float alpha = frag_color.a * material.base_color_factor.a;
    discard_opaque_shadow_alpha(
        material.flags.x,
        material.pbr_alpha.z,
        alpha
    );
}
