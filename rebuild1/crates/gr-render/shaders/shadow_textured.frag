#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 out_moments;

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

vec4 shadow_moments(float depth) {
    depth = clamp(depth, 0.0, 1.0);

    float depth2 = depth * depth;
    float depth3 = depth2 * depth;
    float depth4 = depth2 * depth2;

    float dx = dFdx(depth);
    float dy = dFdy(depth);
    float derivative_variance = 0.25 * (dx * dx + dy * dy);

    return vec4(
        depth,
        min(depth2 + derivative_variance, 1.0),
        depth3,
        depth4
    );
}

void main() {
    vec4 base_color = frag_color * material.base_color_factor;

    if (has_base_color_texture()) {
        base_color *= texture(base_color_texture, frag_uv);
    }

    discard_opaque_shadow_alpha(
        material.flags.x,
        material.pbr_alpha.z,
        base_color.a
    );

    out_moments = shadow_moments(gl_FragCoord.z);
}