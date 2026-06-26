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

#include "shadow_alpha.glsl"

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
    float alpha = frag_color.a * material.base_color_factor.a;

    discard_opaque_shadow_alpha(
        material.flags.x,
        material.pbr_alpha.z,
        alpha
    );

    out_moments = shadow_moments(gl_FragCoord.z);
}
