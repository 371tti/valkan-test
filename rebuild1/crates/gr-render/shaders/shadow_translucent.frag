#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 out_transmittance;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    vec4 emissive_occlusion;
    vec4 pbr_alpha;
    uvec4 flags;
} material;

layout(set = 2, binding = 0) uniform sampler2D shadow_cascade_0;
layout(set = 2, binding = 1) uniform sampler2D shadow_cascade_1;
layout(set = 2, binding = 2) uniform sampler2D shadow_cascade_2;
layout(set = 2, binding = 3) uniform sampler2D shadow_cascade_3;

layout(push_constant) uniform ShadowCascade {
    uint cascade_index;
} shadow_cascade;

#include "shadow_alpha.glsl"

ivec2 shadow_texel_coord() {
    return ivec2(gl_FragCoord.xy);
}

float opaque_depth_for_cascade(uint index) {
    index = min(index, 3u);

    if (index == 0u) {
        return texelFetch(
            shadow_cascade_0,
            shadow_texel_coord(),
            0
        ).r;
    }

    if (index == 1u) {
        return texelFetch(
            shadow_cascade_1,
            shadow_texel_coord(),
            0
        ).r;
    }

    if (index == 2u) {
        return texelFetch(
            shadow_cascade_2,
            shadow_texel_coord(),
            0
        ).r;
    }

    return texelFetch(
        shadow_cascade_3,
        shadow_texel_coord(),
        0
    ).r;
}

void discard_when_hidden_by_opaque(float fragment_depth) {
    uint index = min(shadow_cascade.cascade_index, 3u);
    float opaque_depth = opaque_depth_for_cascade(index);

    if (fragment_depth > opaque_depth + 0.00065) {
        discard;
    }
}

vec3 translucent_transmittance(vec3 color, float alpha) {
    color = clamp(color, vec3(0.0), vec3(1.0));
    alpha = clamp(alpha, 0.0, 1.0);

    return mix(vec3(1.0), color, alpha);
}

void main() {
    vec4 base_color = frag_color * material.base_color_factor;
    float alpha = clamp(base_color.a, 0.0, 1.0);

    discard_translucent_shadow_alpha(material.flags.x, alpha);
    discard_when_hidden_by_opaque(gl_FragCoord.z);

    out_transmittance = vec4(
        translucent_transmittance(base_color.rgb, alpha),
        clamp(gl_FragCoord.z, 0.0, 1.0)
    );
}
