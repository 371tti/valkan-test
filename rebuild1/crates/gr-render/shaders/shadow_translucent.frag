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

layout(push_constant) uniform ShadowCascade {
    uint cascade_index;
} shadow_cascade;

#include "shadow_alpha.glsl"

ivec2 shadow_texel_coord(ivec2 size) {
    return clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - ivec2(1));
}

float opaque_depth_for_cascade(uint index) {
    index = min(index, 2u);
    if (index == 0u) {
        return texelFetch(shadow_cascade_0, shadow_texel_coord(textureSize(shadow_cascade_0, 0)), 0).r;
    }
    if (index == 1u) {
        return texelFetch(shadow_cascade_1, shadow_texel_coord(textureSize(shadow_cascade_1, 0)), 0).r;
    }
    return texelFetch(shadow_cascade_2, shadow_texel_coord(textureSize(shadow_cascade_2, 0)), 0).r;
}

void discard_when_behind_opaque_depth(float fragment_depth) {
    uint index = min(shadow_cascade.cascade_index, 2u);
    float opaque_depth = opaque_depth_for_cascade(index);
    if (fragment_depth > opaque_depth + 0.0008) {
        discard;
    }
}

void main() {
    vec4 base_color = frag_color * material.base_color_factor;
    float alpha = clamp(base_color.a, 0.0, 1.0);
    discard_translucent_shadow_alpha(material.flags.x, alpha);
    discard_when_behind_opaque_depth(gl_FragCoord.z);

    vec3 tint = clamp(base_color.rgb, vec3(0.0), vec3(1.0));
    vec3 transmittance = mix(vec3(1.0), tint, alpha);
    out_transmittance = vec4(transmittance, clamp(gl_FragCoord.z, 0.0, 1.0));
}
