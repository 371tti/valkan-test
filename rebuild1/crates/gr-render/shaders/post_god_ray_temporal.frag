#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D current_rays;
layout(set = 0, binding = 1) uniform sampler2D history_rays;

#include "post_god_ray_common.glsl"

void neighborhood_bounds(vec2 uv, out vec3 min_color, out vec3 max_color) {
    vec2 texel = params.target.xy;
    vec3 c0 = sample_inside(current_rays, uv).rgb;
    vec3 c1 = sample_inside(current_rays, uv + vec2( texel.x, 0.0)).rgb;
    vec3 c2 = sample_inside(current_rays, uv + vec2(-texel.x, 0.0)).rgb;
    vec3 c3 = sample_inside(current_rays, uv + vec2(0.0,  texel.y)).rgb;
    vec3 c4 = sample_inside(current_rays, uv + vec2(0.0, -texel.y)).rgb;

    min_color = min(c0, min(min(c1, c2), min(c3, c4)));
    max_color = max(c0, max(max(c1, c2), max(c3, c4)));
}

void main() {
    vec4 current = sample_inside(current_rays, frag_uv);
    if (params.features.y <= 0.5) {
        out_color = current;
        return;
    }

    vec4 history = sample_inside(history_rays, frag_uv);
    vec3 min_color;
    vec3 max_color;
    neighborhood_bounds(frag_uv, min_color, max_color);

    vec3 clamped_history = clamp(history.rgb, min_color * 0.72, max_color * 1.28 + 0.0008);
    float current_luma = luminance_of(current.rgb);
    float history_luma = luminance_of(clamped_history);
    float disocclusion = smoothstep(0.12, 0.65, abs(current_luma - history_luma));
    float history_weight = mix(0.86, 0.42, disocclusion);

    vec3 color = mix(current.rgb, clamped_history, history_weight);
    float depth = mix(current.a, history.a, history_weight * 0.35);

    out_color = vec4(max(color, vec3(0.0)), depth);
}
