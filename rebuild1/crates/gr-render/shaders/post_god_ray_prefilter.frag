#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D source_image;

#include "post_god_ray_common.glsl"

void add_tap(inout vec4 sum, inout float total, vec2 uv, vec2 offset, float weight) {
    sum += sample_inside(source_image, uv + offset * params.target.xy) * weight;
    total += weight;
}

void main() {
    vec4 sum = vec4(0.0);
    float total = 0.0;

    add_tap(sum, total, frag_uv, vec2(-1.5, -1.5), 1.0);
    add_tap(sum, total, frag_uv, vec2( 0.0, -1.5), 2.0);
    add_tap(sum, total, frag_uv, vec2( 1.5, -1.5), 1.0);
    add_tap(sum, total, frag_uv, vec2(-1.5,  0.0), 2.0);
    add_tap(sum, total, frag_uv, vec2( 0.0,  0.0), 4.0);
    add_tap(sum, total, frag_uv, vec2( 1.5,  0.0), 2.0);
    add_tap(sum, total, frag_uv, vec2(-1.5,  1.5), 1.0);
    add_tap(sum, total, frag_uv, vec2( 0.0,  1.5), 2.0);
    add_tap(sum, total, frag_uv, vec2( 1.5,  1.5), 1.0);

    out_color = max(sum * rcp_safe(total, 0.0001), vec4(0.0));
}
