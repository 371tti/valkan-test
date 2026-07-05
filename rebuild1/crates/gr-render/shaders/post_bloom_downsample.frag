#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D source_image;

layout(push_constant) uniform BloomParams {
    vec4 source_texel;
    vec4 params;
} bloom;

#include "post_bloom_common.glsl"

vec3 bloom_source(vec2 uv) {
    vec3 color = max(texture(source_image, clamp(uv, vec2(0.0), vec2(1.0))).rgb, vec3(0.0));
    if (bloom.params.y > 0.5) {
        color = bloom_extract(color, max(bloom.params.x, 0.001));
    }

    return color;
}

float bloom_firefly_weight(vec3 color) {
    if (bloom.params.y <= 0.5) {
        return 1.0;
    }

    return rcp_safe(1.0 + luminance_of(color), 0.0001);
}

void add_bloom_tap(inout vec3 sum, inout float total_weight, vec2 uv, vec2 offset, float kernel_weight) {
    vec3 color = bloom_source(uv + offset * bloom.source_texel.xy);
    float weight = kernel_weight * bloom_firefly_weight(color);

    sum += color * weight;
    total_weight += weight;
}

vec3 bloom_downsample(vec2 uv) {
    vec3 sum = vec3(0.0);
    float total_weight = 0.0;

    add_bloom_tap(sum, total_weight, uv, vec2(-2.0,  2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, vec2( 0.0,  2.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, vec2( 2.0,  2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, vec2(-2.0,  0.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, vec2( 0.0,  0.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, vec2( 2.0,  0.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, vec2(-2.0, -2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, vec2( 0.0, -2.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, vec2( 2.0, -2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, vec2(-1.0,  1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, vec2( 1.0,  1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, vec2(-1.0, -1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, vec2( 1.0, -1.0), 0.12500);

    return sum * rcp_safe(total_weight, 0.0001);
}

void main() {
    out_color = vec4(bloom_downsample(frag_uv), 1.0);
}
