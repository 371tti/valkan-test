#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D source_image;

struct BloomParams {
    float4 source_texel;
    float4 params;
};
[[vk::push_constant]] ConstantBuffer<BloomParams> bloom;

#include "post_bloom_common.slang"

float3 bloom_source(float2 uv) {
    float3 color = max(sample_texture(source_image, clamp(uv, float2(0.0), float2(1.0))).rgb, float3(0.0));
    if (bloom.params.y > 0.5) {
        color = bloom_extract(color, max(bloom.params.x, 0.001));
    }

    return color;
}

float bloom_firefly_weight(float3 color) {
    if (bloom.params.y <= 0.5) {
        return 1.0;
    }

    return rcp_safe(1.0 + luminance_of(color), 0.0001);
}

void add_bloom_tap(inout float3 sum, inout float total_weight, float2 uv, float2 offset, float kernel_weight) {
    float3 color = bloom_source(uv + offset * bloom.source_texel.xy);
    float weight = kernel_weight * bloom_firefly_weight(color);

    sum += color * weight;
    total_weight += weight;
}

float3 bloom_downsample(float2 uv) {
    float3 sum = float3(0.0);
    float total_weight = 0.0;

    add_bloom_tap(sum, total_weight, uv, float2(-2.0,  2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, float2( 0.0,  2.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, float2( 2.0,  2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, float2(-2.0,  0.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, float2( 0.0,  0.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, float2( 2.0,  0.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, float2(-2.0, -2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, float2( 0.0, -2.0), 0.06250);
    add_bloom_tap(sum, total_weight, uv, float2( 2.0, -2.0), 0.03125);
    add_bloom_tap(sum, total_weight, uv, float2(-1.0,  1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, float2( 1.0,  1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, float2(-1.0, -1.0), 0.12500);
    add_bloom_tap(sum, total_weight, uv, float2( 1.0, -1.0), 0.12500);

    return sum * rcp_safe(total_weight, 0.0001);
}

[shader("fragment")]
void main() {
    out_color = float4(bloom_downsample(frag_uv), 1.0);
}
