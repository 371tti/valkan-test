#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D source_image;

struct BloomParams {
    float4 source_texel;
    float4 params;
};
[[vk::push_constant]] ConstantBuffer<BloomParams> bloom;

float3 bloom_tent_upsample(float2 uv) {
    float2 radius = bloom.source_texel.xy * max(bloom.params.z, 0.25);

    float3 sum = sample_texture(source_image, uv + float2(-radius.x,  radius.y)).rgb;
    sum += sample_texture(source_image, uv + float2( 0.0,       radius.y)).rgb * 2.0;
    sum += sample_texture(source_image, uv + float2( radius.x,  radius.y)).rgb;
    sum += sample_texture(source_image, uv + float2(-radius.x,  0.0)).rgb * 2.0;
    sum += sample_texture(source_image, uv).rgb * 4.0;
    sum += sample_texture(source_image, uv + float2( radius.x,  0.0)).rgb * 2.0;
    sum += sample_texture(source_image, uv + float2(-radius.x, -radius.y)).rgb;
    sum += sample_texture(source_image, uv + float2( 0.0,      -radius.y)).rgb * 2.0;
    sum += sample_texture(source_image, uv + float2( radius.x, -radius.y)).rgb;

    return max(sum * 0.0625, float3(0.0));
}

[shader("fragment")]
void main() {
    out_color = float4(bloom_tent_upsample(frag_uv), 1.0);
}
