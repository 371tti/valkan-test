#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D source_image;

#include "post_god_ray_common.slang"

void add_tap(inout float4 sum, inout float total, float2 uv, float2 offset, float weight) {
    sum += sample_inside(source_image, uv + offset * params.target.xy) * weight;
    total += weight;
}

[shader("fragment")]
void main() {
    float4 sum = float4(0.0);
    float total = 0.0;

    add_tap(sum, total, frag_uv, float2(-1.5, -1.5), 1.0);
    add_tap(sum, total, frag_uv, float2( 0.0, -1.5), 2.0);
    add_tap(sum, total, frag_uv, float2( 1.5, -1.5), 1.0);
    add_tap(sum, total, frag_uv, float2(-1.5,  0.0), 2.0);
    add_tap(sum, total, frag_uv, float2( 0.0,  0.0), 4.0);
    add_tap(sum, total, frag_uv, float2( 1.5,  0.0), 2.0);
    add_tap(sum, total, frag_uv, float2(-1.5,  1.5), 1.0);
    add_tap(sum, total, frag_uv, float2( 0.0,  1.5), 2.0);
    add_tap(sum, total, frag_uv, float2( 1.5,  1.5), 1.0);

    out_color = max(sum * rcp_safe(total, 0.0001), float4(0.0));
}
