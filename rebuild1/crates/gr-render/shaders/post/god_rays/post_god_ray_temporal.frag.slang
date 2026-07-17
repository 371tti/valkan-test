#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D current_rays;
[[vk::binding(1, 0)]] Sampler2D history_rays;

#include "post_god_ray_common.slang"

void neighborhood_bounds(
    float2 uv,
    float3 center_color,
    out float3 min_color,
    out float3 max_color
) {
    float2 texel = params.target.xy;
    float3 c1 = sample_inside(current_rays, uv + float2( texel.x, 0.0)).rgb;
    float3 c2 = sample_inside(current_rays, uv + float2(-texel.x, 0.0)).rgb;
    float3 c3 = sample_inside(current_rays, uv + float2(0.0,  texel.y)).rgb;
    float3 c4 = sample_inside(current_rays, uv + float2(0.0, -texel.y)).rgb;

    min_color = min(center_color, min(min(c1, c2), min(c3, c4)));
    max_color = max(center_color, max(max(c1, c2), max(c3, c4)));
}

[shader("fragment")]
void main() {
    float4 current = sample_inside(current_rays, frag_uv);
    if (params.features.y <= 0.5) {
        out_color = current;
        return;
    }

    float4 history = sample_inside(history_rays, frag_uv);
    float3 min_color;
    float3 max_color;
    neighborhood_bounds(frag_uv, current.rgb, min_color, max_color);

    float3 clamped_history = clamp(history.rgb, min_color * 0.72, max_color * 1.28 + 0.0008);
    float current_luma = luminance_of(current.rgb);
    float history_luma = luminance_of(clamped_history);
    float disocclusion = smoothstep(0.12, 0.65, abs(current_luma - history_luma));
    float history_weight = lerp(0.86, 0.42, disocclusion);

    float3 color = lerp(current.rgb, clamped_history, history_weight);
    float depth = lerp(current.a, history.a, history_weight * 0.35);

    out_color = float4(max(color, float3(0.0)), depth);
}
