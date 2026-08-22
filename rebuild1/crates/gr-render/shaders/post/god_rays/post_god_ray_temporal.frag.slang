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
    min_color = center_color;
    max_color = center_color;
    // A full 3x3 neighborhood gives the temporally rotated samples enough support to reject a
    // single bright/deep march layer without preserving a cross-shaped band pattern.
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            float3 sample_color = sample_inside(
                current_rays,
                uv + float2(float(x) * texel.x, float(y) * texel.y)
            ).rgb;
            min_color = min(min_color, sample_color);
            max_color = max(max_color, sample_color);
        }
    }
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

    float clamp_scale = params.features.z > 0.5 ? 0.90 : 0.72;
    float clamp_expand = params.features.z > 0.5 ? 0.00025 : 0.0008;
    float3 clamped_history = clamp(
        history.rgb,
        min_color * clamp_scale,
        max_color * (2.0 - clamp_scale) + clamp_expand
    );
    float current_luma = luminance_of(current.rgb);
    float history_luma = luminance_of(clamped_history);
    float disocclusion = smoothstep(0.12, 0.65, abs(current_luma - history_luma));
    float history_weight = lerp(0.86, 0.42, disocclusion);

    if (params.features.z > 0.5) {
        // The volume is a directional signal, but this low-resolution history has no motion
        // vectors. A camera/light rotation can therefore keep an old shaft at the same screen
        // position even when its intensity is unchanged. Reduce the accumulation when the RGB
        // direction changes, while retaining enough history to hide the 32-step march noise.
        float3 current_direction = normalize(current.rgb + float3(0.0001));
        float3 history_direction = normalize(clamped_history + float3(0.0001));
        float direction_change = 1.0 - saturate(dot(current_direction, history_direction));
        float directional_disocclusion = smoothstep(0.025, 0.24, direction_change);
        float transmittance_change = abs(current.a - history.a);
        float volume_disocclusion = smoothstep(0.035, 0.22, transmittance_change);
        history_weight *= lerp(0.72, 0.30, directional_disocclusion);
        history_weight *= lerp(1.0, 0.32, volume_disocclusion);
        history_weight = min(history_weight, 0.70);
    }

    float3 color = lerp(current.rgb, clamped_history, history_weight);
    float alpha_weight = params.features.z > 0.5
        ? history_weight
        : history_weight * 0.35;
    float depth = lerp(current.a, history.a, alpha_weight);

    out_color = float4(max(color, float3(0.0)), depth);
}
