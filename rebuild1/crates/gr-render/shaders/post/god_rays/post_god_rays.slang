#pragma once

float4 sample_god_ray_history(float2 uv) {
    if (params.features.y < 0.5) {
        return sample_texture(god_ray_texture_0, clamp_screen_uv(uv));
    }

    return sample_texture(god_ray_texture_1, clamp_screen_uv(uv));
}

float god_ray_depth_similarity(float center_depth, float sample_depth) {
    bool center_background = is_background_depth(center_depth);
    bool sample_background = is_background_depth(sample_depth);
    if (center_background && sample_background) {
        return 1.0;
    }

    if (!center_background && sample_background) {
        return 0.0;
    }

    if (center_background && !sample_background) {
        return 0.35;
    }

    float center_linear = linear_depth(center_depth);
    float sample_linear = linear_depth(sample_depth);

    return exp2(-abs(center_linear - sample_linear) * 0.045);
}

float3 god_ray_bilateral_upscale(float2 uv) {
    float intensity = clamp(params.bloom.w, 0.0, 1.0);
    if (intensity <= 0.0) {
        return float3(0.0);
    }

    float center_depth = depth_at(uv);
    float2 low_texel = params.aa.xy * 4.0;
    float3 sum = float3(0.0);
    float total = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 offset_units = float2(float(x), float(y));
            float2 offset = offset_units * low_texel;
            float2 sample_uv = uv + offset;
            float4 sample_value = sample_god_ray_history(sample_uv);
            float spatial = exp2(-dot(offset_units, offset_units) * 0.72);
            float depth_weight = god_ray_depth_similarity(center_depth, sample_value.a);
            float weight = spatial * depth_weight;

            sum += sample_value.rgb * weight;
            total += weight;
        }
    }

    return max(sum * rcp_safe(total, 0.0001), float3(0.0));
}

float3 post_god_rays(float2 uv) {
    return god_ray_bilateral_upscale(uv);
}

