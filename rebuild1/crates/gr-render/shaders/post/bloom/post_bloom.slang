#pragma once

float3 post_bloom(float2 uv) {
    float bloom_intensity = clamp(params.bloom.x, 0.0, 2.0);
    if (bloom_intensity <= 0.0) {
        return float3(0.0);
    }

    return max(sample_texture(bloom_texture, clamp_screen_uv(uv)).rgb, float3(0.0)) * bloom_intensity;
}

