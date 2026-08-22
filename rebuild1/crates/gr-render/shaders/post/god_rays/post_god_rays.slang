#pragma once

float4 sample_god_ray_history(float2 uv) {
    if (params.features.y < 0.5) {
        return sample_texture(god_ray_texture_0, clamp_screen_uv(uv));
    }

    return sample_texture(god_ray_texture_1, clamp_screen_uv(uv));
}

float volumetric_surface_depth_at(float2 stable_uv) {
    return depth_at(stable_uv);
}

float volumetric_sample_surface_depth(float2 stable_uv) {
    // Match closest_volume_surface_depth() in the low-resolution mask pass. Without the same
    // conservative footprint here, a low-res sample shortened by a single leaf would still be
    // labelled as background during upscale and would create a bright/thin fog seam beside it.
    float nearest = volumetric_surface_depth_at(stable_uv);
    float2 footprint = params.aa.xy * 4.0 * 0.45;
    float sample_depth = volumetric_surface_depth_at(stable_uv + float2( footprint.x, 0.0));
    if (!is_background_depth(sample_depth)) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = volumetric_surface_depth_at(stable_uv + float2(-footprint.x, 0.0));
    if (!is_background_depth(sample_depth)) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = volumetric_surface_depth_at(stable_uv + float2(0.0,  footprint.y));
    if (!is_background_depth(sample_depth)) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = volumetric_surface_depth_at(stable_uv + float2(0.0, -footprint.y));
    if (!is_background_depth(sample_depth)) {
        nearest = min(nearest, sample_depth);
    }
    return nearest;
}

float god_ray_depth_similarity(float center_depth, float sample_depth, bool volumetric) {
    bool center_background = is_background_depth(center_depth);
    bool sample_background = is_background_depth(sample_depth);
    if (center_background && sample_background) {
        return 1.0;
    }

    if (!volumetric) {
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

    if (center_background && !sample_background) {
        // A low-resolution volume sample that contains geometry must never spread its fog into
        // the adjacent sky/background pixel; preserving that sample makes a visible edge film.
        return 0.0;
    }

    if (!center_background && sample_background) {
        return 0.0;
    }

    float center_linear = linear_depth(center_depth);
    float sample_linear = linear_depth(sample_depth);

    float nearest_depth = max(min(center_linear, sample_linear), 0.5);
    float relative_delta = abs(center_linear - sample_linear) / nearest_depth;
    // The previous absolute-only falloff was effectively permissive at scene scale: a distant
    // foreground surface could retain most of its volume weight across a silhouette. Relative
    // depth rejection keeps same-surface samples while sharply reducing edge membrane bleed.
    return exp2(-relative_delta * 12.0);
}

float4 god_ray_bilateral_upscale(float2 uv) {
    bool volumetric = post_volumetric_fog_enabled();
    float center_depth = depth_at(uv);
    float2 low_texel = params.aa.xy * 4.0;
    float3 sum = float3(0.0);
    float transmittance_sum = 0.0;
    float total = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 offset_units = float2(float(x), float(y));
            float2 offset = offset_units * low_texel;
            float2 sample_uv = uv + offset;
            float4 sample_value = sample_god_ray_history(sample_uv);
            float spatial = exp2(-dot(offset_units, offset_units) * 0.72);
            // Legacy targets store depth in alpha. The volumetric target stores transmittance,
            // so recover the full-resolution scene depth for bilateral weights.
            float sample_depth = volumetric
                ? volumetric_sample_surface_depth(sample_uv)
                : sample_value.a;
            float depth_weight = god_ray_depth_similarity(center_depth, sample_depth, volumetric);
            float weight = spatial * depth_weight;

            sum += sample_value.rgb * weight;
            transmittance_sum += sample_value.a * weight;
            total += weight;
        }
    }

    if (volumetric && total <= 0.0001) {
        // A dense foliage cluster can classify every low-resolution footprint as a foreground
        // leaf while the current full-resolution pixel is a tiny background gap. Do not turn that
        // all-rejected case into black (zero transmittance); retain the center ray's integrated
        // result until a neighboring footprint becomes compatible again.
        float4 fallback = sample_god_ray_history(uv);
        return float4(max(fallback.rgb, float3(0.0)), saturate(fallback.a));
    }

    float3 scattering = max(sum * rcp_safe(total, 0.0001), float3(0.0));
    float transmittance = volumetric
        ? saturate(transmittance_sum * rcp_safe(total, 0.0001))
        : 1.0;
    return float4(scattering, transmittance);
}

float4 post_god_rays(float2 uv) {
    float4 scattering = god_ray_bilateral_upscale(uv);
    return scattering;
}
