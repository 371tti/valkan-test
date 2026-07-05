#ifndef REBUILD1_POST_GOD_RAYS_GLSL
#define REBUILD1_POST_GOD_RAYS_GLSL

vec4 sample_god_ray_history(vec2 uv) {
    if (params.features.y < 0.5) {
        return texture(god_ray_texture_0, clamp_screen_uv(uv));
    }

    return texture(god_ray_texture_1, clamp_screen_uv(uv));
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

vec3 god_ray_bilateral_upscale(vec2 uv) {
    float intensity = clamp(params.bloom.w, 0.0, 1.0);
    if (intensity <= 0.0) {
        return vec3(0.0);
    }

    float center_depth = depth_at(uv);
    vec2 low_texel = params.aa.xy * 4.0;
    vec3 sum = vec3(0.0);
    float total = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * low_texel;
            vec2 sample_uv = uv + offset;
            vec4 sample_value = sample_god_ray_history(sample_uv);
            float spatial = exp2(-dot(offset / max(low_texel, vec2(0.0001)), offset / max(low_texel, vec2(0.0001))) * 0.72);
            float depth_weight = god_ray_depth_similarity(center_depth, sample_value.a);
            float weight = spatial * depth_weight;

            sum += sample_value.rgb * weight;
            total += weight;
        }
    }

    return max(sum * rcp_safe(total, 0.0001), vec3(0.0));
}

vec3 post_god_rays(vec2 uv) {
    return god_ray_bilateral_upscale(uv);
}

#endif
