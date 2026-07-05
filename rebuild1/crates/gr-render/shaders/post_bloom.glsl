#ifndef REBUILD1_POST_BLOOM_GLSL
#define REBUILD1_POST_BLOOM_GLSL

vec3 post_bloom(vec2 uv) {
    float bloom_intensity = clamp(params.bloom.x, 0.0, 2.0);
    if (bloom_intensity <= 0.0) {
        return vec3(0.0);
    }

    return max(texture(bloom_texture, clamp_screen_uv(uv)).rgb, vec3(0.0)) * bloom_intensity;
}

#endif
