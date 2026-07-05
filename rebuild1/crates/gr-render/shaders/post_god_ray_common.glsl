#ifndef REBUILD1_POST_GOD_RAY_COMMON_GLSL
#define REBUILD1_POST_GOD_RAY_COMMON_GLSL

#include "common_math.glsl"

const float GOD_RAY_BACKGROUND_DEPTH = 0.9999;

layout(push_constant) uniform GodRayParams {
    vec4 depth;
    vec4 target;
    vec4 bloom;
    vec4 features;
    vec4 source0;
    vec4 color0;
    vec4 source1;
    vec4 color1;
} params;

bool uv_inside(vec2 uv) {
    return all(greaterThanEqual(uv, vec2(0.0))) &&
        all(lessThanEqual(uv, vec2(1.0)));
}

float raw_depth_at(sampler2D depth_texture, vec2 uv) {
    if (!uv_inside(uv)) {
        return 1.0;
    }

    return texture(depth_texture, uv).r;
}

bool god_ray_background_depth(float depth) {
    return depth >= GOD_RAY_BACKGROUND_DEPTH;
}

float god_ray_linear_depth(float depth) {
    return params.depth.z * rcp_safe(
        params.depth.y - depth * params.depth.w,
        0.0001
    );
}

float god_ray_aspect_distance(vec2 a, vec2 b) {
    vec2 delta = a - b;
    delta.x *= params.target.z;

    return length(delta);
}

float distance_to_screen_rect(vec2 uv) {
    vec2 outside = max(max(-uv, uv - vec2(1.0)), vec2(0.0));
    outside.x *= params.target.z;

    return length(outside);
}

float source_screen_fade(vec2 raw_source_uv) {
    return 1.0 - smoothstep(0.04, 0.46, distance_to_screen_rect(raw_source_uv));
}

float directional_source_screen_fade(vec2 raw_source_uv) {
    return 1.0 - smoothstep(0.10, 1.65, distance_to_screen_rect(raw_source_uv));
}

float god_ray_source_screen_fade(vec2 raw_source_uv, float directional) {
    return mix(
        source_screen_fade(raw_source_uv),
        directional_source_screen_fade(raw_source_uv),
        clamp(directional, 0.0, 1.0)
    );
}

vec4 sample_inside(sampler2D image, vec2 uv) {
    if (!uv_inside(uv)) {
        return vec4(0.0);
    }

    return texture(image, uv);
}

vec4 sample_inside_lod(sampler2D image, vec2 uv, float lod) {
    if (!uv_inside(uv)) {
        return vec4(0.0);
    }

    return textureLod(image, uv, lod);
}

#endif
