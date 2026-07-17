#pragma once

#include "common_math.slang"

static const float GOD_RAY_BACKGROUND_DEPTH = 0.9999;

struct GodRayParams {
    float4 depth;
    float4 target;
    float4 bloom;
    float4 features;
    float4 source0;
    float4 color0;
    float4 source1;
    float4 color1;
};
[[vk::push_constant]] ConstantBuffer<GodRayParams> params;

bool uv_inside(float2 uv) {
    return all(uv >= float2(0.0)) && all(uv <= float2(1.0));
}

float raw_depth_at(Sampler2D depth_texture, float2 uv) {
    if (!uv_inside(uv)) {
        return 1.0;
    }

    return sample_texture(depth_texture, uv).r;
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

float god_ray_aspect_distance(float2 a, float2 b) {
    float2 delta = a - b;
    delta.x *= params.target.z;

    return length(delta);
}

float distance_to_screen_rect(float2 uv) {
    float2 outside = max(max(-uv, uv - float2(1.0)), float2(0.0));
    outside.x *= params.target.z;

    return length(outside);
}

float source_screen_fade(float2 raw_source_uv) {
    return 1.0 - smoothstep(0.04, 0.46, distance_to_screen_rect(raw_source_uv));
}

float directional_source_screen_fade(float2 raw_source_uv) {
    return 1.0 - smoothstep(0.10, 1.65, distance_to_screen_rect(raw_source_uv));
}

float god_ray_source_screen_fade(float2 raw_source_uv, float directional) {
    return lerp(
        source_screen_fade(raw_source_uv),
        directional_source_screen_fade(raw_source_uv),
        clamp(directional, 0.0, 1.0)
    );
}

float4 sample_inside(Sampler2D image, float2 uv) {
    if (!uv_inside(uv)) {
        return float4(0.0);
    }

    return sample_texture(image, uv);
}

float4 sample_inside_lod(Sampler2D image, float2 uv, float lod) {
    if (!uv_inside(uv)) {
        return float4(0.0);
    }

    return sample_texture_level(image, uv, lod);
}
