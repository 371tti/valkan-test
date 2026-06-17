#ifndef REBUILD1_POST_COMMON_GLSL
#define REBUILD1_POST_COMMON_GLSL

#include "common_math.glsl"

const float POST_BACKGROUND_DEPTH = 0.9999;

vec2 clamp_screen_uv(vec2 uv) {
    return clamp(uv, vec2(0.0), vec2(1.0));
}

vec3 sample_scene_color(vec2 uv) {
    return texture(scene_color, clamp_screen_uv(uv)).rgb;
}

float depth_at(vec2 uv) {
    return texture(scene_depth, clamp_screen_uv(uv)).r;
}

bool is_background_depth(float depth) {
    return depth >= POST_BACKGROUND_DEPTH;
}

vec3 filmic_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    color = max(color, vec3(0.0));

    vec3 numerator = color * (a * color + b);
    vec3 denominator = color * (c * color + d) + e;

    return saturate(numerator / max(denominator, vec3(0.0001)));
}

vec3 apply_camera_effects(vec3 color) {
    if (params.enabled < 0.5) {
        return max(color, vec3(0.0));
    }

    color = max(color, vec3(0.0));
    color *= params.exposure;
    color *= params.white_balance.rgb;

    color = filmic_tonemap(color);

    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, params.saturation);

    const vec3 contrast_pivot = vec3(0.36);
    color = contrast_pivot + (color - contrast_pivot) * params.contrast;

    return saturate(color);
}

float linear_depth(float depth) {
    return params.depth.z * rcp_safe(
        params.depth.y - depth * params.depth.w,
        0.0001
    );
}

vec3 view_position(vec2 uv, float depth) {
    float z = linear_depth(depth);
    vec2 ndc = uv * 2.0 - 1.0;

    return vec3(
        ndc.x * z * params.camera.z,
        ndc.y * z * params.camera.w,
        -z
    );
}

vec2 project_view_position(vec3 position) {
    float inv_depth = rcp_safe(-position.z, 0.0001);

    vec2 ndc = vec2(
        position.x * params.camera.x * inv_depth,
        position.y * params.camera.y * inv_depth
    );

    return ndc * 0.5 + 0.5;
}

float screen_edge_fade(vec2 uv) {
    vec2 edge = min(uv, vec2(1.0) - uv);
    return smoothstep(0.02, 0.18, min(edge.x, edge.y));
}

bool post_ssao_enabled() {
    return params.ssao.x > 0.0;
}

bool post_ssr_enabled() {
    return params.ssr.x > 0.0 && params.ssr.y >= 1.0;
}

bool post_aa_enabled() {
    return params.aa.w > 0.0;
}

bool post_shadow_softening_enabled() {
    return params.shadow.x > 0.0 && params.shadow.y > 0.0;
}

bool post_requires_surface_material() {
    return post_ssao_enabled() ||
        post_ssr_enabled() ||
        post_shadow_softening_enabled();
}

float depth_at_ssr(vec2 uv) {
    vec2 texel = params.aa.xy;

    float center = depth_at(uv);

    if (is_background_depth(center)) {
        return center;
    }

    float left = depth_at(uv + vec2(-texel.x, 0.0));
    float right = depth_at(uv + vec2(texel.x, 0.0));
    float up = depth_at(uv + vec2(0.0, -texel.y));
    float down = depth_at(uv + vec2(0.0, texel.y));

    float min_depth = min(center, min(min(left, right), min(up, down)));
    float max_depth = max(center, max(max(left, right), max(up, down)));

    float stable = (
        center * 2.0 +
        left +
        right +
        up +
        down
    ) * 0.16666667;

    float depth_range = max_depth - min_depth;

    float edge = smoothstep(
        0.0004,
        0.0045,
        depth_range
    );

    return mix(stable, center, edge);
}

vec3 oct_decode(vec2 encoded) {
    vec2 f = encoded * 2.0 - 1.0;
    vec3 normal = vec3(f, 1.0 - abs(f.x) - abs(f.y));

    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }

    return normalize_fast(normal);
}

struct SurfaceMaterial {
    vec3 normal;
    float source_depth;
    float roughness;
    float reflectance;
    float occlusion_weight;
};

SurfaceMaterial empty_surface_material(float depth) {
    return SurfaceMaterial(
        vec3(0.0, 0.0, 1.0),
        depth,
        1.0,
        0.0,
        0.0
    );
}

bool surface_material_is_background(SurfaceMaterial material) {
    return is_background_depth(material.source_depth);
}

SurfaceMaterial opaque_surface_material(vec4 packed, float depth) {
    return SurfaceMaterial(
        oct_decode(packed.xy),
        depth,
        clamp(packed.z, 0.04, 1.0),
        clamp(packed.w, 0.0, 1.0),
        1.0
    );
}

SurfaceMaterial transparent_surface_material(vec4 packed) {
    float roughness = clamp(packed.z, 0.04, 1.0);

    return SurfaceMaterial(
        oct_decode(packed.xy),
        clamp(packed.w - 1.0, 0.0, POST_BACKGROUND_DEPTH),
        roughness,
        max(0.08, (1.0 - roughness) * 0.28),
        0.25
    );
}

SurfaceMaterial surface_material(vec2 uv) {
    uv = clamp_screen_uv(uv);

    float opaque_depth = depth_at(uv);
    vec4 transparent = texture(scene_transparent_normal_roughness, uv);

    bool has_transparent = transparent.w > 1.0;

    if (is_background_depth(opaque_depth)) {
        if (has_transparent) {
            return transparent_surface_material(transparent);
        }

        return empty_surface_material(opaque_depth);
    }

    vec4 opaque = texture(scene_normal_roughness, uv);

    if (has_transparent) {
        float transparent_depth = clamp(
            transparent.w - 1.0,
            0.0,
            POST_BACKGROUND_DEPTH
        );

        if (transparent_depth <= opaque_depth + 0.00001) {
            return transparent_surface_material(transparent);
        }
    }

    return opaque_surface_material(opaque, opaque_depth);
}

SurfaceMaterial load_feature_surface_material(vec2 uv, out bool valid) {
    valid = post_requires_surface_material();

    return valid
        ? surface_material(uv)
        : empty_surface_material(POST_BACKGROUND_DEPTH);
}

// =========================================================
// Conservative post shadow edge smoothing
//
// This is not a real shadow blur.
// It only performs a small axis-aligned luma stabilization pass.
// Large circular kernels are intentionally avoided because they create
// round blotches when applied to final scene_color.
// =========================================================

const int SHADOW_SOFTEN_TAP_COUNT = 4;

const vec2 SHADOW_SOFTEN_OFFSETS[4] = vec2[4](
    vec2( 1.0,  0.0),
    vec2(-1.0,  0.0),
    vec2( 0.0,  1.0),
    vec2( 0.0, -1.0)
);

float shadow_depth_weight(
    float center_view_depth,
    float sample_view_depth,
    float reject_scale
) {
    float depth_delta = abs(sample_view_depth - center_view_depth);
    float w = rcp_safe(1.0 + depth_delta * reject_scale, 0.0001);
    return w * w;
}

float shadow_normal_weight(vec3 center_normal, vec3 sample_normal) {
    return smoothstep(
        0.84,
        0.995,
        dot(center_normal, sample_normal)
    );
}

float shadow_luma_edge_signal(float min_luma, float max_luma, float limit) {
    float range = max_luma - min_luma;

    return smoothstep(
        max(0.001, limit * 0.08),
        max(0.003, limit * 0.65),
        range
    );
}
vec3 shadow_softened_scene_color(vec2 uv, vec3 color, SurfaceMaterial material) {
    return color;
}
#endif