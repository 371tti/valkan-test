#pragma once

#include "common_math.slang"
#include "post_camera_effects.slang"

static const float POST_BACKGROUND_DEPTH = 0.9999;

float2 clamp_screen_uv(float2 uv) {
    return clamp(uv, float2(0.0), float2(1.0));
}

float3 sample_scene_color(float2 uv) {
    return sample_texture(scene_color, clamp_screen_uv(uv)).rgb;
}

// TAA resolves color onto a stable output grid, while the scene MRTs retain the current
// projection jitter. A raw MRT coordinate r represents stable coordinate r - jitter_ndc / 2,
// therefore post effects must read metadata at stable_uv + jitter_ndc / 2.
float2 post_jitter_ndc() {
    return float2(params.white_balance.w, params.features.w);
}

float2 post_metadata_uv(float2 stable_uv) {
    return clamp_screen_uv(stable_uv + post_jitter_ndc() * 0.5);
}

float depth_at(float2 stable_uv) {
    return sample_texture(scene_depth, post_metadata_uv(stable_uv)).r;
}

float4 opaque_metadata_at(float2 stable_uv) {
    return sample_texture(scene_normal_roughness, post_metadata_uv(stable_uv));
}

float4 transparent_metadata_at(float2 stable_uv) {
    return sample_texture(scene_transparent_normal_roughness, post_metadata_uv(stable_uv));
}

bool is_background_depth(float depth) {
    return depth >= POST_BACKGROUND_DEPTH;
}

float linear_depth(float depth) {
    return params.depth.z * rcp_safe(
        params.depth.y - depth * params.depth.w,
        0.0001
    );
}

float3 view_position(float2 stable_uv, float depth) {
    float z = linear_depth(depth);
    float2 ndc = stable_uv * 2.0 - 1.0;

    return float3(
        ndc.x * z * params.camera.z,
        ndc.y * z * params.camera.w,
        -z
    );
}

float2 project_view_position(float3 position) {
    float inv_depth = rcp_safe(-position.z, 0.0001);

    float2 ndc = float2(
        position.x * params.camera.x * inv_depth,
        position.y * params.camera.y * inv_depth
    );

    return ndc * 0.5 + 0.5;
}

float screen_edge_fade(float2 uv) {
    float2 edge = min(uv, float2(1.0) - uv);
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

bool post_bloom_enabled() {
    return params.bloom.x > 0.0;
}

bool post_god_rays_enabled() {
    return params.god_ray_color1.a > 1.5 ||
        (params.bloom.w > 0.0 &&
            (params.god_ray_source0.w > 0.0 || params.god_ray_source1.w > 0.0));
}

bool post_volumetric_fog_enabled() {
    return params.god_ray_color1.a > 1.5;
}

bool post_transparent_metadata_enabled() {
    return params.features.x > 0.5;
}

uint post_debug_view() {
    return uint(params.features.z + 0.5);
}

bool post_debug_color(float2 uv, out float3 color) {
    uint mode = post_debug_view();
    if (mode == 0u) return false;

    if (mode == 3u) {
        // Scene shading already contains the per-cascade PCSS radius visualization. Bypass tone
        // mapping and post effects so the requested normalized value remains directly readable.
        color = sample_texture(scene_color, frag_uv).rgb;
        return true;
    }

    float depth = depth_at(uv);
    if (mode == 1u) {
        if (is_background_depth(depth)) {
            color = float3(0.015, 0.02, 0.03);
        } else {
            color = oct_decode(opaque_metadata_at(uv).xy) * 0.5 + 0.5;
        }
        return true;
    }
    if (mode == 2u) {
        float normalized = is_background_depth(depth)
            ? 1.0
            : log2(1.0 + linear_depth(depth)) / log2(1.0 + params.depth.y);
        color = float3(saturate(normalized));
        return true;
    }

    return false;
}

bool post_requires_surface_material() {
    return post_ssao_enabled() ||
        post_ssr_enabled();
}

float depth_at_ssr(float2 uv) {
    float2 texel = params.aa.xy;

    float center = depth_at(uv);

    if (is_background_depth(center)) {
        return center;
    }

    float left = depth_at(uv + float2(-texel.x, 0.0));
    float right = depth_at(uv + float2(texel.x, 0.0));
    float up = depth_at(uv + float2(0.0, -texel.y));
    float down = depth_at(uv + float2(0.0, texel.y));

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

    return lerp(stable, center, edge);
}

float depth_at_ssr_trace(float2 uv) {
    return depth_at(uv);
}

float3 oct_decode(float2 encoded) {
    float2 f = encoded * 2.0 - 1.0;
    float3 normal = float3(f, 1.0 - abs(f.x) - abs(f.y));

    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }

    return normalize_fast(normal);
}

struct SurfaceMaterial {
    float3 normal;
    float source_depth;
    float roughness;
    float reflectance;
    float occlusion_weight;
    float transparent_weight;
};

SurfaceMaterial empty_surface_material(float depth) {
    return SurfaceMaterial(
        float3(0.0, 0.0, 1.0),
        depth,
        1.0,
        0.0,
        0.0,
        0.0
    );
}

bool surface_material_is_background(SurfaceMaterial material) {
    return is_background_depth(material.source_depth);
}

SurfaceMaterial opaque_surface_material(float4 packed, float depth) {
    return SurfaceMaterial(
        oct_decode(packed.xy),
        depth,
        clamp(packed.z, 0.04, 1.0),
        clamp(packed.w, 0.0, 1.0),
        1.0,
        0.0
    );
}

SurfaceMaterial transparent_surface_material(float4 packed) {
    float roughness = clamp(packed.z, 0.04, 1.0);

    return SurfaceMaterial(
        oct_decode(packed.xy),
        clamp(packed.w - 1.0, 0.0, POST_BACKGROUND_DEPTH),
        roughness,
        max(0.08, (1.0 - roughness) * 0.28),
        0.25,
        1.0
    );
}

SurfaceMaterial surface_material(float2 uv) {
    uv = clamp_screen_uv(uv);

    float opaque_depth = depth_at(uv);

    if (!post_transparent_metadata_enabled()) {
        if (is_background_depth(opaque_depth)) {
            return empty_surface_material(opaque_depth);
        }

        return opaque_surface_material(
            opaque_metadata_at(uv),
            opaque_depth
        );
    }

    float4 transparent = transparent_metadata_at(uv);
    // The target is RGBA16F: a near device depth packed as 1 + depth can round to exactly 1.
    // Zero is the clear value, so use the validity class instead of comparing against 1.
    bool has_transparent = transparent.w > 0.5;

    if (is_background_depth(opaque_depth)) {
        if (has_transparent) {
            return transparent_surface_material(transparent);
        }

        return empty_surface_material(opaque_depth);
    }

    float4 opaque = opaque_metadata_at(uv);

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

SurfaceMaterial load_feature_surface_material(float2 uv, out bool valid) {
    valid = post_requires_surface_material();

    return valid
        ? surface_material(uv)
        : empty_surface_material(POST_BACKGROUND_DEPTH);
}
