#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D scene_color;
[[vk::binding(1, 0)]] Sampler2D scene_depth;
[[vk::binding(2, 0)]] Sampler2D scene_transparent_normal_roughness;
[[vk::binding(4, 0)]] Sampler2D taa_linear_depth_0;
[[vk::binding(5, 0)]] Sampler2D taa_linear_depth_1;

#include "post_god_ray_common.slang"
#include "post_god_ray_volume.slang"

float depth_open_visibility(float depth) {
    if (depth >= 0.999995) {
        return 1.0;
    }

    if (depth <= 0.0) {
        return 0.0;
    }

    return smoothstep(0.99992, 0.999985, depth);
}

float2 god_ray_raw_scene_uv(float2 stable_uv) {
    return clamp(
        stable_uv + params.jitter_ndc.xy * 0.5,
        float2(0.0),
        float2(1.0)
    );
}

float device_depth_from_linear_depth(float linear_depth) {
    float safe_depth = max(linear_depth, params.depth.x);
    return saturate(
        (params.depth.y - params.depth.z * rcp_safe(safe_depth, 0.0001)) /
        max(params.depth.w, 0.0001)
    );
}

float stable_depth_at(float2 stable_uv) {
    float2 clamped_uv = clamp(stable_uv, float2(0.0), float2(1.0));
    if (params.jitter_ndc.w > 0.5) {
        // TAA depth history stores linear view depth, while this legacy mask uses device depth
        // for its sky/occluder classification. Reuse the same stable output lattice as post SSR
        // so a Halton phase cannot toggle a foliage pixel between sky and geometry.
        float linear_depth = params.jitter_ndc.z > 0.5
            ? sample_texture(taa_linear_depth_1, clamped_uv).r
            : sample_texture(taa_linear_depth_0, clamped_uv).r;
        return device_depth_from_linear_depth(linear_depth);
    }
    return raw_depth_at(scene_depth, god_ray_raw_scene_uv(clamped_uv));
}

float background_visibility(float2 uv, float center_depth) {
    float2 texel = params.target.xy * 0.72;
    float visibility = depth_open_visibility(center_depth);

    visibility = min(visibility, depth_open_visibility(stable_depth_at(uv + float2( texel.x, 0.0))));
    visibility = min(visibility, depth_open_visibility(stable_depth_at(uv + float2(-texel.x, 0.0))));
    visibility = min(visibility, depth_open_visibility(stable_depth_at(uv + float2(0.0,  texel.y))));
    visibility = min(visibility, depth_open_visibility(stable_depth_at(uv + float2(0.0, -texel.y))));

    return visibility;
}

float transparent_coverage(float2 uv) {
    if (params.features.x <= 0.5) {
        return 1.0;
    }

    float4 transparent = sample_texture(
        scene_transparent_normal_roughness,
        god_ray_raw_scene_uv(uv)
    );
    float has_transparent = step(1.0, transparent.w);

    return lerp(1.0, 0.62, has_transparent);
}

float closest_volume_surface_depth(float2 uv, float center_depth) {
    // The volume target is half-resolution while scene depth is full-resolution. A foliage
    // pixel can contain a leaf on only one of the full-resolution taps, so a majority vote loses
    // the leaf and marches through it to the sky. Use the nearest opaque depth in the same small
    // footprint that the full-resolution bilateral upscale uses. The upscale repeats this exact
    // conservative classification, so a short leaf sample cannot be mistaken for a background
    // sample and spread a thin fog ring into the gap beside it.
    float2 footprint = params.target.xy * 0.45;
    float nearest = center_depth;
    float sample_depth = stable_depth_at(uv + float2( footprint.x, 0.0));
    if (sample_depth < GOD_RAY_BACKGROUND_DEPTH) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = stable_depth_at(uv + float2(-footprint.x, 0.0));
    if (sample_depth < GOD_RAY_BACKGROUND_DEPTH) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = stable_depth_at(uv + float2(0.0,  footprint.y));
    if (sample_depth < GOD_RAY_BACKGROUND_DEPTH) {
        nearest = min(nearest, sample_depth);
    }
    sample_depth = stable_depth_at(uv + float2(0.0, -footprint.y));
    if (sample_depth < GOD_RAY_BACKGROUND_DEPTH) {
        nearest = min(nearest, sample_depth);
    }
    return nearest;
}

float local_source_visibility(float2 uv) {
    float3 color = max(
        sample_texture(scene_color, god_ray_raw_scene_uv(uv)).rgb,
        float3(0.0)
    );
    float luma = luminance_of(color);
    float threshold = max(params.bloom.y * 0.72, 0.12);

    return smoothstep(threshold, threshold * 5.5, luma);
}

struct GodRayMaskContext {
    float sky;
    float local;
    float coverage;
};

bool source_active(float4 source) {
    return source.w > 0.0;
}

bool source_directional(float4 source, float4 source_color) {
    return source_active(source) && source_color.a > 0.5;
}

bool source_local(float4 source, float4 source_color) {
    return source_active(source) && source_color.a <= 0.5;
}

GodRayMaskContext god_ray_mask_context(
    float2 uv,
    float depth,
    bool needs_directional,
    bool needs_local
) {
    float sky = needs_directional
        ? background_visibility(uv, depth)
        : 0.0;
    float local = needs_local
        ? local_source_visibility(uv)
        : 0.0;

    return GodRayMaskContext(sky, local, transparent_coverage(uv));
}

float3 source_mask(
    float2 uv,
    float4 source,
    float4 source_color,
    GodRayMaskContext context
) {
    float strength = source.w;
    if (strength <= 0.0) {
        return float3(0.0);
    }

    float directional = clamp(source_color.a, 0.0, 1.0);
    float dist = god_ray_aspect_distance(uv, source.xy);
    float radius = max(source.z, 0.002);
    float disc = exp2(-dist * dist * rcp_safe(radius * radius, 0.00001) * 2.8);
    float source_fade = god_ray_source_screen_fade(source.xy, directional);

    float directional_air = context.sky * source_fade * 0.075;
    float directional_core = context.sky * source_fade * disc * 0.030;
    float directional_mask = directional_air + directional_core;
    float local_mask = context.local * disc * source_fade;
    float mask = lerp(local_mask, directional_mask, directional) * context.coverage;

    return min(source_color.rgb * mask * strength * params.bloom.w, float3(1.2));
}

[shader("fragment")]
void main() {
    // Every GodRay route is evaluated on the stable low-resolution target.  Sampling the raw
    // jittered depth at `frag_uv` makes foliage/geometry coverage toggle with each Halton phase;
    // undo the projection jitter for the scene lookup before constructing the mask.
    float2 depth_uv = frag_uv;
    float depth = stable_depth_at(frag_uv);

    if (params.features.z > 0.5) {
        depth = closest_volume_surface_depth(frag_uv, depth);
        GodRayVolumeResult volume = god_ray_volumetric_integral(frag_uv, depth);
        out_color = float4(volume.scattering, volume.transmittance);
        return;
    }

    float3 mask = float3(0.0);
    bool has_source = source_active(params.source0) || source_active(params.source1);

    if (has_source) {
        bool needs_directional =
            source_directional(params.source0, params.color0) ||
            source_directional(params.source1, params.color1);
        bool needs_local =
            source_local(params.source0, params.color0) ||
            source_local(params.source1, params.color1);
        GodRayMaskContext context = god_ray_mask_context(
            depth_uv,
            depth,
            needs_directional,
            needs_local
        );

        mask = source_mask(frag_uv, params.source0, params.color0, context);
        mask += source_mask(frag_uv, params.source1, params.color1, context);
    }

    out_color = float4(max(mask, float3(0.0)), depth);
}
