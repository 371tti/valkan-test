#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 0)]] Sampler2D source_image;

#include "post_god_ray_common.slang"

float4 sample_soft_lod(float2 uv, float lod) {
    float4 center = sample_inside_lod(source_image, uv, lod);
    float radius = max(lod, 0.0) * 0.85;
    if (radius <= 0.25) {
        return center;
    }

    float2 spread = params.target.xy * radius;
    float4 sum = center * 4.0;
    sum += sample_inside_lod(source_image, uv + float2( spread.x, 0.0), lod);
    sum += sample_inside_lod(source_image, uv + float2(-spread.x, 0.0), lod);
    sum += sample_inside_lod(source_image, uv + float2(0.0,  spread.y), lod);
    sum += sample_inside_lod(source_image, uv + float2(0.0, -spread.y), lod);

    return sum * 0.125;
}

float radial_weight(float t, float directional) {
    float open = smoothstep(0.02, 0.98, t);
    float decay = lerp(exp2(-t * 0.92), exp2(-t * 0.38), directional);

    return open * decay;
}

float4 radial_from_source(float2 uv, float4 source, float4 source_color) {
    if (source.w <= 0.0) {
        return float4(0.0);
    }

    float2 ray = source.xy - uv;
    float ray_distance = god_ray_aspect_distance(uv, source.xy);
    float directional = clamp(source_color.a, 0.0, 1.0);
    float local_shaft = smoothstep(0.015, 0.11, ray_distance) *
        exp2(-ray_distance * 0.82);
    float directional_shaft = smoothstep(source.z * 2.0, source.z * 9.0, ray_distance) * 0.72;
    float shaft = lerp(local_shaft, directional_shaft, directional) *
        god_ray_source_screen_fade(source.xy, directional);

    if (shaft <= 0.0) {
        return float4(0.0);
    }

    const int SAMPLE_COUNT = 64;
    float4 sum = float4(0.0);
    float total = 0.0;

    for (int index = 0; index < SAMPLE_COUNT; index++) {
        float t = (float(index) + 0.5) * (1.0 / float(SAMPLE_COUNT));
        float2 sample_uv = uv + ray * t;
        float lod = t * t * 3.0;
        float weight = radial_weight(t, directional);

        if (!uv_inside(sample_uv)) {
            continue;
        }

        sum += sample_soft_lod(sample_uv, lod) * weight;
        total += weight;
    }

    if (total <= 0.0001) {
        return float4(0.0);
    }

    float4 rays = sum * rcp_safe(total, 0.0001) * shaft;
    rays.rgb = min(rays.rgb, float3(0.65));

    return rays;
}

[shader("fragment")]
void main() {
    float4 rays = radial_from_source(frag_uv, params.source0, params.color0);
    rays += radial_from_source(frag_uv, params.source1, params.color1);

    out_color = max(rays, float4(0.0));
}
