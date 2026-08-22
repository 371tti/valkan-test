#pragma once

#include "common_math.slang"

static const int SSR_MAX_DDA_STEPS = 64;
static const int SSR_REFINE_STEPS = 4;

struct SsrDdaRay {
    float3 origin;
    float3 q;
    float3 dq;
    float k;
    float dk;
    float2 pixel;
    float2 dp;
    float major_length;
    float stride;
    bool permute;
};

struct SsrDdaSample {
    float2 uv;
    float2 pixel;
    float3 q;
    float k;
    float3 position;
    float ray_view_depth;
};

struct SsrHit {
    float2 uv;
    float3 position;
    float distance;
    float delta;
    float thickness;
};

bool ssr_uv_is_valid(float2 uv) {
    return !(
        uv.x <= 0.0 || uv.x >= 1.0 ||
        uv.y <= 0.0 || uv.y >= 1.0
    );
}

float ssr_hit_slack(float thickness) {
    return max(thickness * 0.42, params.ssr.w * 0.28);
}

float2 ssr_screen_size() {
    return float2(1.0) / max(params.aa.xy, float2(0.000001));
}

float3 ssr_q(float3 position) {
    return position * rcp_safe(-position.z, 0.0001);
}

float ssr_k(float3 position) {
    return rcp_safe(-position.z, 0.0001);
}

float distance_squared(float2 a, float2 b) {
    float2 delta = a - b;
    return dot(delta, delta);
}

void swap_if_greater(inout float a, inout float b) {
    if (a > b) {
        float temp = a;
        a = b;
        b = temp;
    }
}

float3 ssr_position_from_qk(float3 q, float k) {
    return q * rcp_safe(k, 0.000001);
}

SsrDdaSample ssr_dda_sample(
    float2 pixel,
    float3 q,
    float k,
    bool permute
) {
    SsrDdaSample dda_sample;
    dda_sample.pixel = permute ? pixel.yx : pixel;
    dda_sample.uv = dda_sample.pixel / ssr_screen_size();
    dda_sample.q = q;
    dda_sample.k = k;
    dda_sample.position = ssr_position_from_qk(q, k);
    dda_sample.ray_view_depth = -dda_sample.position.z;

    return dda_sample;
}

SsrDdaSample ssr_interpolate_segment(
    float2 pixel0,
    float2 pixel1,
    float3 q0,
    float3 q1,
    float k0,
    float k1,
    bool permute,
    float t
) {
    return ssr_dda_sample(
        lerp(pixel0, pixel1, t),
        lerp(q0, q1, t),
        lerp(k0, k1, t),
        permute
    );
}

float ssr_depth_delta(
    float hit_depth,
    float ray_view_depth,
    out float thickness
) {
    if (is_background_depth(hit_depth)) {
        thickness = 0.0;
        return -100000.0;
    }

    float surface_view_depth = linear_depth(hit_depth);
    float base = max(params.ssr.w, 0.01);

    thickness =
        base +
        surface_view_depth * 0.0015 +
        max(ray_view_depth - surface_view_depth, 0.0) * 0.012;

    thickness = min(thickness, max(base * 2.15, 0.13));

    return ray_view_depth - surface_view_depth;
}

bool ssr_segment_depth_hit(
    float hit_depth,
    float ray_min_view_depth,
    float ray_max_view_depth,
    out float delta,
    out float thickness
) {
    if (is_background_depth(hit_depth)) {
        delta = -100000.0;
        thickness = 0.0;
        return false;
    }

    float surface_view_depth = linear_depth(hit_depth);
    float base = max(params.ssr.w, 0.01);
    float ray_span = max(ray_max_view_depth - ray_min_view_depth, 0.0);

    thickness =
        base +
        surface_view_depth * 0.0015 +
        max(ray_max_view_depth - surface_view_depth, 0.0) * 0.010 +
        ray_span * 0.18;

    thickness = min(thickness, max(base * 2.35, 0.15));

    float front_delta = ray_min_view_depth - surface_view_depth;
    float back_delta = ray_max_view_depth - surface_view_depth;
    float slack = ssr_hit_slack(thickness);

    delta = abs(front_delta) < abs(back_delta)
        ? front_delta
        : back_delta;

    return back_delta >= -slack && front_delta <= thickness * 1.08;
}

SsrHit refined_ssr_hit(
    SsrDdaRay ray,
    float2 low_pixel,
    float2 high_pixel,
    float3 low_q,
    float3 high_q,
    float low_k,
    float high_k
) {
    float low = 0.0;
    float high = 1.0;
    float best_delta = -100000.0;
    float best_thickness = max(params.ssr.w, 0.01);

    for (int i = 0; i < SSR_REFINE_STEPS; i++) {
        float mid = (low + high) * 0.5;
        SsrDdaSample mid_sample = ssr_interpolate_segment(
            low_pixel,
            high_pixel,
            low_q,
            high_q,
            low_k,
            high_k,
            ray.permute,
            mid
        );

        if (!ssr_uv_is_valid(mid_sample.uv)) {
            high = mid;
            continue;
        }

        if (mid_sample.position.z >= -params.depth.x) {
            high = mid;
            continue;
        }

        float hit_depth = depth_at_ssr_trace(mid_sample.uv);
        float thickness;
        float delta = ssr_depth_delta(
            hit_depth,
            mid_sample.ray_view_depth,
            thickness
        );

        best_delta = delta;
        best_thickness = thickness;

        if (delta >= -ssr_hit_slack(thickness)) {
            high = mid;
        } else {
            low = mid;
        }
    }

    SsrDdaSample hit_sample = ssr_interpolate_segment(
        low_pixel,
        high_pixel,
        low_q,
        high_q,
        low_k,
        high_k,
        ray.permute,
        high
    );

    SsrHit hit;
    hit.uv = hit_sample.uv;
    hit.position = hit_sample.position;
    hit.distance = length(hit.position - ray.origin);

    float hit_depth = ssr_uv_is_valid(hit.uv)
        ? depth_at_ssr_trace(hit.uv)
        : POST_BACKGROUND_DEPTH;

    hit.delta = ssr_depth_delta(
        hit_depth,
        hit_sample.ray_view_depth,
        hit.thickness
    );

    if (is_background_depth(hit_depth)) {
        hit.delta = best_delta;
        hit.thickness = best_thickness;
    }

    return hit;
}

float3 ssr_reflection_color(
    float2 uv,
    float2 ray_screen_dir,
    float roughness,
    float distance_blur
) {
    float3 center = sample_scene_color(uv);
    float blur = max(smoothstep(0.42, 0.88, roughness), distance_blur);

    if (blur <= 0.04) {
        return center;
    }

    float2 axis = dot(ray_screen_dir, ray_screen_dir) > 0.00000001
        ? normalize2_fast(ray_screen_dir)
        : float2(1.0, 0.0);

    float2 radius = params.aa.xy * lerp(0.85, 4.25, blur * blur);

    float3 glossy = center * 0.56;
    glossy += sample_scene_color(uv + axis * radius) * 0.22;
    glossy += sample_scene_color(uv - axis * radius) * 0.22;

    return lerp(center, glossy, blur);
}

float ssr_material_weight(
    SurfaceMaterial material,
    float ndotv,
    float view_distance
) {
    float smoothness = 1.0 - material.roughness;
    float fresnel = pow5(1.0 - ndotv);
    float smooth_weight = smoothstep(0.20, 0.82, smoothness);
    float grazing_weight = smoothstep(0.08, 0.88, fresnel);
    float distance_weight = smoothstep(1.2, 12.0, view_distance);

    float base =
        material.reflectance *
        lerp(0.35, 1.10, smooth_weight) *
        lerp(0.45, 1.0, distance_weight);

    float transparent_boost =
        material.transparent_weight *
        lerp(0.04, 0.22, smooth_weight) *
        lerp(0.70, 1.20, grazing_weight);

    float grazing_boost =
        grazing_weight *
        smooth_weight *
        lerp(0.06, 0.20, material.transparent_weight);

    return base + transparent_boost + grazing_boost;
}

// Broad, low-reflectance dielectric highlights are better represented by the
// environment term than by a long, noisy screen-space trace.  Keep transparent
// surfaces and highly reflective materials on the full SSR path.
bool ssr_trace_is_worthwhile(SurfaceMaterial material, float smoothness) {
    if (material.transparent_weight > 0.0) {
        return true;
    }

    float reflective_priority = smoothstep(0.20, 0.55, material.reflectance);
    float minimum_smoothness = lerp(0.18, 0.055, reflective_priority);

    return smoothness > minimum_smoothness;
}

bool build_ssr_dda_ray(
    float3 origin,
    float3 ray_dir,
    float view_distance,
    float max_distance,
    int max_steps,
    out SsrDdaRay ray
) {
    ray = SsrDdaRay(
        float3(0.0),
        float3(0.0),
        float3(0.0),
        0.0,
        0.0,
        float2(0.0),
        float2(0.0),
        0.0,
        0.0,
        false
    );
    float clip_distance = max_distance;

    if (ray_dir.z < -0.0001) {
        clip_distance =
            (params.depth.y - view_distance - 0.002) *
            rcp_safe(-ray_dir.z, 0.0001);
    } else if (ray_dir.z > 0.0001) {
        clip_distance =
            (view_distance - params.depth.x - 0.002) *
            rcp_safe(ray_dir.z, 0.0001);
    }

    if (clip_distance <= 0.0) {
        return false;
    }

    float end_distance = min(max_distance, clip_distance);
    float min_distance = max(0.12, view_distance * 0.006);

    if (end_distance <= min_distance + 0.01) {
        return false;
    }

    float3 start_position = origin + ray_dir * min_distance;
    float3 end_position = origin + ray_dir * end_distance;

    if (
        start_position.z >= -params.depth.x ||
        end_position.z >= -params.depth.x
    ) {
        return false;
    }

    float2 screen_size = ssr_screen_size();
    float2 start_uv = project_view_position(start_position);
    float2 end_uv = project_view_position(end_position);

    if (!ssr_uv_is_valid(start_uv) && !ssr_uv_is_valid(end_uv)) {
        return false;
    }

    float2 start_pixel = start_uv * screen_size;
    float2 end_pixel = end_uv * screen_size;

    if (distance_squared(start_pixel, end_pixel) < 0.0001) {
        end_pixel += float2(0.01, 0.01);
    }

    float3 start_q = ssr_q(start_position);
    float3 end_q = ssr_q(end_position);
    float start_k = ssr_k(start_position);
    float end_k = ssr_k(end_position);
    float2 pixel_delta = end_pixel - start_pixel;
    bool permute = false;

    if (abs(pixel_delta.x) < abs(pixel_delta.y)) {
        permute = true;
        pixel_delta = pixel_delta.yx;
        start_pixel = start_pixel.yx;
        end_pixel = end_pixel.yx;
    }

    float major_length = abs(pixel_delta.x);

    if (major_length <= 1.0) {
        return false;
    }

    float stride = max(1.0, ceil(major_length / float(max_steps)));
    float step_dir = pixel_delta.x < 0.0 ? -1.0 : 1.0;
    float inv_dx = rcp_safe(abs(pixel_delta.x), 0.0001);

    float3 dq = (end_q - start_q) * inv_dx * stride;
    float dk = (end_k - start_k) * inv_dx * stride;
    float2 dp = float2(step_dir, pixel_delta.y * inv_dx) * stride;

    ray = SsrDdaRay(
        origin,
        start_q,
        dq,
        start_k,
        dk,
        start_pixel,
        dp,
        major_length,
        stride,
        permute
    );

    return true;
}

float4 screen_space_reflection(float2 uv, SurfaceMaterial material, float ao) {
    if (
        !post_ssr_enabled() ||
        surface_material_is_background(material) ||
        material.reflectance <= 0.001
    ) {
        return float4(0.0);
    }

    float smoothness = 1.0 - material.roughness;

    if (!ssr_trace_is_worthwhile(material, smoothness)) {
        return float4(0.0);
    }

    float3 origin = view_position(uv, material.source_depth);
    float view_distance = -origin.z;
    float3 view_ray = normalize_fast(origin);
    float ndotv = saturate(dot(material.normal, -view_ray));

    if (ndotv <= 0.0001) {
        return float4(0.0);
    }

    float material_weight = ssr_material_weight(
        material,
        ndotv,
        view_distance
    );

    if (material_weight <= 0.002) {
        return float4(0.0);
    }

    float3 ray_dir = normalize_fast(reflect(view_ray, material.normal));

    // Screen-space reflection becomes ill-conditioned at grazing incidence.
    // At that angle a one-texel normal/depth change can swing the reflected
    // ray completely out of the screen, producing streaks and false hits.
    // Fade those rays before doing the (expensive) DDA traversal.  SSR also
    // only has useful information when the ray travels into the scene
    // (negative view-space Z); rays heading back toward the camera cannot hit
    // a surface represented by the depth buffer reliably.
    float view_angle_mask = smoothstep(0.075, 0.30, ndotv);
    float ray_into_scene_mask = smoothstep(0.015, 0.18, -ray_dir.z);
    float angle_mask = view_angle_mask * ray_into_scene_mask;

    if (angle_mask <= 0.01) {
        return float4(0.0);
    }

    int max_steps = int(clamp(params.ssr.y, 1.0, float(SSR_MAX_DDA_STEPS)));
    float max_distance = max(params.ssr.z, 0.25);
    float3 ray_origin =
        origin +
        material.normal * max(0.010, view_distance * 0.0012);

    SsrDdaRay ray;

    if (
        !build_ssr_dda_ray(
            ray_origin,
            ray_dir,
            view_distance,
            max_distance,
            max_steps,
            ray
        )
    ) {
        return float4(0.0);
    }

    int step_count = int(
        clamp(
            ceil(ray.major_length * rcp_safe(ray.stride, 1.0)),
            1.0,
            float(max_steps)
        )
    );

    float2 dda_pixel = ray.pixel;
    float3 dda_q = ray.q;
    float dda_k = ray.k;
    float origin_edge_fade = screen_edge_fade(uv);
    float pixel_scale = rcp_safe(length(params.aa.xy), 0.000001);

    for (int i = 0; i < step_count; i++) {

        float2 low_pixel = dda_pixel;
        float3 low_q = dda_q;
        float low_k = dda_k;
        SsrDdaSample low_sample = ssr_dda_sample(
            low_pixel,
            low_q,
            low_k,
            ray.permute
        );

        dda_pixel += ray.dp;
        dda_q += ray.dq;
        dda_k += ray.dk;

        SsrDdaSample current_sample = ssr_dda_sample(
            dda_pixel,
            dda_q,
            dda_k,
            ray.permute
        );
        float2 hit_uv = current_sample.uv;

        if (!ssr_uv_is_valid(hit_uv)) {
            break;
        }

        if (current_sample.position.z >= -params.depth.x) {
            break;
        }

        float hit_depth = depth_at_ssr_trace(hit_uv);

        float thickness;
        float depth_delta;
        float ray_min_view_depth = low_sample.ray_view_depth;
        float ray_max_view_depth = current_sample.ray_view_depth;

        swap_if_greater(ray_min_view_depth, ray_max_view_depth);

        bool candidate_hit = ssr_segment_depth_hit(
            hit_depth,
            ray_min_view_depth,
            ray_max_view_depth,
            depth_delta,
            thickness
        );

        if (candidate_hit) {
            SsrHit refined = refined_ssr_hit(
                ray,
                low_pixel,
                dda_pixel,
                low_q,
                dda_q,
                low_k,
                dda_k
            );

            if (!ssr_uv_is_valid(refined.uv)) {
                continue;
            }

            if (
                refined.delta < -ssr_hit_slack(refined.thickness) * 1.35 ||
                refined.delta > refined.thickness * 1.15
            ) {
                continue;
            }

            SurfaceMaterial hit_material = surface_material(refined.uv);

            if (surface_material_is_background(hit_material)) {
                continue;
            }

            float2 ray_screen_dir = refined.uv - uv;
            float pixel_span = length(ray_screen_dir * pixel_scale);
            float same_normal = dot(hit_material.normal, material.normal);
            float same_depth =
                abs(linear_depth(hit_material.source_depth) - view_distance);

            bool likely_same_surface =
                same_normal > lerp(0.965, 0.925, material.transparent_weight) &&
                (
                    pixel_span < lerp(18.0, 7.0, material.transparent_weight) ||
                    same_depth < max(0.12, view_distance * 0.010)
                );

            if (likely_same_surface) {
                continue;
            }

            float hit_facing = saturate(dot(hit_material.normal, -ray_dir));

            if (hit_facing <= 0.035) {
                continue;
            }

            float distance_fade =
                1.0 - smoothstep(
                    max_distance * 0.72,
                    max_distance,
                    refined.distance
                );

            float edge_fade = origin_edge_fade * screen_edge_fade(refined.uv);
            float hit_fade =
                1.0 - smoothstep(
                    refined.thickness * 0.45,
                    refined.thickness * 1.18,
                    abs(refined.delta)
                );
            float self_fade = smoothstep(
                lerp(2.5, 0.85, material.transparent_weight),
                lerp(12.0, 4.0, material.transparent_weight),
                pixel_span
            );
            float roughness_fade =
                1.0 - smoothstep(0.62, 0.94, material.roughness) * 0.72;
            float visibility = lerp(0.68, 1.0, ao);

            float candidate_weight =
                params.ssr.x *
                material_weight *
                angle_mask *
                distance_fade *
                edge_fade *
                hit_fade *
                hit_facing *
                self_fade *
                roughness_fade *
                visibility;

            if (candidate_weight <= 0.0025) {
                continue;
            }

            float distance_blur =
                smoothstep(max_distance * 0.34, max_distance, refined.distance) *
                smoothstep(0.38, 0.82, material.roughness);

            float3 hit_color = ssr_reflection_color(
                refined.uv,
                ray_screen_dir,
                material.roughness,
                distance_blur
            );

            float highlight_lift =
                1.0 + saturate(luminance_of(hit_color) - 0.7) * 0.24;

            return float4(
                hit_color * highlight_lift,
                saturate(candidate_weight * 1.65)
            );
        }
    }

    return float4(0.0);
}

float3 apply_screen_space_reflection(float3 color, float4 reflection) {
    if (reflection.a <= 0.0) {
        return color;
    }

    float reflected_luma = luminance_of(reflection.rgb);
    float base_luma = luminance_of(color);

    float3 glossy = lerp(
        max(color, reflection.rgb * 1.04),
        reflection.rgb + color * 0.12,
        smoothstep(base_luma, base_luma + 1.2, reflected_luma)
    );

    float3 sheen = reflection.rgb * reflection.a * 0.10;

    return lerp(color, glossy, reflection.a) + sheen;
}
