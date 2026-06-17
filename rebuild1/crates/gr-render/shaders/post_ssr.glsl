#ifndef REBUILD1_POST_SSR_GLSL
#define REBUILD1_POST_SSR_GLSL

#include "common_math.glsl"

struct SsrHit {
    vec2 uv;
    float distance;
};

vec3 ssr_reflection_color(
    vec2 uv,
    vec2 ray_screen_dir,
    float roughness,
    float distance_blur
) {
    vec3 center = texture(scene_color, uv).rgb;
    float blur = max(smoothstep(0.18, 0.82, roughness), distance_blur);

    if (blur <= 0.001) {
        return center;
    }

    float dir_len2 = dot(ray_screen_dir, ray_screen_dir);

    vec2 axis = dir_len2 > 0.00000001
        ? ray_screen_dir * inversesqrt(dir_len2)
        : vec2(1.0, 0.0);

    vec2 tangent = vec2(-axis.y, axis.x);

    vec2 radius = params.aa.xy * mix(0.75, 7.5, blur * blur);
    vec2 along = axis * radius;
    vec2 across = tangent * radius * 0.62;

    vec3 glossy = center * 0.42;
    glossy += texture(scene_color, clamp(uv + along, vec2(0.0), vec2(1.0))).rgb * 0.16;
    glossy += texture(scene_color, clamp(uv - along, vec2(0.0), vec2(1.0))).rgb * 0.16;
    glossy += texture(scene_color, clamp(uv + across, vec2(0.0), vec2(1.0))).rgb * 0.13;
    glossy += texture(scene_color, clamp(uv - across, vec2(0.0), vec2(1.0))).rgb * 0.13;

    return mix(center, glossy, blur);
}

float ssr_depth_delta(float hit_depth, vec3 ray_position, out float thickness) {
    if (is_background_depth(hit_depth)) {
        thickness = 0.0;
        return -100000.0;
    }

    float ray_view_depth = -ray_position.z;
    float surface_view_depth = linear_depth(hit_depth);

    float base = max(params.ssr.w, 0.01);

    thickness =
        base +
        surface_view_depth * 0.0025 +
        max(ray_view_depth - surface_view_depth, 0.0) * 0.025;

    thickness = min(thickness, max(base * 2.5, 0.16));

    return ray_view_depth - surface_view_depth;
}

SsrHit refined_ssr_hit(
    vec3 ray_origin,
    vec3 ray_dir,
    float low_distance,
    float high_distance,
    float base_thickness,
    float water_weight
) {
    float low = max(low_distance, 0.0);
    float high = max(high_distance, low + 0.0001);

    for (int i = 0; i < 6; i++) {
        float mid = (low + high) * 0.5;
        vec3 mid_position = ray_origin + ray_dir * mid;

        if (mid_position.z >= -params.depth.x) {
            high = mid;
            continue;
        }

        vec2 mid_uv = project_view_position(mid_position);

        if (
            any(lessThan(mid_uv, vec2(0.0))) ||
            any(greaterThan(mid_uv, vec2(1.0)))
        ) {
            high = mid;
            continue;
        }

        float hit_depth = depth_at_ssr(mid_uv);
        float thickness;
        float delta = ssr_depth_delta(hit_depth, mid_position, thickness);

        float negative_slack = mix(
            base_thickness * 0.30,
            thickness * 0.36,
            water_weight
        );

        if (delta >= -negative_slack) {
            high = mid;
        } else {
            low = mid;
        }
    }

    SsrHit hit;
    hit.distance = high;
    hit.uv = project_view_position(ray_origin + ray_dir * high);

    return hit;
}

vec4 screen_space_reflection(vec2 uv, SurfaceMaterial material, float ao) {
    if (
        !post_ssr_enabled() ||
        surface_material_is_background(material) ||
        material.reflectance <= 0.001
    ) {
        return vec4(0.0);
    }

    float smoothness = 1.0 - material.roughness;

    vec3 origin = view_position(uv, material.source_depth);
    vec3 view_ray = normalize_fast(origin);

    float ndotv = saturate(dot(material.normal, -view_ray));

    if (ndotv <= 0.0001) {
        return vec4(0.0);
    }

    float fresnel = pow5(1.0 - ndotv);
    float smooth_weight = smoothstep(0.02, 0.86, smoothness);
    float reflectance = max(material.reflectance, 0.08 + 0.28 * smooth_weight);

    float view_distance = -origin.z;

    float distant_grazing =
        smoothstep(8.0, 80.0, view_distance) *
        smoothstep(0.55, 0.98, 1.0 - ndotv);

    float near_reflection_fade = smoothstep(2.5, 18.0, view_distance);

    float raw_grazing_reflection =
        smoothstep(0.10, 0.92, fresnel) *
        mix(0.35, 1.0, near_reflection_fade);

    float grazing_reflection = max(raw_grazing_reflection, distant_grazing);

    float rough_noise_fade = mix(
        1.0 - smoothstep(0.38, 0.86, material.roughness) * 0.90,
        1.0,
        grazing_reflection * 0.65
    );

    float material_weight =
        mix(
            reflectance * mix(0.18, 1.28, smooth_weight),
            1.0,
            grazing_reflection
        ) *
        mix(0.32, 1.0, near_reflection_fade) *
        rough_noise_fade;

    if (material_weight <= 0.001) {
        return vec4(0.0);
    }

    vec3 ray_dir = normalize_fast(reflect(view_ray, material.normal));

    int max_steps = int(clamp(params.ssr.y, 1.0, 96.0));
    float inv_steps = rcp_safe(float(max_steps), 1.0);

    float water_weight =
        smoothstep(0.72, 0.98, smoothness) *
        smoothstep(0.08, 0.74, 1.0 - ndotv) *
        smoothstep(4.0, 24.0, view_distance);

    float max_distance = max(params.ssr.z, 0.1) * mix(1.0, 1.45, water_weight);
    float base_thickness = max(params.ssr.w, 0.01);
    float min_distance = max(0.08, -origin.z * 0.003);

    vec3 ray_origin =
        origin +
        material.normal * max(0.015, -origin.z * 0.0015);

    float previous_distance = min_distance;
    float previous_delta = -100000.0;
    vec2 previous_uv = uv;
    bool previous_has_depth = false;

    float origin_edge_fade = screen_edge_fade(uv);
    float pixel_scale = rcp_safe(length(params.aa.xy), 0.000001);

    for (int i = 0; i < 96; i++) {
        if (i >= max_steps) {
            break;
        }

        float step_t = (float(i) + 1.0) * inv_steps;

        float step_curve = mix(
            step_t * step_t,
            step_t,
            mix(0.70, 1.0, water_weight)
        );

        float ray_distance = mix(min_distance, max_distance, step_curve);
        vec3 ray_position = ray_origin + ray_dir * ray_distance;

        if (ray_position.z >= -params.depth.x) {
            break;
        }

        vec2 hit_uv = project_view_position(ray_position);

        if (
            any(lessThan(hit_uv, vec2(0.0))) ||
            any(greaterThan(hit_uv, vec2(1.0)))
        ) {
            break;
        }

        float hit_depth = depth_at_ssr(hit_uv);

        if (is_background_depth(hit_depth)) {
            continue;
        }

        float thickness;
        float depth_delta = ssr_depth_delta(hit_depth, ray_position, thickness);

        float negative_slack = mix(
            base_thickness * 0.30,
            thickness * 0.36,
            water_weight
        );

        bool crossed_surface = previous_has_depth
            ? previous_delta < -negative_slack && depth_delta >= -negative_slack
            : depth_delta >= -negative_slack;

        bool near_surface =
            abs(depth_delta) <= thickness * mix(0.75, 1.15, water_weight);

        bool within_surface =
            depth_delta < thickness &&
            depth_delta > -thickness * mix(1.60, 2.20, water_weight);

        bool candidate_surface =
            (crossed_surface && within_surface) ||
            (previous_has_depth && near_surface && previous_delta < thickness);

        if (candidate_surface) {
            float refine_low = previous_has_depth ? previous_distance : min_distance;

            SsrHit refined = refined_ssr_hit(
                ray_origin,
                ray_dir,
                refine_low,
                ray_distance,
                base_thickness,
                water_weight
            );

            vec2 refined_uv = refined.uv;
            float refined_distance = refined.distance;
            vec3 refined_position = ray_origin + ray_dir * refined_distance;

            if (
                any(lessThan(refined_uv, vec2(0.0))) ||
                any(greaterThan(refined_uv, vec2(1.0)))
            ) {
                previous_distance = ray_distance;
                previous_delta = depth_delta;
                previous_uv = hit_uv;
                previous_has_depth = true;
                continue;
            }

            float refined_hit_depth = depth_at_ssr(refined_uv);

            if (is_background_depth(refined_hit_depth)) {
                previous_distance = ray_distance;
                previous_delta = depth_delta;
                previous_uv = hit_uv;
                previous_has_depth = true;
                continue;
            }

            float refined_thickness;
            float refined_delta = ssr_depth_delta(
                refined_hit_depth,
                refined_position,
                refined_thickness
            );

            float refined_negative_slack = mix(
                base_thickness * 0.30,
                refined_thickness * 0.36,
                water_weight
            );

            bool refined_valid =
                refined_delta >= -refined_negative_slack * 1.75 &&
                refined_delta <= refined_thickness * 1.20;

            if (!refined_valid) {
                previous_distance = refined_distance;
                previous_delta = -100000.0;
                previous_uv = refined_uv;
                previous_has_depth = true;
                continue;
            }

            SurfaceMaterial hit_material = surface_material(refined_uv);

            float pixel_span = length((refined_uv - uv) * pixel_scale);

            float close_self_fade = smoothstep(2.0, 10.0, pixel_span);
            float water_self_fade = smoothstep(0.45, 3.0, pixel_span);
            float self_fade = mix(close_self_fade, water_self_fade, water_weight);

            float same_normal = dot(hit_material.normal, material.normal);

            bool likely_same_surface =
                same_normal > 0.94 &&
                pixel_span < mix(42.0, 12.0, water_weight);

            bool too_close_self_hit =
                self_fade <= 0.025;

            if (likely_same_surface || too_close_self_hit) {
                previous_distance = refined_distance;
                previous_delta = -100000.0;
                previous_uv = refined_uv;
                previous_has_depth = true;
                continue;
            }

            float normal_fade =
                saturate(dot(hit_material.normal, -ray_dir) * 0.5 + 0.5);

            float normal_distance_fade =
                1.0 - smoothstep(
                    max_distance * 0.62,
                    max_distance,
                    refined_distance
                );

            float water_distance_fade =
                1.0 - smoothstep(
                    max_distance * 0.90,
                    max_distance,
                    refined_distance
                );

            float distance_fade = mix(
                normal_distance_fade,
                water_distance_fade,
                water_weight
            );

            float hit_fade =
                1.0 - smoothstep(
                    refined_thickness * 0.45,
                    refined_thickness * 1.15,
                    abs(refined_delta)
                );

            float visibility = mix(0.62, 1.0, ao);

            float candidate_weight =
                params.ssr.x *
                material_weight *
                origin_edge_fade *
                screen_edge_fade(refined_uv) *
                distance_fade *
                hit_fade *
                visibility *
                normal_fade *
                self_fade;

            if (candidate_weight <= 0.003) {
                previous_distance = refined_distance;
                previous_delta = -100000.0;
                previous_uv = refined_uv;
                previous_has_depth = true;
                continue;
            }

            vec2 ray_screen_dir = refined_uv - uv;

            float far_water =
                water_weight *
                smoothstep(
                    max_distance * 0.25,
                    max_distance * 0.90,
                    refined_distance
                );

            float rough_material_blur =
                smoothstep(0.45, 0.85, material.roughness) *
                (1.0 - grazing_reflection * 0.45);

            float distance_blur = max(
                far_water * 0.85,
                rough_material_blur * 0.75
            );

            vec3 hit_color = ssr_reflection_color(
                refined_uv,
                ray_screen_dir,
                material.roughness,
                distance_blur
            );

            float highlight_lift =
                1.0 + saturate(luminance_of(hit_color) - 0.6) * 0.42;

            float stability_fade = mix(1.0, 0.76, far_water);

            float weight = candidate_weight * stability_fade;

            return vec4(hit_color * highlight_lift, saturate(weight * 2.15));
        }

        previous_distance = ray_distance;
        previous_delta = depth_delta;
        previous_uv = hit_uv;
        previous_has_depth = true;
    }

    return vec4(0.0);
}

vec3 apply_screen_space_reflection(vec3 color, vec4 reflection) {
    if (reflection.a <= 0.0) {
        return color;
    }

    float reflected_luma = luminance_of(reflection.rgb);
    float base_luma = luminance_of(color);

    vec3 glossy = mix(
        max(color, reflection.rgb * 1.08),
        reflection.rgb + color * 0.16,
        smoothstep(base_luma, base_luma + 1.4, reflected_luma)
    );

    vec3 sheen = reflection.rgb * reflection.a * 0.18;

    return mix(color, glossy, reflection.a) + sheen;
}

#endif