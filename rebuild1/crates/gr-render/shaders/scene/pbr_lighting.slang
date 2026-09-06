#pragma once

#include "common_math.slang"

static const float PBR_PI = 3.14159265359;
static const float INV_PBR_PI = 0.31830988618;

float3 fresnel_schlick(float cos_theta, float3 f0) {
    return f0 + (1.0 - f0) * pow5(saturate(1.0 - cos_theta));
}

float3 fresnel_schlick_roughness(float cos_theta, float3 f0, float roughness) {
    float3 rough_f0 = max(float3(1.0 - roughness), f0);
    return f0 + (rough_f0 - f0) * pow5(saturate(1.0 - cos_theta));
}

float distribution_ggx(float3 normal, float3 half_vector, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float ndoth = saturate(dot(normal, half_vector));
    float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;

    // Keep the denominator floor tiny so low-roughness highlights stay sharp.
    return alpha2 / max(PBR_PI * denom * denom, 0.0000001);
}

float geometry_schlick_ggx(float ndot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return ndot / max(ndot * (1.0 - k) + k, 0.0001);
}

float geometry_smith(float ndotv, float ndotl, float roughness) {
    return geometry_schlick_ggx(ndotv, roughness) *
        geometry_schlick_ggx(ndotl, roughness);
}

float specular_occlusion(float ndotv, float ao, float roughness) {
    if (ao >= 1.0) {
        return 1.0;
    }

    // A compact Frostbite/Filament-style approximation that avoids crushing metals.
    float exponent = exp2(-16.0 * roughness - 1.0);
    return clamp(pow(ndotv + ao, exponent) - 1.0 + ao, 0.0, 1.0);
}

float3 dielectric_indirect_albedo(float3 base_color, float metallic) {
    float3 floor_albedo = float3(0.022, 0.024, 0.028);
    return lerp(max(base_color, floor_albedo), base_color, metallic);
}

float3 diffuse_multibounce_visibility(float visibility, float3 albedo) {
    float3 a = 2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c = 2.7552 * albedo + 0.6903;
    float3 bounce = ((visibility * a + b) * visibility + c) * visibility;
    return max(float3(visibility), bounce);
}

float3 hemisphere_light(float3 direction) {
    float sky = saturate(direction.y * 0.5 + 0.5);
    float3 ground_color = float3(0.046, 0.039, 0.032);
    float3 horizon_color = float3(0.078, 0.083, 0.096);
    float3 sky_color = float3(0.170, 0.205, 0.270);
    float3 lower = lerp(ground_color, horizon_color, sky);
    float3 upper = lerp(horizon_color, sky_color, sky * sky);
    return lerp(lower, upper, sky);
}

float transparent_specular_boost(float alpha) {
    float transparency = 1.0 - clamp(alpha, 0.0, 1.0);
    return lerp(1.0, rcp_safe(max(alpha, 0.22), 0.22), transparency * 0.55);
}

float shadow_visibility(float3 shadow) {
    return max(max(shadow.r, shadow.g), shadow.b);
}

float3 direct_specular_shadow(float3 shadow, float roughness, float visibility) {
    return shadow * lerp(visibility, 1.0, roughness);
}

float reflected_directional_shadow(
    float3 reflection,
    float3 light,
    float3 shadow,
    float roughness,
    float visibility
) {
    float smoothness = 1.0 - roughness;
    float alignment = saturate(dot(reflection, light));
    float directional_lobe = smoothstep(
        lerp(0.70, 0.965, smoothness),
        1.0,
        alignment
    );
    float occlusion =
        (1.0 - visibility) *
        directional_lobe *
        lerp(0.28, 1.0, smoothness);

    return 1.0 - occlusion;
}

float3 pbr_direct_unshadowed(
    float3 base_color,
    float3 normal,
    float3 view,
    float3 light,
    float3 radiance,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float ndotl
) {
    if (ndotl <= 0.0001) {
        return float3(0.0);
    }

    float3 half_vector = normalize_fast(light + view);
    float vdoth = saturate(dot(view, half_vector));

    float3 fresnel = fresnel_schlick(vdoth, f0);
    float3 diffuse = (float3(1.0) - fresnel) *
        ((1.0 - metallic) * INV_PBR_PI) *
        base_color;

    float distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith(ndotv, ndotl, roughness);
    float3 specular = distribution *
        geometry *
        fresnel *
        rcp_safe(4.0 * ndotv * ndotl, 0.0001);
    specular *= spec_boost;

    return (diffuse + specular) * radiance * ndotl;
}

float local_light_falloff_in_range(float distance_sq, float range_sq) {
    float falloff = 1.0 - distance_sq * rcp_safe(range_sq, 0.0001);
    falloff *= falloff;
    return falloff * range_sq * rcp_safe(distance_sq + range_sq * 0.35, 0.0001);
}

bool local_light_vector(
    float3 sample_position,
    float range_sq,
    out float3 light,
    out float distance_sq,
    out float light_distance
) {
    float3 to_light = sample_position - frag_world_pos;
    distance_sq = dot(to_light, to_light);
    if (distance_sq >= range_sq || distance_sq <= 0.000001) {
        light = float3(0.0);
        light_distance = 0.0;
        return false;
    }

    float inv_distance = rsqrt(distance_sq);
    light_distance = distance_sq * inv_distance;
    light = to_light * inv_distance;
    return true;
}

float area_adjusted_roughness(float roughness, float source_radius, float light_distance) {
    if (source_radius <= 0.0001) {
        return roughness;
    }

    float angular_size = source_radius * rcp_safe(light_distance, 0.0001);
    return clamp(sqrt(roughness * roughness + angular_size * angular_size), roughness, 1.0);
}

float local_shadow_depth_from_distance(float distance, float near_plane, float far_plane) {
    float safe_distance = max(distance, near_plane + 0.0001);
    return (far_plane * (safe_distance - near_plane))
        * rcp_safe(safe_distance * (far_plane - near_plane), 0.0001);
}

float local_shadow_cube_face_distance(float3 direction, float light_distance) {
    // The depth cubemap stores perspective face depth, not radial point-light distance.
    float3 axis = abs(direction);
    float major_axis = max(max(axis.x, axis.y), axis.z);
    return light_distance * major_axis;
}

float local_shadow_sample_depth(int shadow_index, float3 direction) {
    if (shadow_index == 0) {
        return sample_texture(local_shadow_depth[0], direction).r;
    }
    if (shadow_index == 1) {
        return sample_texture(local_shadow_depth[1], direction).r;
    }
    if (shadow_index == 2) {
        return sample_texture(local_shadow_depth[2], direction).r;
    }
    return sample_texture(local_shadow_depth[3], direction).r;
}

float local_shadow_compare(
    int shadow_index,
    float3 direction,
    float compare_depth
) {
    return step(compare_depth, local_shadow_sample_depth(shadow_index, direction));
}

float local_light_visibility(
    int light_index,
    float3 light,
    float light_distance,
    float ndotl,
    float source_radius,
    float shadow_strength
) {
    if (shadow_strength <= 0.0) {
        return 1.0;
    }

    int shadow_index = clamp(light_index, 0, 3);
    float4 shadow_params = frame_camera.local_shadow_params[shadow_index];
    if (shadow_params.x <= 0.5 || light_distance <= 0.0001) {
        return 1.0;
    }

    float3 light_to_receiver = -light;
    float cube_face_distance = local_shadow_cube_face_distance(light_to_receiver, light_distance);
    float near_plane = max(shadow_params.z, 0.001);
    float far_plane = max(shadow_params.w, near_plane + 0.1);
    if (cube_face_distance <= near_plane || cube_face_distance >= far_plane) {
        return 1.0;
    }

    float receiver_depth = local_shadow_depth_from_distance(
        cube_face_distance,
        near_plane,
        far_plane
    );
    float slope_bias = 0.00075 + 0.0028 * (1.0 - clamp(ndotl, 0.0, 1.0));
    float source_bias = clamp(source_radius * rcp_safe(far_plane, 0.0001) * 0.012, 0.0, 0.003);
    float bias = slope_bias + source_bias;

    float3 sample_direction = light_to_receiver;
    float3 basis_axis = abs(sample_direction.z) < 0.999
        ? float3(0.0, 0.0, 1.0)
        : float3(0.0, 1.0, 0.0);
    float3 tangent = normalize_fast(cross(basis_axis, sample_direction));
    float3 bitangent = cross(sample_direction, tangent);
    float diagonal_offset = max(shadow_params.y, 0.0) * 0.70710678;
    float3 tangent_offset = tangent * diagonal_offset;
    float3 bitangent_offset = bitangent * diagonal_offset;
    float compare_depth = receiver_depth - bias;
    float visibility = 0.25 * (
        local_shadow_compare(
            shadow_index,
            sample_direction + tangent_offset + bitangent_offset,
            compare_depth
        ) +
        local_shadow_compare(
            shadow_index,
            sample_direction + tangent_offset - bitangent_offset,
            compare_depth
        ) +
        local_shadow_compare(
            shadow_index,
            sample_direction - tangent_offset + bitangent_offset,
            compare_depth
        ) +
        local_shadow_compare(
            shadow_index,
            sample_direction - tangent_offset - bitangent_offset,
            compare_depth
        )
    );

    return lerp(1.0, clamp(visibility, 0.04, 1.0), clamp(shadow_strength * 1.12, 0.0, 1.0));
}

float3 local_light_resolved_lighting(
    float3 base_color,
    float3 normal,
    float3 view,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float3 light,
    float distance_sq,
    float light_distance,
    float ndotl,
    float3 radiance_color,
    float range_sq,
    float source_radius,
    float shadow_strength,
    float emission_facing,
    float sample_weight,
    int light_index
) {
    if (min(min(emission_facing, sample_weight), ndotl) <= 0.0001) {
        return float3(0.0);
    }

    float falloff = local_light_falloff_in_range(distance_sq, range_sq);
    float visibility = local_light_visibility(
        light_index,
        light,
        light_distance,
        ndotl,
        source_radius,
        shadow_strength
    );
    float local_roughness = area_adjusted_roughness(
        roughness,
        source_radius,
        light_distance
    );
    float3 radiance = radiance_color *
        falloff *
        visibility *
        emission_facing *
        sample_weight;

    return pbr_direct_unshadowed(
        base_color,
        normal,
        view,
        light,
        radiance,
        ndotv,
        f0,
        metallic,
        local_roughness,
        spec_boost,
        ndotl
    );
}

float3 local_light_sample_lighting(
    float3 base_color,
    float3 normal,
    float3 view,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float3 sample_position,
    float3 radiance_color,
    float range_sq,
    float source_radius,
    float shadow_strength,
    float emission_facing,
    float sample_weight,
    int light_index
) {
    float3 light;
    float distance_sq;
    float light_distance;
    if (!local_light_vector(sample_position, range_sq, light, distance_sq, light_distance)) {
        return float3(0.0);
    }

    return local_light_resolved_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost,
        light,
        distance_sq,
        light_distance,
        dot(normal, light),
        radiance_color,
        range_sq,
        source_radius,
        shadow_strength,
        emission_facing,
        sample_weight,
        light_index
    );
}

float spot_light_facing(float3 light_to_receiver, float3 direction, float2 cone_cos) {
    float inner_cos = max(cone_cos.x, cone_cos.y + 0.0001);
    float outer_cos = min(cone_cos.y, inner_cos - 0.0001);
    float angle_cos = dot(direction, light_to_receiver);
    float cone = smoothstep(outer_cos, inner_cos, angle_cos);
    return cone * cone;
}

float3 spot_light_lighting(
    float3 base_color,
    float3 normal,
    float3 view,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float3 position,
    float3 radiance_color,
    float range_sq,
    float3 direction,
    float2 cone_cos,
    float source_radius,
    float shadow_strength,
    int light_index
) {
    float3 light;
    float distance_sq;
    float light_distance;
    if (!local_light_vector(position, range_sq, light, distance_sq, light_distance)) {
        return float3(0.0);
    }

    float facing = spot_light_facing(-light, direction, cone_cos);
    return local_light_resolved_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost,
        light,
        distance_sq,
        light_distance,
        dot(normal, light),
        radiance_color,
        range_sq,
        source_radius,
        shadow_strength,
        facing,
        1.0,
        light_index
    );
}

float3 rectangular_area_light_lighting(
    float3 base_color,
    float3 normal,
    float3 view,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float3 center,
    float3 radiance_color,
    float range_sq,
    float reach_sq,
    float3 direction,
    float source_radius,
    float shadow_strength,
    int light_index
) {
    float3 center_light;
    float center_distance_sq;
    float center_distance;
    if (!local_light_vector(center, reach_sq, center_light, center_distance_sq, center_distance)) {
        return float3(0.0);
    }

    float center_ndotl = saturate(dot(normal, center_light));
    float visibility = local_light_visibility(
        light_index,
        center_light,
        center_distance,
        center_ndotl,
        source_radius,
        shadow_strength
    );

    float facing = saturate(dot(direction, -center_light));
    return local_light_resolved_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost,
        center_light,
        center_distance_sq,
        center_distance,
        center_ndotl,
        radiance_color,
        range_sq,
        source_radius,
        0.0,
        facing,
        visibility,
        light_index
    );
}

float3 emissive_mesh_lighting(
    float3 base_color,
    float3 normal,
    float3 view,
    float ndotv,
    float3 f0,
    float metallic,
    float roughness,
    float spec_boost
) {
    float3 direct = float3(0.0);
    int light_count = int(clamp(frame_camera.emissive_light_count.x, 0.0, 4.0));
    if (light_count <= 0) {
        return direct;
    }

    for (int i = 0; i < light_count; i++) {
        float4 light_position_radius = frame_camera.emissive_light_position_radius[i];
        float4 light_direction_radius = frame_camera.emissive_light_direction_radius[i];
        float4 light_size_kind = frame_camera.emissive_light_size_kind[i];
        float4 light_color = frame_camera.emissive_light_color[i];
        if (light_color.a <= 0.0001) {
            continue;
        }

        float3 radiance_color = light_color.rgb;
        float range = max(light_position_radius.w, 0.001);
        float range_sq = range * range;
        float source_radius = max(light_direction_radius.w, 0.0);
        float kind = light_size_kind.z;
        float shadow_strength = light_size_kind.w;

        if (kind > 2.5) {
            float2 half_size = max(light_size_kind.xy, float2(0.001));
            float rect_source_radius = max(source_radius, length(half_size));
            float reach = range + rect_source_radius;
            direct += rectangular_area_light_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                reach * reach,
                light_direction_radius.xyz,
                rect_source_radius,
                shadow_strength,
                i
            );
        } else if (kind > 1.5) {
            direct += spot_light_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                light_direction_radius.xyz,
                light_size_kind.xy,
                source_radius,
                shadow_strength,
                i
            );
        } else {
            direct += local_light_sample_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                source_radius,
                shadow_strength,
                1.0,
                1.0,
                i
            );
        }
    }

    return direct;
}

float3 pbr_shadow(
    float3 shadow_normal,
    float3 light,
    float2 shadow_sample_seed,
    out float opaque_visibility,
    out float3 transmittance
) {
    float camera_depth = frag_view_depth;
    return shadow_factor(
        shadow_depth_raw,
        shadow_depth,
        translucent_shadow_0,
        translucent_shadow_1,
        translucent_shadow_2,
        translucent_shadow_3,
        frame_camera.shadow_view_proj,
        frag_world_pos,
        shadow_sample_seed,
        shadow_normal,
        light,
        camera_depth,
        frame_camera.shadow_cascade_splits,
        frame_camera.shadow_cascade_texel_world,
        frame_camera.shadow_cascade_depth_span,
        frame_camera.shadow_cascade_resolution,
        frame_camera.stable_csm_pcss_params,
        frame_camera.stable_csm_receiver_params,
        frame_camera.light_color.w,
        opaque_visibility,
        transmittance
    );
}

// The visibility history stores the scalar in R and the previous receiver depth in G. Rebuild a
// depth-aware bilinear footprint explicitly instead of asking the linear sampler to average the
// two channels first: at an oblique shadow edge, a foreground and a background texel can be at
// nearly the same screen-space distance while carrying unrelated PCSS visibility values.
float4 pcss_depth_aware_history(
    float2 previous_uv,
    float expected_previous_depth,
    out float depth_confidence
) {
    int2 dimensions = max(texture_dimensions(pcss_shadow_history, 0), int2(1));
    float2 texel_position = previous_uv * float2(dimensions) - float2(0.5);
    float2 texel_floor = floor(texel_position);
    int2 base_coordinate = int2(texel_floor);
    float2 fraction = saturate(texel_position - texel_floor);
    float depth_tolerance = max(0.01, abs(expected_previous_depth) * 0.006);

    float4 weighted_history = float4(0.0);
    float weight_sum = 0.0;
    [unroll]
    for (int y = 0; y <= 1; ++y) {
        [unroll]
        for (int x = 0; x <= 1; ++x) {
            int2 coordinate = clamp(
                base_coordinate + int2(x, y),
                int2(0),
                dimensions - int2(1)
            );
            float spatial_weight =
                (x == 0 ? 1.0 - fraction.x : fraction.x) *
                (y == 0 ? 1.0 - fraction.y : fraction.y);
            float4 candidate = load_texel(pcss_shadow_history, coordinate, 0);
            float candidate_depth_confidence = 1.0 - smoothstep(
                depth_tolerance,
                depth_tolerance * 4.0,
                abs(candidate.g - expected_previous_depth)
            );
            float weight = spatial_weight * candidate_depth_confidence;
            weighted_history += candidate * weight;
            weight_sum += weight;
        }
    }

    depth_confidence = saturate(weight_sum);
    if (weight_sum <= 0.0001) {
        depth_confidence = 0.0;
        return float4(0.0);
    }
    return weighted_history / weight_sum;
}

// Temporal filtering is intentionally limited to the directional PCSS visibility scalar. The
// BRDF, transmittance, local lights, and emissive terms remain current-frame values, so shadow
// history cannot smear highlights or alter the energy of unrelated lighting paths.
float pcss_temporal_visibility(float current_visibility) {
#ifdef SCENE_FAST_PATH
    return current_visibility;
#else
    if (frame_camera.pcss_temporal.x <= 0.5 || frame_camera.pcss_temporal.y <= 0.0001) {
        return current_visibility;
    }
    float3 world_position = frag_world_pos;
    float4 previous_clip = mul(
        frame_camera.previous_view_projection,
        float4(world_position, 1.0)
    );
    if (previous_clip.w <= 0.0001) {
        return current_visibility;
    }
    float2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    if (!all(previous_uv >= float2(0.0)) || !all(previous_uv <= float2(1.0))) {
        return current_visibility;
    }
    // The visibility history belongs to the previous frame's rasterization. Reproject the
    // receiver into that frame before sampling; using the current pixel coordinate here would
    // blur a moving shadow in screen space and is especially visible when the light and camera
    // approach a right angle.
    float expected_previous_depth = abs(previous_clip.w);
    float depth_confidence;
    float4 history = pcss_depth_aware_history(
        previous_uv,
        expected_previous_depth,
        depth_confidence
    );
    if (depth_confidence <= 0.0001) {
        return current_visibility;
    }
    // Keep PCSS history strong enough to suppress finite-tap flicker, but let a moving Sun reduce
    // retention so the old visibility scalar does not trail across the new shadow edge.
    float camera_or_light_reactivity = saturate(frame_camera.pcss_temporal.w);
    // A visibility difference is exactly the signal temporal PCSS is meant to suppress: finite
    // blocker/filter taps can toggle a receiver between two values even while its world/depth
    // reprojection is valid. The old fixed `current +/- 0.5` clamp prevented a static receiver
    // from ever converging (a 0/1 sequence was forced to alternate around 0.5). Use the full
    // visibility range while the camera and Sun are stationary; only tighten the clamp when the
    // reprojection is reactive enough that a shadow edge may genuinely have moved.
    float visibility_history_delta = lerp(1.0, 0.25, camera_or_light_reactivity);
    float bounded_history = clamp(
        history.r,
        current_visibility - visibility_history_delta,
        current_visibility + visibility_history_delta
    );
    // The CPU packs ordinary light motion at half strength, so a moving Sun still keeps useful
    // temporal averaging. Camera motion remains fully reactive in the same lane; a true light cut
    // disables the history before this shader is reached.
    float feedback = frame_camera.pcss_temporal.y * depth_confidence *
        (1.0 - camera_or_light_reactivity * 0.35);
    return saturate(lerp(current_visibility, bounded_history, feedback));
#endif
}

bool scene_lighting_disabled() {
    bool directional =
        dot(frame_camera.light_color.rgb, frame_camera.light_color.rgb) > 0.000001;
    bool environment =
        frame_camera.ambient_color.w > 0.000001 ||
        dot(frame_camera.ambient_color.rgb, frame_camera.ambient_color.rgb) > 0.000001;
#ifdef SCENE_FAST_PATH
    return !directional && !environment;
#else
    bool local = frame_camera.emissive_light_count.x > 0.5;
    return !directional && !environment && !local;
#endif
}

// The scene render pass keeps the PCSS attachment bound for every metadata/debug view. The
// backend uses this alpha as a write gate: normal shading replaces the cleared history sample,
// while diagnostics leave the LOADed previous image untouched so raw/history comparisons do not
// feed their own visualization back into the temporal buffer.
float pcss_history_output_alpha() {
    return frame_camera.debug_view.x < 0.5 ? 1.0 : 0.0;
}

float3 shade_pbr(
    float3 base_color,
    float3 normal,
    float3 shadow_normal,
    float metallic,
    float roughness,
    float occlusion,
    float3 emissive,
    float alpha,
    float2 shadow_sample_seed,
    out float pcss_visibility
) {
    pcss_visibility = 1.0;
    bool debug_filter_radius = frame_camera.debug_view.x > 2.5 && frame_camera.debug_view.x < 3.5;
    bool debug_pcss_visibility_raw = frame_camera.debug_view.x > 13.5 &&
        frame_camera.debug_view.x < 14.5;
    bool debug_pcss_history = frame_camera.debug_view.x > 14.5 &&
        frame_camera.debug_view.x < 15.5;
    if (debug_pcss_history) {
        // The graph disables the history attachment write while this view is active, so this is
        // the previous frame's stored value rather than a value produced by the current debug
        // shading. It exposes whether temporal integration is actually receiving valid data.
        int2 dimensions = max(texture_dimensions(pcss_shadow_history, 0), int2(1));
        // SV_Position is already at the pixel center (x+0.5,y+0.5). Do not add another half
        // texel here; that would make this diagnostic average the four neighboring history
        // pixels and hide the reprojection/accumulation behavior it is meant to expose.
        float2 history_uv = shadow_sample_seed / float2(dimensions);
        float history_visibility = sample_texture_level(
            pcss_shadow_history,
            saturate(history_uv),
            0.0
        ).r;
        pcss_visibility = history_visibility;
        return history_visibility.xxx;
    }
    if (debug_filter_radius || debug_pcss_visibility_raw) {
        float debug_visibility;
        float3 debug_transmittance;
        if (frame_camera.light_dir.w > 0.5) {
            float3 debug_shadow = pbr_shadow(
                shadow_normal,
                -frame_camera.light_dir.xyz,
                shadow_sample_seed,
                debug_visibility,
                debug_transmittance
            );
            if (debug_pcss_visibility_raw) {
                // This branch intentionally bypasses pcss_temporal_visibility. It is the A/B
                // reference for deciding whether close-up noise is produced by spatial PCSS or
                // by the visibility-history reprojection/feedback path.
                pcss_visibility = debug_visibility;
                return debug_visibility.xxx;
            }
            pcss_visibility = pcss_temporal_visibility(debug_visibility);
            return debug_shadow;
        }
        return float3(0.0);
    }
    base_color = max(base_color, float3(0.0));
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    occlusion = clamp(occlusion, 0.0, 1.0);
    float mirror_surface =
        smoothstep(0.92, 0.995, metallic) *
        (1.0 - smoothstep(0.045, 0.085, roughness));
    float direct_lighting_weight = 1.0 - mirror_surface;
    float3 view = normalize_fast(frame_camera.camera_pos.xyz - frag_world_pos);
    float ndotv = max(dot(normal, view), 0.0001);

    float3 f0 = lerp(float3(0.04), base_color, metallic);
    float spec_boost = lerp(transparent_specular_boost(alpha), 1.0, metallic);
    float smoothness = 1.0 - roughness;

    float3 direct = float3(0.0);
    float3 light = float3(0.0, 1.0, 0.0);
    float directional_visibility = 1.0;
    float3 directional_transmittance = float3(1.0);
    bool has_directional_light =
        dot(frame_camera.light_color.rgb, frame_camera.light_color.rgb) > 0.000001;

    if (has_directional_light) {
        light = -frame_camera.light_dir.xyz;
        float ndotl = saturate(dot(normal, light));

        if (ndotl > 0.0001) {
            // `light_dir.w` reports shadow-map availability, not whether the light exists. A
            // caster-free scene still receives direct light and skips all cascade samples.
            if (frame_camera.light_dir.w > 0.5 && direct_lighting_weight > 0.0001) {
                pbr_shadow(
                    shadow_normal,
                    light,
                    shadow_sample_seed,
                    directional_visibility,
                    directional_transmittance
                );
                directional_visibility = pcss_temporal_visibility(
                    directional_visibility
                );
                pcss_visibility = directional_visibility;
            } else {
                directional_visibility = 1.0;
            }

            if (direct_lighting_weight > 0.0001) {
                float3 half_vector = normalize_fast(light + view);
                float vdoth = saturate(dot(view, half_vector));

                float3 fresnel = fresnel_schlick(vdoth, f0);
                float3 diffuse = (float3(1.0) - fresnel) *
                    ((1.0 - metallic) * INV_PBR_PI) *
                    base_color;

                float distribution = distribution_ggx(normal, half_vector, roughness);
                float geometry = geometry_smith(ndotv, ndotl, roughness);
                float3 specular = distribution *
                    geometry *
                    fresnel *
                    rcp_safe(4.0 * ndotv * ndotl, 0.0001);
                specular *= spec_boost;

                float3 directional_radiance = (
                        diffuse + specular
                    ) *
                    directional_transmittance *
                    frame_camera.light_color.rgb *
                    ndotl;
                direct = directional_radiance * directional_visibility;
            }
        }
    }
    if (direct_lighting_weight > 0.0001) {
#ifndef SCENE_FAST_PATH
        direct += emissive_mesh_lighting(
            base_color,
            normal,
            view,
            ndotv,
            f0,
            metallic,
            roughness,
            spec_boost
        );
#endif
        direct *= direct_lighting_weight;
    }

    float indirect_strength = max(frame_camera.ambient_color.w, 0.0);
    float ambient_energy = dot(
        frame_camera.ambient_color.rgb,
        frame_camera.ambient_color.rgb
    );
    if (indirect_strength <= 0.000001 && ambient_energy <= 0.000001) {
        return direct + emissive;
    }

    float3 diffuse_irradiance = frame_camera.ambient_color.rgb;
    if (indirect_strength > 0.000001) {
        diffuse_irradiance += hemisphere_light(normal) * indirect_strength;
    }

    float3 indirect_diffuse = float3(0.0);
    if (direct_lighting_weight > 0.0001 && metallic < 0.9999) {
        float3 indirect_albedo = dielectric_indirect_albedo(base_color, metallic);
        indirect_diffuse =
            indirect_albedo *
            (1.0 - metallic) *
            diffuse_irradiance *
            diffuse_multibounce_visibility(occlusion, indirect_albedo) *
            direct_lighting_weight;
    }

    float3 reflection = reflect(-view, normal);
    float3 specular_env = frame_camera.ambient_color.rgb * 0.35;
    if (indirect_strength > 0.000001) {
        specular_env += hemisphere_light(reflection) * indirect_strength;
    }

    float3 indirect_specular = float3(0.0);
    if (dot(specular_env, specular_env) > 0.000001) {
        indirect_specular =
            specular_env *
            fresnel_schlick_roughness(ndotv, f0, roughness) *
            (0.18 + smoothness * smoothness * 0.82) *
            specular_occlusion(ndotv, occlusion, roughness) *
            spec_boost;
    }

    return indirect_diffuse + indirect_specular + direct + emissive;
}

float material_reflectance(float3 base_color, float metallic, float roughness) {
    // This is a visibility weight for post SSR, not strict physical F0.
    float conductor = max(max(base_color.r, base_color.g), base_color.b);
    float dielectric = lerp(0.14, 0.055, roughness);
    return clamp(lerp(dielectric, conductor, metallic), 0.0, 1.0);
}

float2 oct_encode_unit(float3 normal) {
    normal *= rcp_safe(abs(normal.x) + abs(normal.y) + abs(normal.z), 0.0001);
    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }
    return normal.xy * 0.5 + 0.5;
}

float4 pack_view_normal_material(float3 normal, float roughness, float metallic, float3 base_color) {
    float3 view_normal = mul(frame_camera.view, float4(normal, 0.0)).xyz;
    return float4(
        oct_encode_unit(view_normal),
        roughness,
        material_reflectance(base_color, metallic, roughness)
    );
}
