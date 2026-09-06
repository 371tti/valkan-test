#pragma once

// Ground-truth ambient occlusion (GTAO) style screen-space integration.
//
// The original implementation in this file compared a handful of neighbours against the
// receiver depth. That is an obscurance heuristic: it has no hemisphere integration, has a
// strong view-distance dependence, and a single foreground texel can darken a whole radius.
// GTAO instead searches a horizon in a number of 2D slices and analytically integrates the
// cosine-weighted visible arc for each slice. The implementation below keeps the same post
// descriptor contract (current depth + view-space normal) while using a bounded, deterministic
// sample budget selected by params.ssao.w.
// Reference: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion"
// (Activision Research, Algorithm 1 / Equations 7-8).

static const float GTAO_PI = 3.14159265358979323846;
static const float GTAO_HALF_PI = 1.57079632679489661923;
static const float GTAO_GOLDEN_RATIO = 0.6180339887498948482;

float gtao_fraction(float value) {
    return value - floor(value);
}

// A tiny analytic noise tile. A stable per-pixel rotation avoids a regular slice pattern without
// introducing temporal shimmer (the production renderer currently has no AO history buffer).
float gtao_noise(float2 uv, float2 salt) {
    float2 texel = max(params.aa.xy, float2(0.000001));
    float2 pixel = floor(uv / texel);
    float value = gtao_fraction(
        dot(pixel + salt, float2(0.06711056, 0.00583715)) +
        dot(pixel.yx + salt * 1.37, float2(0.754877666, 0.569840291))
    );
    return gtao_fraction(52.9829189 * value);
}

// Fast acos approximation from the GTAO reference implementation. The approximation is smooth
// over [-1, 1] and avoids making the horizon search ALU-heavy on the common path.
float gtao_fast_acos(float value) {
    value = clamp(value, -1.0, 1.0);
    float absolute_value = abs(value);
    float result = (-0.156583 * absolute_value + GTAO_HALF_PI) *
        sqrt(max(1.0 - absolute_value, 0.0));
    return value >= 0.0 ? result : GTAO_PI - result;
}

float gtao_sample_distance_weight(float distance, float radius) {
    // GTAO is a near-field term. Fade the horizon contribution over the outer half of the
    // effect radius instead of abruptly dropping a sample at the boundary.
    float falloff_start = radius * 0.50;
    return saturate((radius - distance) / max(radius - falloff_start, 0.0001));
}

bool gtao_uv_is_valid(float2 uv) {
    return all(uv >= float2(0.0)) && all(uv <= float2(1.0));
}

float gtao_horizon_arc(
    float horizon_cos,
    float normal_phase,
    float projected_normal_cos,
    float horizon_sign
) {
    // Equation (8) integrates the two opposite sides separately. The side represented by
    // horizonCos1 is the negative arc (`h0 = -acos(horizonCos1)` in Algorithm 1); dropping this
    // sign makes tilted receivers over/under-occlude even when no blocker is present.
    float horizon = horizon_sign * gtao_fast_acos(horizon_cos);
    // Keep the horizon inside the slice's valid +/- 90 degree interval. This is the same
    // conservative clamp used by the reference GTAO integration.
    horizon = normal_phase + clamp(
        horizon - normal_phase,
        -GTAO_HALF_PI,
        GTAO_HALF_PI
    );
    return (
        projected_normal_cos +
        2.0 * horizon * sin(normal_phase) -
        cos(2.0 * horizon - normal_phase)
    ) * 0.25;
}

float screen_space_ao(float2 uv, SurfaceMaterial material) {
    if (surface_material_is_background(material)) {
        return 1.0;
    }

    float3 center_position = view_position(uv, material.source_depth);
    float center_view_depth = max(-center_position.z, params.depth.x);
    float3 view_vector = normalize_fast(-center_position);
    float3 surface_normal = normalize_fast(material.normal);

    float effect_radius = clamp(params.ssao.y, 0.05, 2.5);
    float radius_pixels = effect_radius / max(
        center_view_depth * 2.0 * abs(params.camera.z) * max(params.aa.x, 0.000001),
        0.0001
    );
    // Prevent a near-plane pixel from turning a bounded effect into an unbounded texture walk.
    radius_pixels = clamp(radius_pixels, 1.5, 96.0);
    float2 pixel_radius_uv = radius_pixels * max(params.aa.xy, float2(0.000001));
    float min_sample = min(1.0, 1.3 / max(radius_pixels, 1.0));
    float depth_bias = max(params.ssao.z, max(0.0005, center_view_depth * 0.0015));

    int quality_samples = int(clamp(params.ssao.w, 1.0, 8.0));
    int slice_count = quality_samples <= 2 ? 2 : (quality_samples <= 4 ? 3 : 4);
    int steps_per_slice = quality_samples <= 2 ? 2 : (quality_samples <= 4 ? 3 : 4);
    float slice_inverse = rcp_safe(float(slice_count), 1.0);
    float noise_slice = gtao_noise(uv, float2(0.0, 0.0));
    float noise_sample = gtao_noise(uv, float2(17.0, 31.0));
    float visibility = 0.0;

    for (int slice = 0; slice < slice_count; slice++) {
        float slice_fraction = (float(slice) + noise_slice) * slice_inverse;
        float phi = slice_fraction * GTAO_PI;
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);

        // UV y grows downward while view-space y grows upward. The sign flip makes the two
        // screen-space directions below agree with the view-space slice basis.
        float2 screen_direction = float2(cos_phi, -sin_phi);
        float3 direction_vector = float3(cos_phi, sin_phi, 0.0);
        float3 in_plane_direction = direction_vector -
            dot(direction_vector, view_vector) * view_vector;
        float3 axis_vector = normalize_fast(cross(in_plane_direction, view_vector));

        float3 projected_normal = surface_normal -
            axis_vector * dot(surface_normal, axis_vector);
        float projected_normal_length = sqrt(max(dot(projected_normal, projected_normal), 0.000001));
        float projected_normal_cos = saturate(
            dot(projected_normal, view_vector) / projected_normal_length
        );
        float normal_sign = dot(in_plane_direction, projected_normal) >= 0.0 ? 1.0 : -1.0;
        float normal_phase = normal_sign * gtao_fast_acos(projected_normal_cos);

        // These are the visible-hemisphere horizon baselines for the two sides of the slice.
        // Starting from the baselines (instead of a zero/one depth comparison) prevents flat
        // surfaces and back-facing grazing views from becoming artificially dark.
        float low_horizon_cos0 = cos(normal_phase + GTAO_HALF_PI);
        float low_horizon_cos1 = cos(normal_phase - GTAO_HALF_PI);
        float horizon_cos0 = low_horizon_cos0;
        float horizon_cos1 = low_horizon_cos1;

        for (int step_index = 0; step_index < steps_per_slice; step_index++) {
            float sample_noise = gtao_fraction(
                noise_sample +
                (float(slice * steps_per_slice + step_index) * GTAO_GOLDEN_RATIO)
            );
            // A squared radial distribution keeps samples near the receiver where thin crevices
            // carry most of the perceptual AO signal, while still reaching the full radius.
            float sample_fraction = (float(step_index) + sample_noise) /
                float(steps_per_slice);
            sample_fraction = sample_fraction * sample_fraction;
            sample_fraction = min(sample_fraction + min_sample, 1.0);
            float2 sample_offset = screen_direction * pixel_radius_uv * sample_fraction;

            // The data sampler is nearest-neighbour. Snapping the offset to its texel lattice
            // keeps the reconstructed 3D ray and the depth lookup on the same sample.
            float2 sample_offset_pixels = round(
                sample_offset / max(params.aa.xy, float2(0.000001))
            );
            if (abs(sample_offset_pixels.x) + abs(sample_offset_pixels.y) < 0.5) {
                continue;
            }

            float2 texel = max(params.aa.xy, float2(0.000001));
            float2 sample_uv0 = uv + sample_offset_pixels * texel;
            float2 sample_uv1 = uv - sample_offset_pixels * texel;

            // Evaluate both sides of a slice. The paper's outer integral is over [0, pi], so
            // each slice represents a pair of opposite image-space rays.
            float3 sample_position = float3(0.0);
            float3 sample_delta = float3(0.0);
            float sample_distance = 0.0;
            float candidate_cos = 0.0;
            float range_weight = 0.0;

            // Check the UV before touching the clamped depth sampler. Apart from avoiding an
            // unnecessary edge read, this makes the screen-space domain explicit: an off-screen
            // ray contributes the analytic hemisphere baseline, never a duplicated border texel.
            if (gtao_uv_is_valid(sample_uv0)) {
                float sample_depth = depth_at(sample_uv0);
                if (!is_background_depth(sample_depth)) {
                    sample_position = view_position(sample_uv0, sample_depth);
                    sample_delta = sample_position - center_position;
                    sample_distance = sqrt(max(dot(sample_delta, sample_delta), 0.000001));
                    candidate_cos = dot(sample_delta, view_vector) / sample_distance;
                    range_weight = gtao_sample_distance_weight(sample_distance, effect_radius);
                    candidate_cos = lerp(low_horizon_cos0, candidate_cos, range_weight);
                    // A receiver bias only suppresses self-occlusion; it must not attenuate a real
                    // blocker. The depth test is therefore applied after reconstructing the vector.
                    if (linear_depth(sample_depth) + depth_bias > center_view_depth) {
                        candidate_cos = low_horizon_cos0;
                    }
                    // Conservative thin-feature compensation: a farther sample that would lower the
                    // horizon only blends part-way back toward the slice baseline. This follows the
                    // paper's thickness heuristic and avoids one-pixel leaves/silhouettes producing
                    // a full-radius black halo.
                    horizon_cos0 = candidate_cos >= horizon_cos0
                        ? candidate_cos
                        : lerp(horizon_cos0, candidate_cos, 0.24);
                }
            }

            if (gtao_uv_is_valid(sample_uv1)) {
                float sample_depth = depth_at(sample_uv1);
                if (!is_background_depth(sample_depth)) {
                    sample_position = view_position(sample_uv1, sample_depth);
                    sample_delta = sample_position - center_position;
                    sample_distance = sqrt(max(dot(sample_delta, sample_delta), 0.000001));
                    candidate_cos = dot(sample_delta, view_vector) / sample_distance;
                    range_weight = gtao_sample_distance_weight(sample_distance, effect_radius);
                    candidate_cos = lerp(low_horizon_cos1, candidate_cos, range_weight);
                    if (linear_depth(sample_depth) + depth_bias > center_view_depth) {
                        candidate_cos = low_horizon_cos1;
                    }
                    horizon_cos1 = candidate_cos >= horizon_cos1
                        ? candidate_cos
                        : lerp(horizon_cos1, candidate_cos, 0.24);
                }
            }
        }

        horizon_cos0 = max(horizon_cos0, low_horizon_cos0);
        horizon_cos1 = max(horizon_cos1, low_horizon_cos1);
        // Equation (8) of GTAO: integrate the two horizon arcs analytically and weight the slice
        // by the length of the projected normal.
        float arc0 = gtao_horizon_arc(horizon_cos1, normal_phase, projected_normal_cos, -1.0);
        float arc1 = gtao_horizon_arc(horizon_cos0, normal_phase, projected_normal_cos, 1.0);
        visibility += projected_normal_length * (arc0 + arc1);
    }

    visibility *= slice_inverse;
    // A visible surface should retain a small amount of ambient response even in a completely
    // enclosed screen-space corner. This also keeps the later material AO multiplier stable.
    return max(0.03, clamp(visibility, 0.0, 1.0));
}

float screen_space_ao_debug(float2 uv) {
    SurfaceMaterial material = surface_material(uv);
    return screen_space_ao(uv, material);
}
