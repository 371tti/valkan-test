#pragma once

// Screen motion follows the renderer-wide temporal convention:
// previous_uv = current_uv + motion_vector.
float2 temporal_previous_uv(float2 current_uv, float2 motion_vector) {
    return current_uv + motion_vector;
}

bool temporal_uv_valid(float2 uv) {
    return all(uv >= float2(0.0)) && all(uv <= float2(1.0));
}

// History images are written on the output lattice, but their color/depth at a pixel still comes
// from the jittered scene sample that produced that frame. The exact previous jittered projection
// is therefore the correct same-world history coordinate. A caller may derive a stable fallback
// coordinate when the exact footprint is disoccluded, but it must not replace the normal path.
float2 temporal_stable_previous_uv(
    float2 matrix_previous_uv,
    float2 current_jitter_pixels,
    float2 previous_jitter_pixels,
    float2 texel_size
) {
    // Keep this helper for the explicit stable-grid fallback used by TAA and pre-TAA consumers.
    // The primary TAA lookup intentionally uses the unmodified matrix projection above.
    return matrix_previous_uv +
        (current_jitter_pixels - previous_jitter_pixels) * texel_size;
}

// Reconstructs a history color continuously over the four texels covered by a sub-pixel
// reprojection while keeping foreground/background surfaces separate.  Picking one texel is
// stable for rejection, but it turns the history itself into a nearest-neighbour signal: as the
// Halton phase crosses a texel edge, the selected center jumps and the volumetric shaft flickers
// in rectangular pixels.  Depth-aware bilinear weights retain the continuous motion coordinate
// on a flat surface and remove only samples belonging to a different depth layer.
float4 temporal_depth_aware_history_sample(
    Sampler2D previous_history_color,
    Sampler2D previous_linear_depth,
    float2 previous_uv,
    float expected_previous_linear_depth,
    float absolute_threshold,
    float relative_threshold,
    bool jitter_only,
    out int2 best_coordinate,
    out float depth_confidence
) {
    int2 dimensions = max(texture_dimensions(previous_linear_depth, 0), int2(1));
    float2 texel_position = previous_uv * float2(dimensions) - float2(0.5);
    float2 texel_floor = floor(texel_position);
    float2 fraction = saturate(texel_position - texel_floor);
    int2 base_coordinate = int2(texel_floor);
    float allowed_difference = max(absolute_threshold, 0.0) +
        abs(expected_previous_linear_depth) * max(relative_threshold, 0.0);
    allowed_difference = max(allowed_difference, 0.000001);

    float4 weighted_color = float4(0.0);
    float weight_sum = 0.0;
    float best_error = 3.402823466e+38;
    float best_spatial_weight = -1.0;
    best_coordinate = clamp(base_coordinate, int2(0), dimensions - int2(1));

    [unroll]
    for (int y = 0; y <= 1; ++y) {
        [unroll]
        for (int x = 0; x <= 1; ++x) {
            int2 coordinate = clamp(
                base_coordinate + int2(x, y),
                int2(0),
                dimensions - int2(1)
            );
            float sampled_depth = load_texel(previous_linear_depth, coordinate, 0).r;
            float error = abs(sampled_depth - expected_previous_linear_depth);
            float spatial_weight =
                (x == 0 ? 1.0 - fraction.x : fraction.x) *
                (y == 0 ? 1.0 - fraction.y : fraction.y);
            float normalized_error = error / allowed_difference;
            // Jitter-only history still needs depth separation.  Treating every tap as compatible
            // at a static edge blends the foreground and background colors into one history texel;
            // the next Halton phase then unblends a different pair and the pixel flashes.  Use a
            // broad *continuous* shoulder for jitter-only reprojection instead of the old binary
            // reject: quantized depth may move a matching tap slightly, while a real neighboring
            // layer remains outside the shoulder.
            float depth_weight = jitter_only
                ? 1.0 - smoothstep(0.35, 2.0, normalized_error)
                : 1.0 - smoothstep(0.30, 1.20, normalized_error);
            float weight = spatial_weight * depth_weight;
            if (weight > 0.0) {
                // The depth footprint and history color share the same jittered raster. Keep the
                // color lookup at the matching texel and use the hardware-linear sampler only for
                // the sub-pixel footprint itself; this prevents a second unrelated layer from
                // entering after depth filtering.
                weighted_color += load_texel(previous_history_color, coordinate, 0) * weight;
                weight_sum += weight;
            }

            bool depth_is_better = error < best_error - 0.000001;
            bool equally_deep_but_closer =
                abs(error - best_error) <= 0.000001 && spatial_weight > best_spatial_weight;
            if (depth_is_better || equally_deep_but_closer) {
                best_error = error;
                best_spatial_weight = spatial_weight;
                best_coordinate = coordinate;
            }
        }
    }

    float normalized_best_error = best_error / allowed_difference;
    depth_confidence = jitter_only
        ? 1.0 - smoothstep(0.70, 2.0, normalized_best_error)
        : 1.0 - smoothstep(0.70, 1.35, normalized_best_error);
    if (weight_sum <= 0.000001) {
        // A completely disoccluded footprint is the one case where the caller must discard the
        // history.  Negative confidence is an explicit sentinel; confidence == 0 remains a valid
        // (but fully reactive) footprint and is blended continuously instead of taking a binary
        // history-valid branch.
        depth_confidence = -1.0;
        return float4(0.0);
    }
    return weighted_color / weight_sum;
}

float temporal_normal_confidence(
    float3 current_normal,
    float3 previous_normal,
    float reject_cosine
) {
    float threshold = clamp(reject_cosine, -1.0, 1.0);
    float alignment = dot(current_normal, previous_normal);
    if (threshold >= 0.999) {
        return alignment >= threshold ? 1.0 : 0.0;
    }
    return smoothstep(threshold, min(threshold + 0.05, 0.999), alignment);
}

bool temporal_depth_valid(
    float sampled_previous_linear_depth,
    float expected_previous_linear_depth,
    float absolute_threshold,
    float relative_threshold
) {
    float allowed_difference = max(absolute_threshold, 0.0) +
        abs(expected_previous_linear_depth) * max(relative_threshold, 0.0);
    return abs(sampled_previous_linear_depth - expected_previous_linear_depth) <=
        allowed_difference;
}

bool temporal_normal_valid(
    float3 current_normal,
    float3 previous_normal,
    float reject_cosine
) {
    return dot(current_normal, previous_normal) >= reject_cosine;
}

float temporal_sign_not_zero(float value) {
    return value < 0.0 ? -1.0 : 1.0;
}

float2 temporal_sign_not_zero(float2 value) {
    return float2(
        temporal_sign_not_zero(value.x),
        temporal_sign_not_zero(value.y)
    );
}

// Scene normal history keeps the scene MRT's octahedral encoding in UNORM8.
float3 temporal_decode_octahedral_normal(float2 encoded) {
    float2 f = encoded * 2.0 - 1.0;
    float3 normal = float3(f, 1.0 - abs(f.x) - abs(f.y));
    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * temporal_sign_not_zero(normal.xy);
    }
    return normalize(normal);
}

// Reconstructs the current world position and projects it through the exact previous jittered
// matrix. expected_previous_linear_depth is in world/view units because perspective clip.w is
// the positive view depth for the renderer's right-handed Vulkan projection.
bool temporal_reproject_world(
    float2 current_uv,
    float current_device_depth,
    float4x4 inverse_current_view_projection,
    float4x4 previous_view_projection,
    out float3 world_position,
    out float2 previous_uv,
    out float expected_previous_linear_depth
) {
    float4 current_clip = float4(current_uv * 2.0 - 1.0, current_device_depth, 1.0);
    float4 world_h = mul(inverse_current_view_projection, current_clip);
    if (abs(world_h.w) <= 0.000001) {
        world_position = float3(0.0);
        previous_uv = current_uv;
        expected_previous_linear_depth = 0.0;
        return false;
    }

    world_position = world_h.xyz / world_h.w;
    float4 previous_clip = mul(previous_view_projection, float4(world_position, 1.0));
    if (previous_clip.w <= 0.000001) {
        previous_uv = current_uv;
        expected_previous_linear_depth = 0.0;
        return false;
    }

    previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    expected_previous_linear_depth = abs(previous_clip.w);
    // The caller removes the current/previous jitter delta before checking the history extent.
    return true;
}
